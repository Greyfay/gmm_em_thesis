# implementation/_torch_gmm_em.py
"""Gaussian Mixture Model (GMM) EM in PyTorch (sklearn-aligned).

This is a correctness/reference implementation meant to be as close as practical to
scikit-learn's GaussianMixture behavior while remaining in PyTorch.

Key alignment choices:
- We store covariances_ but compute and USE precisions_cholesky_ for E-step log-probs
  (like sklearn), which improves numerical stability on ill-conditioned covariances.
- reg_covar is ADDED (not clamped) to the diagonal in the M-step.
- nk smoothing uses: nk = resp.sum(0) + 10 * eps(dtype), like sklearn.
- Default init_params is 'kmeans' (sklearn default), implemented as k-means++ seeding
  followed by a few Lloyd iterations.
- Supports 'k-means++' init explicitly.
- Supports user-supplied init: weights_init, means_init, precisions_init.
- Supports sklearn-based initialization: 'scikit_kmeans' and 'scikit_random' to match
  sklearn's initialization exactly, helping align results between implementations.

Initialization options:
- 'kmeans': PyTorch k-means++ followed by Lloyd iterations
- 'k-means++': PyTorch k-means++ seeding only (no refinement)
- 'random': PyTorch random uniform responsibilities
- 'random_from_data': Random samples from data as initial means
- 'scikit_kmeans': Use sklearn's KMeans for initialization (matches sklearn exactly)
- 'scikit_random': Use sklearn's random initialization (matches sklearn exactly)

Covariance storage formats:
- spherical: cov shape (K,)              # one variance per component
- diag:      cov shape (K, D)            # per-component per-dimension variances
- tied:      cov shape (D, D)            # one shared full covariance
- full:      cov shape (K, D, D)         # full covariance per component

Exposed sklearn-like attributes after fit:
- weights_, means_, covariances_
- precisions_, precisions_cholesky_
- converged_, n_iter_, lower_bound_
- lower_bounds_ (history list; handy for thesis/debugging)

Notes:
- Some parts (full/tied log-prob, KMeans) use small Python loops for clarity.
  You can later replace them with batched GPU kernels while keeping the API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans, kmeans_plusplus


# ---------------------------
# Utilities
# ---------------------------

def _check_cov_type(cov_type: str) -> None:
    if cov_type not in ("spherical", "diag", "tied", "full"):
        raise ValueError(f"Unknown covariance_type={cov_type!r}")

def _nk_eps(dtype: torch.dtype) -> float:
    """Match sklearn's nk smoothing: 10 * machine epsilon for dtype."""
    return float(10.0 * torch.finfo(dtype).eps)

def _safe_log(x: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(x.dtype).tiny
    return torch.log(x.clamp_min(tiny))


def _add_reg_diag(cov: torch.Tensor, reg_covar: float) -> torch.Tensor:
    """Add reg_covar to diagonal (works for (D,D) or (K,D,D))."""
    if reg_covar == 0.0:
        return cov
    if cov.dim() == 2:
        D = cov.shape[0]
        return cov + reg_covar * torch.eye(D, device=cov.device, dtype=cov.dtype)
    if cov.dim() == 3:
        _, D, _ = cov.shape
        eye = torch.eye(D, device=cov.device, dtype=cov.dtype)
        return cov + reg_covar * eye.unsqueeze(0)
    raise ValueError("cov must be (D,D) or (K,D,D)")


# ---------------------------
# Precision-Cholesky helpers (sklearn-style)
# ---------------------------

@torch.no_grad()
def _compute_precisions_cholesky(cov: torch.Tensor, cov_type: str) -> torch.Tensor:
    """Compute precisions_cholesky from covariances.

    Shapes returned (sklearn-like):
    - diag:      (K, D) where entry is 1/sqrt(var)
    - spherical: (K,)   where entry is 1/sqrt(var)
    - tied:      (D, D) lower-triangular, such that precision = P^T P
    - full:      (K, D, D) lower-triangular per component

    For tied/full:
      cov = L L^T (L lower).  precision_chol = inv(L) (lower).
      precision = inv(cov) = inv(L^T) inv(L) = precision_chol^T precision_chol.
    """
    _check_cov_type(cov_type)

    if cov_type == "diag":
        return 1.0 / torch.sqrt(cov)

    if cov_type == "spherical":
        return 1.0 / torch.sqrt(cov)

    if cov_type == "tied":
        L = torch.linalg.cholesky(cov)
        I = torch.eye(L.shape[0], device=cov.device, dtype=cov.dtype)
        return torch.linalg.solve_triangular(L, I, upper=False)

    # full
    K, D, _ = cov.shape
    L = torch.linalg.cholesky(cov)  # (K, D, D) batched
    I = torch.eye(D, device=cov.device, dtype=cov.dtype).unsqueeze(0).expand(K, D, D)
    out = torch.linalg.solve_triangular(L, I, upper=False)  # batched solve
    return out


@torch.no_grad()
def _compute_precisions(prec_chol: torch.Tensor, cov_type: str) -> torch.Tensor:
    """Compute precisions (inverse covariances) from precisions_cholesky."""
    _check_cov_type(cov_type)

    if cov_type == "diag":
        return prec_chol * prec_chol

    if cov_type == "spherical":
        return prec_chol * prec_chol

    if cov_type == "tied":
        # precision = P^T P
        return prec_chol.transpose(-1, -2) @ prec_chol

    # full
    # precision = P^T P using batched matrix multiplication
    return torch.bmm(prec_chol.transpose(-1, -2), prec_chol)


# ---------------------------
# Log Gaussian probability via precisions_cholesky_ (sklearn-style E-step)
# ---------------------------

def _estimate_log_gaussian_prob_diag_precchol(
    X: torch.Tensor,
    means: torch.Tensor,
    precisions_chol: torch.Tensor,
) -> torch.Tensor:
    """Diag log N(X | means, cov) using precisions_cholesky (1/sqrt(var))."""
    N, D = X.shape
    K, D2 = means.shape
    assert D == D2
    assert precisions_chol.shape == (K, D)

    diff = X.unsqueeze(1) - means.unsqueeze(0)  # (N,K,D)
    y = diff * precisions_chol.unsqueeze(0)     # (N,K,D)
    mahal = torch.sum(y * y, dim=2)             # (N,K)

    # 0.5 * logdet(precision) = sum_d log(prec_chol_{k,d})
    log_det_term = torch.sum(torch.log(precisions_chol), dim=1)  # (K,)

    return -0.5 * (D * math.log(2 * math.pi) + mahal) + log_det_term.unsqueeze(0)


def _estimate_log_gaussian_prob_spherical_precchol(
    X: torch.Tensor,
    means: torch.Tensor,
    precisions_chol: torch.Tensor,
) -> torch.Tensor:
    """Spherical log N using precisions_cholesky (1/sqrt(var))."""
    N, D = X.shape
    K, D2 = means.shape
    assert D == D2
    assert precisions_chol.shape == (K,)

    diff = X.unsqueeze(1) - means.unsqueeze(0)  # (N,K,D)
    y = diff * precisions_chol.view(1, K, 1)    # (N,K,D)
    mahal = torch.sum(y * y, dim=2)             # (N,K)

    # sum log diag(P) where P = prec_chol * I -> D * log(prec_chol)
    log_det_term = D * torch.log(precisions_chol)  # (K,)

    return -0.5 * (D * math.log(2 * math.pi) + mahal) + log_det_term.unsqueeze(0)


def _estimate_log_gaussian_prob_tied_precchol(
    X: torch.Tensor,
    means: torch.Tensor,
    precisions_chol: torch.Tensor,
) -> torch.Tensor:
    """Tied-cov log N using precision_cholesky (D,D lower)."""
    N, D = X.shape
    K, D2 = means.shape
    assert D == D2
    assert precisions_chol.shape == (D, D)

    # log_det_term = sum log diag(prec_chol)
    log_det_term = torch.sum(torch.log(torch.diagonal(precisions_chol)))

    diff = X.unsqueeze(1) - means.unsqueeze(0)  # (N,K,D)

    # y = diff @ P^T using einsum for all components at once
    # diff: (N,K,D), precisions_chol: (D,D)
    # y[n,k,:] = diff[n,k,:] @ P^T
    y = torch.einsum('nkd,ed->nke', diff, precisions_chol)  # (N,K,D)
    mahal = torch.sum(y * y, dim=2)  # (N,K)

    return -0.5 * (D * math.log(2 * math.pi) + mahal) + log_det_term


def _estimate_log_gaussian_prob_full_precchol(
    X: torch.Tensor,
    means: torch.Tensor,
    precisions_chol: torch.Tensor,
) -> torch.Tensor:
    """Full-cov log N using precision_cholesky (K,D,D lower)."""
    N, D = X.shape
    K, D2 = means.shape
    assert D == D2
    assert precisions_chol.shape == (K, D, D)

    diff = X.unsqueeze(1) - means.unsqueeze(0)  # (N,K,D)

    # Compute log determinant terms for all components at once
    # precisions_chol: (K,D,D), extract diagonals: (K,D)
    log_det_term = torch.sum(torch.log(torch.diagonal(precisions_chol, dim1=1, dim2=2)), dim=1)  # (K,)

    # Compute Mahalanobis distance using einsum
    # y[n,k,:] = diff[n,k,:] @ P[k]^T
    y = torch.einsum('nkd,kde->nke', diff, precisions_chol.transpose(-1, -2))  # (N,K,D)
    mahal = torch.sum(y * y, dim=2)  # (N,K)

    return -0.5 * (D * math.log(2 * math.pi) + mahal) + log_det_term.unsqueeze(0)


def _estimate_log_gaussian_prob_precchol(
    X: torch.Tensor,
    means: torch.Tensor,
    precisions_chol: torch.Tensor,
    cov_type: str,
) -> torch.Tensor:
    _check_cov_type(cov_type)
    if cov_type == "diag":
        return _estimate_log_gaussian_prob_diag_precchol(X, means, precisions_chol)
    if cov_type == "spherical":
        return _estimate_log_gaussian_prob_spherical_precchol(X, means, precisions_chol)
    if cov_type == "tied":
        return _estimate_log_gaussian_prob_tied_precchol(X, means, precisions_chol)
    return _estimate_log_gaussian_prob_full_precchol(X, means, precisions_chol)


# ---------------------------
# EM steps
# ---------------------------

def _expectation_step_precchol(
    X: torch.Tensor,
    means: torch.Tensor,
    precisions_chol: torch.Tensor,
    weights: torch.Tensor,
    cov_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """E-step using precisions_cholesky (sklearn-style)."""
    log_prob = _estimate_log_gaussian_prob_precchol(X, means, precisions_chol, cov_type)  # (N,K)
    weighted_log_prob = log_prob + _safe_log(weights).unsqueeze(0)  # (N,K)

    log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)  # (N,)
    log_resp = weighted_log_prob - log_prob_norm.unsqueeze(1)  # (N,K)

    return log_prob_norm.mean(), log_resp


def _maximization_step(
    X: torch.Tensor,
    means: torch.Tensor,
    cov: torch.Tensor,
    weights: torch.Tensor,
    log_resp: torch.Tensor,
    cov_type: str,
    reg_covar: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """M-step (sklearn-style) producing updated means/cov/weights."""
    _check_cov_type(cov_type)
    N, D = X.shape
    K, D2 = means.shape
    assert D == D2
    assert log_resp.shape == (N, K)

    resp = log_resp.exp()  # (N,K)

    nk = resp.sum(dim=0) + _nk_eps(resp.dtype)  # (K,)

    new_weights = nk / nk.sum()
    new_means = (resp.T @ X) / nk.unsqueeze(1)  # (K,D)

    diff = X.unsqueeze(1) - new_means.unsqueeze(0)  # (N,K,D)

    if cov_type == "diag":
        # Vectorized computation: (resp * diff^2).sum over N, then divide by nk
        # resp: (N,K), diff: (N,K,D)
        weighted_sq_diff = resp.unsqueeze(2) * (diff ** 2)  # (N,K,D)
        new_cov = weighted_sq_diff.sum(dim=0) / nk.unsqueeze(1)  # (K,D)
        new_cov = new_cov + reg_covar

    elif cov_type == "spherical":
        avg_X2 = (resp.T @ (X * X)) / nk.unsqueeze(1)     # (K,D)
        avg_means2 = new_means * new_means               # (K,D)
        diag_cov = (avg_X2 - avg_means2) + reg_covar      # (K,D)
        new_cov = diag_cov.mean(dim=1)                    # (K,)

    elif cov_type == "tied":
        # Compute tied covariance in one einsum operation
        # sum_k sum_n resp[n,k] * diff[n,k,:] * diff[n,k,:]^T
        cov_sum = torch.einsum('nk,nkd,nke->de', resp, diff, diff)  # (D,D)
        new_cov = cov_sum / nk.sum()
        new_cov = new_cov + reg_covar * torch.eye(D, device=X.device, dtype=X.dtype)

    else:  # full
        # Compute full covariance for all components in one einsum operation
        # For each k: sum_n resp[n,k] * diff[n,k,:] * diff[n,k,:]^T / nk[k]
        cov_sum = torch.einsum('nk,nkd,nke->kde', resp, diff, diff)  # (K,D,D)
        new_cov = cov_sum / nk.unsqueeze(1).unsqueeze(2)  # (K,D,D)
        eye = torch.eye(D, device=X.device, dtype=X.dtype)
        new_cov = new_cov + reg_covar * eye.unsqueeze(0)

    return new_means, new_cov, new_weights


# ---------------------------
# Initialization helpers
# ---------------------------

@torch.no_grad()
def _kmeans_plus_plus_init_centroids(X: torch.Tensor, K: int) -> torch.Tensor:
    """k-means++ seeding. Returns centroids (K, D)."""
    N, D = X.shape
    device, dtype = X.device, X.dtype

    centroids = torch.empty((K, D), device=device, dtype=dtype)

    # First centroid uniformly
    i0 = torch.randint(0, N, (1,), device=device).item()
    centroids[0] = X[i0]

    # Closest squared dist to any chosen centroid so far
    closest_d2 = torch.sum((X - centroids[0]) ** 2, dim=1)  # (N,)

    for k in range(1, K):
        probs = closest_d2 / closest_d2.sum()
        idx = torch.multinomial(probs, 1).item()
        centroids[k] = X[idx]

        d2_new = torch.sum((X - centroids[k]) ** 2, dim=1)
        closest_d2 = torch.minimum(closest_d2, d2_new)

    return centroids


@torch.no_grad()
def _initialize_from_sklearn(X: torch.Tensor, K: int, method: str) -> torch.Tensor:
    """Use sklearn's initialization to create resp matrix. Returns log_resp (N,K)."""
    N, D = X.shape
    device, dtype = X.device, X.dtype
    
    # Convert to numpy for sklearn
    X_np = X.cpu().numpy()
    
    resp = np.zeros((N, K), dtype=np.float64)
    
    if method == "scikit_kmeans":
        # Use sklearn's KMeans initialization
        label = KMeans(
            n_clusters=K, 
            n_init=1, 
            random_state=None
        ).fit(X_np).labels_
        resp[np.arange(N), label] = 1
    elif method == "scikit_random":
        # Use sklearn's random initialization
        resp = np.random.uniform(size=(N, K))
        resp /= resp.sum(axis=1)[:, np.newaxis]
    else:
        raise ValueError(f"Unknown sklearn init method: {method}")
    
    # Convert back to torch
    resp_torch = torch.from_numpy(resp).to(device=device, dtype=dtype)
    return _safe_log(resp_torch)


@torch.no_grad()
def _kmeans_lloyd_with_init(
    X: torch.Tensor,
    centroids: torch.Tensor,
    n_iter: int = 10,
) -> torch.Tensor:
    """Run Lloyd iterations starting from provided centroids. Returns labels (N,)."""
    N, D = X.shape
    K, D2 = centroids.shape
    assert D == D2

    labels = torch.zeros((N,), device=X.device, dtype=torch.long)
    for _ in range(n_iter):
        d2 = torch.cdist(X, centroids).pow(2)  # (N,K)
        labels = torch.argmin(d2, dim=1)

        # Parallelize centroid updates using scatter_add
        counts = torch.zeros((K,), device=X.device, dtype=X.dtype)
        sums = torch.zeros((K, D), device=X.device, dtype=X.dtype)
        
        counts.scatter_add_(0, labels, torch.ones((N,), device=X.device, dtype=X.dtype))
        sums.scatter_add_(0, labels.unsqueeze(1).expand(N, D), X)
        
        # Handle empty clusters
        empty_mask = counts == 0
        if empty_mask.any():
            random_idx = torch.randint(0, N, (empty_mask.sum().item(),), device=X.device)
            centroids[empty_mask] = X[random_idx]
            counts[empty_mask] = 1.0
        
        centroids = sums / counts.unsqueeze(1)

    return labels


# ---------------------------
# Model wrapper
# ---------------------------

@dataclass
class GMMParams:
    weights: torch.Tensor
    means: torch.Tensor
    cov: torch.Tensor
    prec_chol: torch.Tensor
    cov_type: str


class TorchGaussianMixture:
    """Sklearn-shaped GaussianMixture-like class in PyTorch."""

    def __init__(
        self,
        n_components: int,
        covariance_type: str = "diag",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        warm_start: bool = False,
        device=None,
        dtype=None,
        kmeans_iter: int = 10,
        # User-supplied init (sklearn-style):
        weights_init: Optional[torch.Tensor] = None,
        means_init: Optional[torch.Tensor] = None,
        precisions_init: Optional[torch.Tensor] = None,
    ) -> None:
        _check_cov_type(covariance_type)
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if reg_covar < 0:
            raise ValueError("reg_covar must be non-negative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if n_init <= 0:
            raise ValueError("n_init must be positive")
        if init_params not in ("kmeans", "k-means++", "random", "random_from_data", "scikit_random", "scikit_kmeans"):
            raise ValueError(
                "init_params must be one of: 'kmeans', 'k-means++', 'random', 'random_from_data', 'scikit_random', 'scikit_kmeans'"
            )

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.warm_start = warm_start
        self.device = device
        self.dtype = dtype
        self.kmeans_iter = kmeans_iter

        # User init
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

        # sklearn-like fitted attributes
        self.weights_: Optional[torch.Tensor] = None
        self.means_: Optional[torch.Tensor] = None
        self.covariances_: Optional[torch.Tensor] = None
        self.precisions_cholesky_: Optional[torch.Tensor] = None
        self.precisions_: Optional[torch.Tensor] = None

        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = float("-inf")
        self.lower_bounds_: List[float] = []

        self._params: Optional[GMMParams] = None

    def _to_device_dtype(self, X: torch.Tensor) -> torch.Tensor:
        if self.device is not None:
            X = X.to(self.device)
        if self.dtype is not None:
            X = X.to(self.dtype)
        return X

    # don't really get this

    def _n_parameters(self, D: int) -> int:
        """Parameter count like sklearn for AIC/BIC."""
        K = self.n_components
        # weights: K-1
        p = K - 1
        # means: K*D
        p += K * D
        ct = self.covariance_type
        if ct == "spherical":
            p += K
        elif ct == "diag":
            p += K * D
        elif ct == "tied":
            p += D * (D + 1) // 2
        else:
            p += K * D * (D + 1) // 2
        return int(p)

    def _convert_precisions_to_cov(self, precisions: torch.Tensor, D: int) -> torch.Tensor:
        """Convert sklearn-style precisions_init (inverse covariance) to cov storage."""
        ct = self.covariance_type
        K = self.n_components

        if ct == "diag":
            if precisions.shape != (K, D):
                raise ValueError(f"precisions_init must have shape (K,D) for diag, got {tuple(precisions.shape)}")
            cov = 1.0 / precisions
            return cov

        if ct == "spherical":
            if precisions.shape != (K,):
                raise ValueError(f"precisions_init must have shape (K,) for spherical, got {tuple(precisions.shape)}")
            cov = 1.0 / precisions
            return cov

        if ct == "tied":
            if precisions.shape != (D, D):
                raise ValueError(f"precisions_init must have shape (D,D) for tied, got {tuple(precisions.shape)}")
            cov = torch.linalg.inv(precisions)
            cov = _add_reg_diag(cov, self.reg_covar)
            return cov

        # full
        if precisions.shape != (K, D, D):
            raise ValueError(f"precisions_init must have shape (K,D,D) for full, got {tuple(precisions.shape)}")
        cov = torch.linalg.inv(precisions)
        cov = _add_reg_diag(cov, self.reg_covar)
        return cov

    @torch.no_grad()
    def _make_params(self, weights: torch.Tensor, means: torch.Tensor, cov: torch.Tensor) -> GMMParams:
        prec_chol = _compute_precisions_cholesky(cov, self.covariance_type)
        return GMMParams(weights=weights, means=means, cov=cov, prec_chol=prec_chol, cov_type=self.covariance_type)

    @torch.no_grad()
    def _initialize_from_log_resp(self, X: torch.Tensor, log_resp: torch.Tensor) -> GMMParams:
        K = self.n_components
        N, D = X.shape

        # start with dummy parameters
        means0 = torch.zeros((K, D), device=X.device, dtype=X.dtype)
        if self.covariance_type == "diag":
            cov0 = torch.ones((K, D), device=X.device, dtype=X.dtype)
        elif self.covariance_type == "spherical":
            cov0 = torch.ones((K,), device=X.device, dtype=X.dtype)
        elif self.covariance_type == "tied":
            cov0 = torch.eye(D, device=X.device, dtype=X.dtype)
        else:
            cov0 = torch.stack([torch.eye(D, device=X.device, dtype=X.dtype) for _ in range(K)], dim=0)
        weights0 = torch.full((K,), 1.0 / K, device=X.device, dtype=X.dtype)

        # Run one M-step to get real initial parameters
        means, cov, weights = _maximization_step(
            X,
            means0,
            cov0,
            weights0,
            log_resp,
            self.covariance_type,
            reg_covar=self.reg_covar,
        )
        return self._make_params(weights, means, cov)

    @torch.no_grad()
    def _initialize(self, X: torch.Tensor) -> GMMParams:
        X = self._to_device_dtype(X)
        N, D = X.shape
        K = self.n_components

        # 1) User-supplied init (sklearn-style)
        if self.means_init is not None:
            means = self._to_device_dtype(self.means_init)
            if means.shape != (K, D):
                raise ValueError(f"means_init must have shape (K,D) = {(K,D)}, got {tuple(means.shape)}")

            if self.weights_init is None:
                weights = torch.full((K,), 1.0 / K, device=X.device, dtype=X.dtype)
            else:
                weights = self._to_device_dtype(self.weights_init)
                if weights.shape != (K,):
                    raise ValueError(f"weights_init must have shape (K,), got {tuple(weights.shape)}")
                weights = weights / weights.sum()

            if self.precisions_init is not None:
                precisions = self._to_device_dtype(self.precisions_init)
                cov = self._convert_precisions_to_cov(precisions, D)
            else:
                # If user gives means but not precisions, start from global covariance/variance.
                Xc = X - X.mean(dim=0, keepdim=True)
                if self.covariance_type in ("tied", "full"):
                    cov_global = (Xc.T @ Xc) / max(N - 1, 1)
                    cov_global = _add_reg_diag(cov_global, self.reg_covar)
                    cov = cov_global if self.covariance_type == "tied" else cov_global.unsqueeze(0).expand(K, D, D).contiguous()
                elif self.covariance_type == "diag":
                    var = Xc.var(dim=0, unbiased=False) + self.reg_covar
                    cov = var.unsqueeze(0).expand(K, D).contiguous()
                else:
                    var = Xc.var(dim=0, unbiased=False).mean() + self.reg_covar
                    cov = torch.full((K,), var, device=X.device, dtype=X.dtype)

            return self._make_params(weights, means, cov)

        # 2) Built-in init modes
        if self.init_params == "random":
            resp = torch.rand((N, K), device=X.device, dtype=X.dtype)
            resp = resp / resp.sum(dim=1, keepdim=True)
            log_resp = _safe_log(resp)
            return self._initialize_from_log_resp(X, log_resp)

        if self.init_params in ("kmeans", "k-means++"):
            centroids = _kmeans_plus_plus_init_centroids(X, K)

            if self.init_params == "kmeans":
                labels = _kmeans_lloyd_with_init(X, centroids, n_iter=self.kmeans_iter)
            else:
                # k-means++ init: assign once from seeded centroids (no Lloyd refinement)
                d2 = torch.cdist(X, centroids).pow(2)
                labels = torch.argmin(d2, dim=1)

            resp = torch.zeros((N, K), device=X.device, dtype=X.dtype)
            resp[torch.arange(N, device=X.device), labels] = 1.0
            log_resp = _safe_log(resp)
            return self._initialize_from_log_resp(X, log_resp)

        if self.init_params in ("scikit_kmeans", "scikit_random"):
            log_resp = _initialize_from_sklearn(X, K, self.init_params)
            return self._initialize_from_log_resp(X, log_resp)

        # random_from_data
        idx = torch.randperm(N, device=X.device)[:K]
        means = X[idx].clone()
        weights = torch.full((K,), 1.0 / K, device=X.device, dtype=X.dtype)

        Xc = X - X.mean(dim=0, keepdim=True)
        if self.covariance_type in ("tied", "full"):
            cov_global = (Xc.T @ Xc) / max(N - 1, 1)
            cov_global = _add_reg_diag(cov_global, self.reg_covar)
            cov = cov_global if self.covariance_type == "tied" else cov_global.unsqueeze(0).expand(K, D, D).contiguous()
        elif self.covariance_type == "diag":
            var = Xc.var(dim=0, unbiased=False) + self.reg_covar
            cov = var.unsqueeze(0).expand(K, D).contiguous()
        else:
            var = Xc.var(dim=0, unbiased=False).mean() + self.reg_covar
            cov = torch.full((K,), var, device=X.device, dtype=X.dtype)

        return self._make_params(weights, means, cov)

    # -----------------------
    # Public API
    # -----------------------

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "TorchGaussianMixture":
        X = self._to_device_dtype(X)
        N, D = X.shape

        best_lower = torch.tensor(float("-inf"), device=X.device, dtype=X.dtype)
        best_params: Optional[GMMParams] = None
        best_n_iter = 0
        best_converged = False
        best_history: List[float] = []

        n_init = 1 if (self.warm_start and self._params is not None) else self.n_init

        for _ in range(n_init):
            p = self._params if (self.warm_start and self._params is not None) else self._initialize(X)

            prev_lower = torch.tensor(float("-inf"), device=X.device, dtype=X.dtype)
            converged = False
            history: List[float] = []

            for it in range(self.max_iter):
                lower, log_resp = _expectation_step_precchol(X, p.means, p.prec_chol, p.weights, p.cov_type)
                history.append(float(lower.item()))

                means, cov, weights = _maximization_step(
                    X, p.means, p.cov, p.weights, log_resp, p.cov_type, reg_covar=self.reg_covar
                )
                p = self._make_params(weights, means, cov)

                change = lower - prev_lower
                if torch.abs(change) < self.tol:
                    converged = True
                    break
                prev_lower = lower

            if prev_lower > best_lower:
                best_lower = prev_lower
                best_params = p
                best_n_iter = it + 1
                best_converged = converged
                best_history = history

        assert best_params is not None

        # Save internal + sklearn-like attributes
        self._params = best_params

        self.weights_ = best_params.weights
        self.means_ = best_params.means
        self.covariances_ = best_params.cov
        self.precisions_cholesky_ = best_params.prec_chol
        self.precisions_ = _compute_precisions(best_params.prec_chol, best_params.cov_type)

        self.lower_bound_ = float(best_lower.item())
        self.lower_bounds_ = best_history
        self.n_iter_ = best_n_iter
        self.converged_ = best_converged

        return self

    @torch.no_grad()
    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        """Per-sample log-likelihood (N,)."""
        if self._params is None:
            raise RuntimeError("Model is not fitted yet.")
        X = self._to_device_dtype(X)

        log_prob = _estimate_log_gaussian_prob_precchol(
            X, self._params.means, self._params.prec_chol, self._params.cov_type
        )  # (N,K)
        weighted = log_prob + _safe_log(self._params.weights).unsqueeze(0)
        return torch.logsumexp(weighted, dim=1)

    @torch.no_grad()
    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Mean log-likelihood."""
        return self.score_samples(X).mean()

    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Posterior responsibilities (N,K)."""
        if self._params is None:
            raise RuntimeError("Model is not fitted yet.")
        X = self._to_device_dtype(X)
        _, log_resp = _expectation_step_precchol(
            X, self._params.means, self._params.prec_chol, self._params.weights, self._params.cov_type
        )
        return log_resp.exp()

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.predict_proba(X), dim=1)

    @torch.no_grad()
    def aic(self, X: torch.Tensor) -> torch.Tensor:
        """Akaike information criterion."""
        X = self._to_device_dtype(X)
        N, D = X.shape
        ll = self.score_samples(X).sum()
        p = self._n_parameters(D)
        return 2.0 * p - 2.0 * ll

    @torch.no_grad()
    def bic(self, X: torch.Tensor) -> torch.Tensor:
        """Bayesian information criterion."""
        X = self._to_device_dtype(X)
        N, D = X.shape
        ll = self.score_samples(X).sum()
        p = self._n_parameters(D)
        return math.log(N) * p - 2.0 * ll

    @torch.no_grad()
    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the mixture.

        Returns:
          X: (n_samples, D)
          labels: (n_samples,)
        """
        if self._params is None:
            raise RuntimeError("Model is not fitted yet.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        p = self._params
        K, D = p.means.shape

        cat = torch.distributions.Categorical(probs=p.weights)
        labels = cat.sample((n_samples,))  # (n_samples,)

        # Generate samples more efficiently by indexing
        # Select means and covariances based on labels, then sample all at once
        selected_means = p.means[labels]  # (n_samples, D)
        
        if p.cov_type == "diag":
            std = torch.sqrt(p.cov[labels])  # (n_samples, D)
            noise = torch.randn((n_samples, D), device=p.means.device, dtype=p.means.dtype)
            X_out = selected_means + noise * std

        elif p.cov_type == "spherical":
            std = torch.sqrt(p.cov[labels]).unsqueeze(1)  # (n_samples, 1)
            noise = torch.randn((n_samples, D), device=p.means.device, dtype=p.means.dtype)
            X_out = selected_means + noise * std

        elif p.cov_type == "tied":
            # Sample from N(0, cov) then add component-specific means
            mvn = torch.distributions.MultivariateNormal(
                loc=torch.zeros(D, device=p.means.device, dtype=p.means.dtype),
                covariance_matrix=p.cov,
            )
            X_out = mvn.sample((n_samples,)) + selected_means

        else:  # full
            # For full covariance, we need to handle each component separately
            # but we can batch sample generation
            X_out = torch.empty((n_samples, D), device=p.means.device, dtype=p.means.dtype)
            for k in range(K):
                mask = labels == k
                if mask.any():
                    n_k = int(mask.sum().item())
                    mvn = torch.distributions.MultivariateNormal(loc=p.means[k], covariance_matrix=p.cov[k])
                    X_out[mask] = mvn.sample((n_k,))

        return X_out, labels
