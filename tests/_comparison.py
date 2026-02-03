import os
import sys
import math
import numpy as np
import torch
import pytest

from scipy import linalg, stats

# Make local modules importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---- your torch kernels ----
from implementation._torch_gmm_em import (
    _estimate_log_gaussian_prob_diag_precchol as _estimate_log_gaussian_prob_diag,
    _estimate_log_gaussian_prob_spherical_precchol as _estimate_log_gaussian_prob_spherical,
    _estimate_log_gaussian_prob_tied_precchol as _estimate_log_gaussian_prob_tied,
    _estimate_log_gaussian_prob_full_precchol as _estimate_log_gaussian_prob_full,
    _expectation_step_precchol as _expectation_step,
    _maximization_step,
    _compute_precisions_cholesky as _torch_compute_precisions_cholesky,
)

# ---- sklearn reference kernels (the same ones used by the official tests) ----
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _compute_log_det_cholesky,
    _estimate_log_gaussian_prob as _sk_estimate_log_gaussian_prob,
    _estimate_gaussian_parameters as _sk_estimate_gaussian_parameters,
)

torch.set_default_dtype(torch.float64)

COVARIANCE_TYPE = ["full", "tied", "diag", "spherical"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _torch_log_prob(X_t: torch.Tensor, means_t: torch.Tensor, cov_t: torch.Tensor, cov_type: str) -> torch.Tensor:
    """Compute log probability using covariances (converts to precision_chol internally)."""
    prec_chol_t = _torch_compute_precisions_cholesky(cov_t, cov_type)
    if cov_type == "diag":
        return _estimate_log_gaussian_prob_diag(X_t, means_t, prec_chol_t)
    if cov_type == "spherical":
        return _estimate_log_gaussian_prob_spherical(X_t, means_t, prec_chol_t)
    if cov_type == "tied":
        return _estimate_log_gaussian_prob_tied(X_t, means_t, prec_chol_t)
    if cov_type == "full":
        return _estimate_log_gaussian_prob_full(X_t, means_t, prec_chol_t)
    raise ValueError(cov_type)


def _random_spd(rng: np.random.RandomState, d: int) -> np.ndarray:
    A = rng.randn(d, d)
    C = A @ A.T
    C /= np.trace(C) / d
    return C


def _random_cov(rng: np.random.RandomState, K: int, D: int, cov_type: str, reg: float = 1e-6) -> np.ndarray:
    if cov_type == "diag":
        return (0.5 + rng.rand(K, D)) ** 2
    if cov_type == "spherical":
        return (0.5 + rng.rand(K)) ** 2
    if cov_type == "tied":
        return _random_spd(rng, D) + reg * np.eye(D)
    if cov_type == "full":
        covs = []
        for _ in range(K):
            covs.append(_random_spd(rng, D) + reg * np.eye(D))
        return np.stack(covs, axis=0)
    raise ValueError(cov_type)


def _sk_resp_from_logprob(log_prob: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # matches the logic used in many sklearn GM tests: softmax(log_prob + log(weights))
    log_resp_unnorm = log_prob + np.log(weights)[None, :]
    log_norm = np.logaddexp.reduce(log_resp_unnorm, axis=1, keepdims=True)
    return np.exp(log_resp_unnorm - log_norm)


def _torch_log_likelihood_mean(X_t, means_t, cov_t, weights_t, cov_type) -> float:
    """Compute mean log likelihood."""
    log_prob = _torch_log_prob(X_t, means_t, cov_t, cov_type)
    log_w = torch.log(weights_t).unsqueeze(0)
    return torch.logsumexp(log_prob + log_w, dim=1).mean().item()


def _torch_expectation_step(X_t, means_t, cov_t, weights_t, cov_type) -> torch.Tensor:
    """E-step wrapper that returns responsibilities (not log responsibilities)."""
    prec_chol_t = _torch_compute_precisions_cholesky(cov_t, cov_type)
    _, log_resp = _expectation_step(X_t, means_t, prec_chol_t, weights_t, cov_type)
    return log_resp.exp()


def _make_dummy_params(K: int, D: int, cov_type: str):
    dummy_means = torch.zeros(K, D, dtype=torch.float64)
    dummy_cov = {
        "diag": torch.ones(K, D, dtype=torch.float64),
        "spherical": torch.ones(K, dtype=torch.float64),
        "tied": torch.eye(D, dtype=torch.float64),
        "full": torch.stack([torch.eye(D, dtype=torch.float64) for _ in range(K)]),
    }[cov_type]
    dummy_weights = torch.full((K,), 1.0 / K, dtype=torch.float64)
    return dummy_means, dummy_cov, dummy_weights


# ---------------------------------------------------------------------
# 1) Port of sklearn's "log_probabilities" sanity test (diag naive)
# ---------------------------------------------------------------------
def _naive_lmvnpdf_diag(X: np.ndarray, means: np.ndarray, covars_diag: np.ndarray) -> np.ndarray:
    resp = np.empty((len(X), len(means)))
    stds = np.sqrt(covars_diag)
    for i, (mean, std) in enumerate(zip(means, stds)):
        resp[:, i] = stats.norm.logpdf(X, mean, std).sum(axis=1)
    return resp


def test_torch_log_probabilities_match_naive_diag_reference():
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 500, 2, 2

    means = rng.rand(n_components, n_features)
    covars_diag = rng.rand(n_components, n_features) + 0.1  # avoid tiny variances
    X = rng.rand(n_samples, n_features)

    log_prob_naive = _naive_lmvnpdf_diag(X, means, covars_diag)

    # diag
    X_t = torch.as_tensor(X, dtype=torch.float64)
    means_t = torch.as_tensor(means, dtype=torch.float64)
    cov_diag_t = torch.as_tensor(covars_diag, dtype=torch.float64)
    log_prob_t = _torch_log_prob(X_t, means_t, cov_diag_t, "diag").cpu().numpy()
    np.testing.assert_allclose(log_prob_t, log_prob_naive, rtol=1e-6, atol=1e-6)

    # full: build full cov matrices from diag
    cov_full = np.array([np.diag(v) for v in covars_diag])
    cov_full_t = torch.as_tensor(cov_full, dtype=torch.float64)
    log_prob_t_full = _torch_log_prob(X_t, means_t, cov_full_t, "full").cpu().numpy()
    np.testing.assert_allclose(log_prob_t_full, log_prob_naive, rtol=1e-6, atol=1e-6)

    # tied: average diag -> one shared covariance
    cov_tied = np.diag(covars_diag.mean(axis=0))
    cov_tied_t = torch.as_tensor(cov_tied, dtype=torch.float64)
    log_prob_t_tied = _torch_log_prob(X_t, means_t, cov_tied_t, "tied").cpu().numpy()
    log_prob_naive_tied = _naive_lmvnpdf_diag(X, means, np.tile(np.diag(cov_tied), (n_components, 1)))
    np.testing.assert_allclose(log_prob_t_tied, log_prob_naive_tied, rtol=1e-6, atol=1e-6)

    # spherical: average variances per component -> one scalar per component
    cov_sph = covars_diag.mean(axis=1)
    cov_sph_t = torch.as_tensor(cov_sph, dtype=torch.float64)
    log_prob_t_sph = _torch_log_prob(X_t, means_t, cov_sph_t, "spherical").cpu().numpy()
    cov_sph_diag = np.array([[c] * n_features for c in cov_sph])
    log_prob_naive_sph = _naive_lmvnpdf_diag(X, means, cov_sph_diag)
    np.testing.assert_allclose(log_prob_t_sph, log_prob_naive_sph, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 1b) Direct oracle: torch log_prob vs sklearn log_prob (all cov types)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_torch_log_prob_matches_sklearn_oracle(cov_type):
    rng = np.random.RandomState(11)
    N, K, D = 150, 3, 4

    X = rng.randn(N, D)
    means = rng.randn(K, D)
    cov = _random_cov(rng, K, D, cov_type, reg=1e-6)

    X_t = torch.as_tensor(X, dtype=torch.float64)
    means_t = torch.as_tensor(means, dtype=torch.float64)
    cov_t = torch.as_tensor(cov, dtype=torch.float64)

    log_prob_t = _torch_log_prob(X_t, means_t, cov_t, cov_type).cpu().numpy()
    prec_chol_sk = _compute_precision_cholesky(cov, cov_type)
    log_prob_sk = _sk_estimate_log_gaussian_prob(X, means, prec_chol_sk, cov_type)

    np.testing.assert_allclose(log_prob_t, log_prob_sk, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 1c) Per-sample log-likelihood matches sklearn oracle (final objective)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_torch_log_likelihood_matches_sklearn_oracle(cov_type):
    rng = np.random.RandomState(13)
    N, K, D = 200, 4, 3

    X = rng.randn(N, D)
    means = rng.randn(K, D)
    cov = _random_cov(rng, K, D, cov_type, reg=1e-6)
    weights = rng.rand(K)
    weights /= weights.sum()

    # sklearn oracle: ll_n = logsumexp(log_prob + log(weights))
    prec_chol_sk = _compute_precision_cholesky(cov, cov_type)
    log_prob_sk = _sk_estimate_log_gaussian_prob(X, means, prec_chol_sk, cov_type)
    ll_sk = np.logaddexp.reduce(log_prob_sk + np.log(weights)[None, :], axis=1)

    # torch
    X_t = torch.as_tensor(X, dtype=torch.float64)
    means_t = torch.as_tensor(means, dtype=torch.float64)
    cov_t = torch.as_tensor(cov, dtype=torch.float64)
    weights_t = torch.as_tensor(weights, dtype=torch.float64)

    log_prob_t = _torch_log_prob(X_t, means_t, cov_t, cov_type)
    ll_t = torch.logsumexp(log_prob_t + torch.log(weights_t)[None, :], dim=1).cpu().numpy()

    np.testing.assert_allclose(ll_t, ll_sk, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 2) Responsibilities normalized and match sklearn softmax oracle
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_torch_responsibilities_are_normalized_and_match_sklearn_softmax(cov_type):
    rng = np.random.RandomState(0)
    N, K, D = 200, 3, 4

    X = rng.randn(N, D)
    means = rng.randn(K, D) * 2.0
    cov = _random_cov(rng, K, D, cov_type, reg=1e-6)
    weights = rng.rand(K)
    weights /= weights.sum()

    X_t = torch.as_tensor(X, dtype=torch.float64)
    means_t = torch.as_tensor(means, dtype=torch.float64)
    cov_t = torch.as_tensor(cov, dtype=torch.float64)
    weights_t = torch.as_tensor(weights, dtype=torch.float64)

    resp_t = _torch_expectation_step(X_t, means_t, cov_t, weights_t, cov_type).cpu().numpy()

    np.testing.assert_allclose(resp_t.sum(axis=1), np.ones(N), atol=1e-10)

    prec_chol = _compute_precision_cholesky(cov, cov_type)
    log_prob_sk = _sk_estimate_log_gaussian_prob(X, means, prec_chol, cov_type)
    resp_sk = _sk_resp_from_logprob(log_prob_sk, weights)

    np.testing.assert_allclose(resp_t, resp_sk, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------
# 3) M-step matches sklearn estimate_gaussian_parameters (oracle)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_torch_m_step_matches_sklearn_estimate_gaussian_parameters(cov_type):
    rng = np.random.RandomState(2)
    N, K, D = 300, 4, 5
    reg_covar = 1e-6

    X = rng.randn(N, D)
    resp = rng.rand(N, K)
    resp /= resp.sum(axis=1, keepdims=True)

    X_t = torch.as_tensor(X, dtype=torch.float64)
    log_resp_t = torch.log(torch.as_tensor(resp, dtype=torch.float64))
    dummy_means, dummy_cov, dummy_weights = _make_dummy_params(K, D, cov_type)

    means_t, cov_t, weights_t = _maximization_step(
        X_t, dummy_means, dummy_cov, dummy_weights, log_resp_t, cov_type, reg_covar=reg_covar
    )

    nk_sk, means_sk, cov_sk = _sk_estimate_gaussian_parameters(X, resp, reg_covar, cov_type)
    weights_sk = nk_sk / nk_sk.sum()

    np.testing.assert_allclose(means_t.cpu().numpy(), means_sk, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cov_t.cpu().numpy(), cov_sk, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(weights_t.cpu().numpy(), weights_sk, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------
# 4) Precision / covariance consistency checks (math identities)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_torch_covariance_precision_cholesky_consistency(cov_type):
    rng = np.random.RandomState(3)
    N, K, D = 400, 3, 4
    reg_covar = 1e-6

    X = rng.randn(N, D)
    resp = rng.rand(N, K)
    resp /= resp.sum(axis=1, keepdims=True)

    X_t = torch.as_tensor(X, dtype=torch.float64)
    log_resp_t = torch.log(torch.as_tensor(resp, dtype=torch.float64))

    dummy_means, dummy_cov, dummy_weights = _make_dummy_params(K, D, cov_type)
    _, cov_t, _ = _maximization_step(
        X_t, dummy_means, dummy_cov, dummy_weights, log_resp_t, cov_type, reg_covar=reg_covar
    )
    cov_np = cov_t.cpu().numpy()

    prec_chol = _compute_precision_cholesky(cov_np, cov_type)

    if cov_type == "full":
        prec = np.array([P @ P.T for P in prec_chol])
        inv_cov = np.array([linalg.inv(C) for C in cov_np])
        np.testing.assert_allclose(prec, inv_cov, rtol=1e-6, atol=1e-6)
    elif cov_type == "tied":
        prec = prec_chol @ prec_chol.T
        inv_cov = linalg.inv(cov_np)
        np.testing.assert_allclose(prec, inv_cov, rtol=1e-6, atol=1e-6)
    elif cov_type == "diag":
        np.testing.assert_allclose(cov_np, 1.0 / (prec_chol ** 2), rtol=1e-6, atol=1e-6)
    else:  # spherical
        np.testing.assert_allclose(cov_np, 1.0 / (prec_chol ** 2), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 5) compute_log_det_cholesky style check (math identity)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_torch_covariances_produce_correct_log_det_via_sklearn_helpers(cov_type):
    rng = np.random.RandomState(4)
    N, K, D = 300, 3, 4
    reg_covar = 1e-6

    X = rng.randn(N, D)
    resp = rng.rand(N, K)
    resp /= resp.sum(axis=1, keepdims=True)

    X_t = torch.as_tensor(X, dtype=torch.float64)
    log_resp_t = torch.log(torch.as_tensor(resp, dtype=torch.float64))
    dummy_means, dummy_cov, dummy_weights = _make_dummy_params(K, D, cov_type)

    _, cov_t, _ = _maximization_step(
        X_t, dummy_means, dummy_cov, dummy_weights, log_resp_t, cov_type, reg_covar=reg_covar
    )
    cov_np = cov_t.cpu().numpy()

    prec_chol = _compute_precision_cholesky(cov_np, cov_type)
    log_det = _compute_log_det_cholesky(prec_chol, cov_type, n_features=D)

    if cov_type == "full":
        dets = np.array([linalg.det(C) for C in cov_np])
    elif cov_type == "tied":
        dets = linalg.det(cov_np)
    elif cov_type == "diag":
        dets = np.array([np.prod(v) for v in cov_np])
    else:  # spherical
        dets = cov_np ** D

    np.testing.assert_allclose(log_det, -0.5 * np.log(dets), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 5b) Precision-Cholesky computation equivalence (sklearn vs torch)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_torch_precision_cholesky_matches_sklearn_precision(cov_type):
    rng = np.random.RandomState(5)
    K, D = 4, 3
    cov = _random_cov(rng, K, D, cov_type, reg=1e-6)

    cov_t = torch.as_tensor(cov, dtype=torch.float64)
    prec_chol_t = _torch_compute_precisions_cholesky(cov_t, cov_type).cpu().numpy()
    prec_chol_sk = _compute_precision_cholesky(cov, cov_type)

    if cov_type == "full":
        prec_t = np.array([P.T @ P for P in prec_chol_t])
        prec_sk = np.array([P @ P.T for P in prec_chol_sk])
        inv_cov = np.array([linalg.inv(C) for C in cov])
        np.testing.assert_allclose(prec_t, inv_cov, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(prec_sk, inv_cov, rtol=1e-6, atol=1e-6)
    elif cov_type == "tied":
        prec_t = prec_chol_t.T @ prec_chol_t
        prec_sk = prec_chol_sk @ prec_chol_sk.T
        inv_cov = linalg.inv(cov)
        np.testing.assert_allclose(prec_t, inv_cov, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(prec_sk, inv_cov, rtol=1e-6, atol=1e-6)
    else:
        np.testing.assert_allclose(prec_chol_t, prec_chol_sk, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 5c) Log-det + Mahalanobis term correctness (full/tied)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", ["full", "tied"])
def test_log_prob_terms_full_tied(cov_type):
    rng = np.random.RandomState(6)
    N, K, D = 50, 3, 4
    X = rng.randn(N, D)
    means = rng.randn(K, D)
    cov = _random_cov(rng, K, D, cov_type, reg=1e-6)

    X_t = torch.as_tensor(X, dtype=torch.float64)
    means_t = torch.as_tensor(means, dtype=torch.float64)
    cov_t = torch.as_tensor(cov, dtype=torch.float64)
    log_prob = _torch_log_prob(X_t, means_t, cov_t, cov_type).cpu().numpy()

    log_2pi = math.log(2 * math.pi)

    if cov_type == "tied":
        inv_cov = linalg.inv(cov)
        _, logdet_cov = np.linalg.slogdet(cov)
        log_det_term = -0.5 * logdet_cov
        for k in range(K):
            diff = X - means[k]
            mahal = np.sum(diff @ inv_cov * diff, axis=1)
            expected = -0.5 * (D * log_2pi + mahal) + log_det_term
            np.testing.assert_allclose(log_prob[:, k], expected, rtol=1e-6, atol=1e-6)
    else:  # full
        for k in range(K):
            inv_cov = linalg.inv(cov[k])
            _, logdet_cov = np.linalg.slogdet(cov[k])
            log_det_term = -0.5 * logdet_cov
            diff = X - means[k]
            mahal = np.sum(diff @ inv_cov * diff, axis=1)
            expected = -0.5 * (D * log_2pi + mahal) + log_det_term
            np.testing.assert_allclose(log_prob[:, k], expected, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 5d) Sufficient-statistics covariance identities (manual derivation)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_sufficient_statistics_covariance_identities(cov_type):
    rng = np.random.RandomState(7)
    N, K, D = 200, 3, 4
    reg_covar = 1e-6

    X = rng.randn(N, D)
    resp = rng.rand(N, K)
    resp /= resp.sum(axis=1, keepdims=True)

    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = (resp.T @ X) / nk[:, None]

    if cov_type == "full":
        cov_ref = np.empty((K, D, D), dtype=X.dtype)
        for k in range(K):
            diff = X - means[k]
            cov_ref[k] = (resp[:, k][:, None] * diff).T @ diff / nk[k]
            cov_ref[k].flat[:: D + 1] += reg_covar
    elif cov_type == "tied":
        cov_full = np.empty((K, D, D), dtype=X.dtype)
        for k in range(K):
            diff = X - means[k]
            cov_full[k] = (resp[:, k][:, None] * diff).T @ diff / nk[k]
        cov_ref = (nk[:, None, None] * cov_full).sum(axis=0) / nk.sum()
        cov_ref.flat[:: D + 1] += reg_covar
    elif cov_type == "diag":
        avg_X2 = (resp.T @ (X * X)) / nk[:, None]
        avg_means2 = means * means
        cov_ref = avg_X2 - avg_means2 + reg_covar
    else:  # spherical
        avg_X2 = (resp.T @ (X * X)) / nk[:, None]
        avg_means2 = means * means
        diag_cov = avg_X2 - avg_means2 + reg_covar
        cov_ref = diag_cov.mean(axis=1)

    X_t = torch.as_tensor(X, dtype=torch.float64)
    log_resp_t = torch.log(torch.as_tensor(resp, dtype=torch.float64))
    dummy_means, dummy_cov, dummy_weights = _make_dummy_params(K, D, cov_type)
    _, cov_t, _ = _maximization_step(
        X_t, dummy_means, dummy_cov, dummy_weights, log_resp_t, cov_type, reg_covar=reg_covar
    )

    np.testing.assert_allclose(cov_t.cpu().numpy(), cov_ref, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 5e) Whitening identity (catches transpose / wrong-side prec_chol bugs)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", ["full", "tied"])
def test_whitening_identity_matches_mahalanobis(cov_type):
    rng = np.random.RandomState(14)
    N, K, D = 128, 3, 5

    X = rng.randn(N, D)
    means = rng.randn(K, D)
    cov = _random_cov(rng, K, D, cov_type, reg=1e-6)

    cov_t = torch.as_tensor(cov, dtype=torch.float64)
    prec_chol_t = _torch_compute_precisions_cholesky(cov_t, cov_type)

    # Compare, per component:
    # ||(X - mu) @ P^T||^2 == (X - mu)^T inv(cov) (X - mu)
    if cov_type == "tied":
        inv_cov = linalg.inv(cov)
        P = prec_chol_t.cpu().numpy()  # (D,D) lower
        for k in range(K):
            diff = X - means[k]  # (N,D)
            whiten_sq = np.sum((diff @ P.T) ** 2, axis=1)
            mahal = np.sum(diff @ inv_cov * diff, axis=1)
            np.testing.assert_allclose(whiten_sq, mahal, rtol=1e-6, atol=1e-6)
    else:  # full
        P_all = prec_chol_t.cpu().numpy()  # (K,D,D) lower
        for k in range(K):
            inv_cov = linalg.inv(cov[k])
            diff = X - means[k]
            whiten_sq = np.sum((diff @ P_all[k].T) ** 2, axis=1)
            mahal = np.sum(diff @ inv_cov * diff, axis=1)
            np.testing.assert_allclose(whiten_sq, mahal, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 5h) Cross covariance-type identities (full/diag/tied/spherical)
# ---------------------------------------------------------------------
def test_cross_covariance_type_identities():
    rng = np.random.RandomState(12)
    N, K, D = 250, 4, 5
    reg_covar = 1e-6

    X = rng.randn(N, D)
    resp = rng.rand(N, K)
    resp /= resp.sum(axis=1, keepdims=True)

    X_t = torch.as_tensor(X, dtype=torch.float64)
    log_resp_t = torch.log(torch.as_tensor(resp, dtype=torch.float64))

    # Full covariances
    dummy_means, dummy_cov, dummy_weights = _make_dummy_params(K, D, "full")
    _, cov_full_t, _ = _maximization_step(
        X_t, dummy_means, dummy_cov, dummy_weights, log_resp_t, "full", reg_covar=reg_covar
    )
    cov_full = cov_full_t.cpu().numpy()  # (K,D,D)

    # Diag covariances
    dummy_means, dummy_cov, dummy_weights = _make_dummy_params(K, D, "diag")
    _, cov_diag_t, _ = _maximization_step(
        X_t, dummy_means, dummy_cov, dummy_weights, log_resp_t, "diag", reg_covar=reg_covar
    )
    cov_diag = cov_diag_t.cpu().numpy()  # (K,D)

    # Tied covariances
    dummy_means, dummy_cov, dummy_weights = _make_dummy_params(K, D, "tied")
    _, cov_tied_t, _ = _maximization_step(
        X_t, dummy_means, dummy_cov, dummy_weights, log_resp_t, "tied", reg_covar=reg_covar
    )
    cov_tied = cov_tied_t.cpu().numpy()  # (D,D)

    # Spherical covariances
    dummy_means, dummy_cov, dummy_weights = _make_dummy_params(K, D, "spherical")
    _, cov_sph_t, _ = _maximization_step(
        X_t, dummy_means, dummy_cov, dummy_weights, log_resp_t, "spherical", reg_covar=reg_covar
    )
    cov_sph = cov_sph_t.cpu().numpy()  # (K,)

    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps

    np.testing.assert_allclose(cov_diag, np.diagonal(cov_full, axis1=1, axis2=2), rtol=1e-6, atol=1e-6)

    cov_tied_ref = (nk[:, None, None] * cov_full).sum(axis=0) / nk.sum()
    np.testing.assert_allclose(cov_tied, cov_tied_ref, rtol=1e-6, atol=1e-6)

    cov_sph_ref = cov_diag.mean(axis=1)
    np.testing.assert_allclose(cov_sph, cov_sph_ref, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 5f) Label-switching invariance (permute components)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_label_switching_invariance(cov_type):
    rng = np.random.RandomState(8)
    N, K, D = 120, 4, 3

    X = rng.randn(N, D)
    means = rng.randn(K, D)
    cov = _random_cov(rng, K, D, cov_type, reg=1e-6)
    weights = rng.rand(K)
    weights /= weights.sum()

    perm = rng.permutation(K)

    X_t = torch.as_tensor(X, dtype=torch.float64)
    means_t = torch.as_tensor(means, dtype=torch.float64)
    cov_t = torch.as_tensor(cov, dtype=torch.float64)
    weights_t = torch.as_tensor(weights, dtype=torch.float64)

    log_prob = _torch_log_prob(X_t, means_t, cov_t, cov_type).cpu().numpy()
    resp = _torch_expectation_step(X_t, means_t, cov_t, weights_t, cov_type).cpu().numpy()

    means_p = means[perm]
    weights_p = weights[perm]
    if cov_type in ("full", "diag", "spherical"):
        cov_p = cov[perm]
    else:  # tied
        cov_p = cov

    means_p_t = torch.as_tensor(means_p, dtype=torch.float64)
    cov_p_t = torch.as_tensor(cov_p, dtype=torch.float64)
    weights_p_t = torch.as_tensor(weights_p, dtype=torch.float64)

    log_prob_p = _torch_log_prob(X_t, means_p_t, cov_p_t, cov_type).cpu().numpy()
    resp_p = _torch_expectation_step(X_t, means_p_t, cov_p_t, weights_p_t, cov_type).cpu().numpy()

    np.testing.assert_allclose(log_prob[:, perm], log_prob_p, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(resp[:, perm], resp_p, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 6) Degenerate data requires regularization for finite numbers
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_degenerate_data_requires_regularization_for_finite_numbers(cov_type):
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5
    K = n_samples  # nasty case like official test

    X = np.vstack(
        (np.ones((n_samples // 2, n_features)), np.zeros((n_samples // 2, n_features)))
    ).astype(np.float64)

    means0 = rng.randn(K, n_features)
    cov0 = _random_cov(rng, K, n_features, cov_type, reg=0.0)
    weights0 = rng.rand(K)
    weights0 /= weights0.sum()

    X_t = torch.as_tensor(X, dtype=torch.float64)
    dummy_means, dummy_cov, dummy_weights = _make_dummy_params(K, n_features, cov_type)

    log_resp_init = torch.log(torch.full((n_samples, K), 1.0 / K, dtype=torch.float64))
    means1, cov1, weights1 = _maximization_step(
        X_t, dummy_means, dummy_cov, dummy_weights, log_resp_init, cov_type, reg_covar=1e-6
    )
    resp1 = _torch_expectation_step(X_t, means1, cov1, weights1, cov_type)
    ll1 = _torch_log_likelihood_mean(X_t, means1, cov1, weights1, cov_type)

    assert torch.isfinite(resp1).all().item(), "Regularization should produce finite responsibilities"
    assert np.isfinite(ll1), "Regularization should produce finite log likelihood"


# ---------------------------------------------------------------------
# 7) Monotonic likelihood (kernel-level) -- FIXED (no log(resp))
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_torch_em_monotonic_likelihood_like_official(cov_type):
    rng = np.random.RandomState(42)
    N, K, D = 500, 3, 2
    reg_covar = 1e-6

    X = rng.rand(N, D).astype(np.float64)
    means_t = torch.as_tensor(rng.randn(K, D).astype(np.float64), dtype=torch.float64)
    cov_t = torch.as_tensor(_random_cov(rng, K, D, cov_type, reg=reg_covar).astype(np.float64), dtype=torch.float64)
    weights_t = torch.as_tensor(rng.rand(K).astype(np.float64), dtype=torch.float64)
    weights_t /= weights_t.sum()

    X_t = torch.as_tensor(X, dtype=torch.float64)

    ll_prev = -np.inf
    for _ in range(50):
        prec_chol_t = _torch_compute_precisions_cholesky(cov_t, cov_type)
        _, log_resp_t = _expectation_step(X_t, means_t, prec_chol_t, weights_t, cov_type)

        means_t, cov_t, weights_t = _maximization_step(
            X_t, means_t, cov_t, weights_t, log_resp_t, cov_type, reg_covar=reg_covar
        )
        ll = _torch_log_likelihood_mean(X_t, means_t, cov_t, weights_t, cov_type)
        assert ll >= ll_prev - 1e-10
        ll_prev = ll


# ---------------------------------------------------------------------
# 8) GPU parity (CPU vs CUDA) for log_prob and responsibilities
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_parity_logprob_and_resp(cov_type):
    rng = np.random.RandomState(123)
    N, K, D = 128, 3, 4

    X = rng.randn(N, D)
    means = rng.randn(K, D)
    cov = _random_cov(rng, K, D, cov_type, reg=1e-6)
    weights = rng.rand(K)
    weights /= weights.sum()

    X_cpu = torch.as_tensor(X, dtype=torch.float64)
    means_cpu = torch.as_tensor(means, dtype=torch.float64)
    cov_cpu = torch.as_tensor(cov, dtype=torch.float64)
    weights_cpu = torch.as_tensor(weights, dtype=torch.float64)

    X_gpu = X_cpu.cuda()
    means_gpu = means_cpu.cuda()
    cov_gpu = cov_cpu.cuda()
    weights_gpu = weights_cpu.cuda()

    logp_cpu = _torch_log_prob(X_cpu, means_cpu, cov_cpu, cov_type).cpu().numpy()
    logp_gpu = _torch_log_prob(X_gpu, means_gpu, cov_gpu, cov_type).cpu().numpy()
    np.testing.assert_allclose(logp_cpu, logp_gpu, rtol=1e-5, atol=1e-6)

    resp_cpu = _torch_expectation_step(X_cpu, means_cpu, cov_cpu, weights_cpu, cov_type).cpu().numpy()
    resp_gpu = _torch_expectation_step(X_gpu, means_gpu, cov_gpu, weights_gpu, cov_type).cpu().numpy()
    np.testing.assert_allclose(resp_cpu, resp_gpu, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------
# 9) Gradient checks (autograd consistency)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_gradcheck_log_prob_cov_types(cov_type):
    """Test gradients w.r.t. X and means (main use case), not covariances."""
    torch.manual_seed(0)
    N, K, D = 5, 2, 3

    X = torch.randn(N, D, dtype=torch.float64, requires_grad=True)
    means = torch.randn(K, D, dtype=torch.float64, requires_grad=True)

    if cov_type == "diag":
        cov = (0.5 + torch.rand(K, D, dtype=torch.float64)) ** 2
    elif cov_type == "spherical":
        cov = (0.5 + torch.rand(K, dtype=torch.float64)) ** 2
    elif cov_type == "tied":
        A = torch.randn(D, D, dtype=torch.float64)
        cov = (A @ A.T) / D + 1e-3 * torch.eye(D, dtype=torch.float64)
    else:  # full
        cov_list = []
        for _ in range(K):
            A = torch.randn(D, D, dtype=torch.float64)
            cov_list.append((A @ A.T) / D + 1e-3 * torch.eye(D, dtype=torch.float64))
        cov = torch.stack(cov_list, dim=0)

    def fn(X_i, means_i):
        return _torch_log_prob(X_i, means_i, cov, cov_type).sum()

    torch.autograd.gradcheck(fn, (X, means), eps=1e-6, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------
# 10) Extreme-value stability (large/small scales)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_extreme_value_stability(cov_type):
    rng = np.random.RandomState(7)
    N, K, D = 64, 3, 6

    X = (rng.randn(N, D) * 1e6).astype(np.float64)
    means = (rng.randn(K, D) * 1e6).astype(np.float64)

    if cov_type == "diag":
        cov = (1e-12 + rng.rand(K, D) * 1e-6).astype(np.float64)
    elif cov_type == "spherical":
        cov = (1e-12 + rng.rand(K) * 1e-6).astype(np.float64)
    elif cov_type == "tied":
        cov = (_random_spd(rng, D) * 1e-6 + 1e-12 * np.eye(D)).astype(np.float64)
    else:  # full
        cov = np.stack([_random_spd(rng, D) * 1e-6 + 1e-12 * np.eye(D) for _ in range(K)], axis=0)

    X_t = torch.as_tensor(X, dtype=torch.float64)
    means_t = torch.as_tensor(means, dtype=torch.float64)
    cov_t = torch.as_tensor(cov, dtype=torch.float64)

    log_prob = _torch_log_prob(X_t, means_t, cov_t, cov_type)
    assert torch.isfinite(log_prob).all().item()


# ---------------------------------------------------------------------
# 11) Randomized fuzzing (multiple seeds, shapes)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_random_fuzzing_properties(seed, cov_type):
    rng = np.random.RandomState(seed)
    N = rng.randint(50, 200)
    D = rng.randint(2, 8)
    K = rng.randint(2, 6)

    X = rng.randn(N, D)
    means = rng.randn(K, D)
    cov = _random_cov(rng, K, D, cov_type, reg=1e-6)
    weights = rng.rand(K)
    weights /= weights.sum()

    X_t = torch.as_tensor(X, dtype=torch.float64)
    means_t = torch.as_tensor(means, dtype=torch.float64)
    cov_t = torch.as_tensor(cov, dtype=torch.float64)
    weights_t = torch.as_tensor(weights, dtype=torch.float64)

    log_prob = _torch_log_prob(X_t, means_t, cov_t, cov_type)
    resp = _torch_expectation_step(X_t, means_t, cov_t, weights_t, cov_type)

    assert torch.isfinite(log_prob).all().item()
    assert torch.isfinite(resp).all().item()
    np.testing.assert_allclose(resp.sum(dim=1).cpu().numpy(), np.ones(N), atol=1e-10)


# ---------------------------------------------------------------------
# 12) Reproducibility (deterministic settings, same outputs)
# ---------------------------------------------------------------------
def test_reproducibility_same_seed_same_outputs():
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_cudnn_det = torch.backends.cudnn.deterministic
    prev_cudnn_bench = torch.backends.cudnn.benchmark

    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        seed = 1234
        np.random.seed(seed)
        torch.manual_seed(seed)

        N, K, D = 128, 3, 4
        cov_type = "full"

        X = torch.randn(N, D, dtype=torch.float64)
        means = torch.randn(K, D, dtype=torch.float64)
        cov = _random_cov(np.random.RandomState(seed), K, D, cov_type, reg=1e-6)
        weights = torch.rand(K, dtype=torch.float64)
        weights /= weights.sum()

        cov_t = torch.as_tensor(cov, dtype=torch.float64)

        logp1 = _torch_log_prob(X, means, cov_t, cov_type)
        resp1 = _torch_expectation_step(X, means, cov_t, weights, cov_type)

        np.random.seed(seed)
        torch.manual_seed(seed)

        X2 = torch.randn(N, D, dtype=torch.float64)
        means2 = torch.randn(K, D, dtype=torch.float64)
        cov2 = _random_cov(np.random.RandomState(seed), K, D, cov_type, reg=1e-6)
        weights2 = torch.rand(K, dtype=torch.float64)
        weights2 /= weights2.sum()
        cov_t2 = torch.as_tensor(cov2, dtype=torch.float64)

        logp2 = _torch_log_prob(X2, means2, cov_t2, cov_type)
        resp2 = _torch_expectation_step(X2, means2, cov_t2, weights2, cov_type)

        torch.testing.assert_close(logp1, logp2, rtol=0, atol=0)
        torch.testing.assert_close(resp1, resp2, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(prev_det)
        torch.backends.cudnn.deterministic = prev_cudnn_det
        torch.backends.cudnn.benchmark = prev_cudnn_bench
# ---------------------------------------------------------------------
# 13) Empirical KL divergence between sklearn and torch implementations
# ---------------------------------------------------------------------

@pytest.mark.parametrize("cov_type", COVARIANCE_TYPE)
def test_empirical_kl_divergence_matches_sklearn(cov_type):
    rng = np.random.RandomState(21)
    N_samples = 5000
    K, D = 4, 3

    means = rng.randn(K, D)
    cov = _random_cov(rng, K, D, cov_type, reg=1e-6)
    weights = rng.rand(K)
    weights /= weights.sum()

    component_assignments = rng.choice(K, size=N_samples, p=weights)
    X_samples = np.zeros((N_samples, D))

    for k in range(K):
        mask = component_assignments == k
        n_k = mask.sum()
        if n_k == 0:
            continue

        mean_k = means[k]
        if cov_type == "full":
            cov_k = cov[k]
        elif cov_type == "tied":
            cov_k = cov
        elif cov_type == "diag":
            cov_k = np.diag(cov[k])
        elif cov_type == "spherical":
            cov_k = cov[k] * np.eye(D)
        else:
            raise ValueError(cov_type)

        X_samples[mask] = rng.multivariate_normal(mean_k, cov_k, size=n_k)

    # sklearn log P(x)
    prec_chol_sk = _compute_precision_cholesky(cov, cov_type)
    log_prob_sk = _sk_estimate_log_gaussian_prob(X_samples, means, prec_chol_sk, cov_type)
    ll_sk = np.logaddexp.reduce(log_prob_sk + np.log(weights)[None, :], axis=1)

    # torch log Q(x)
    X_t = torch.from_numpy(X_samples)
    means_t = torch.from_numpy(means)
    cov_t = torch.from_numpy(cov)
    weights_t = torch.from_numpy(weights)

    log_prob_t = _torch_log_prob(X_t, means_t, cov_t, cov_type)
    ll_t = torch.logsumexp(log_prob_t + torch.log(weights_t)[None, :], dim=1).cpu().numpy()

    # Strongest correctness check (pointwise)
    np.testing.assert_allclose(ll_t, ll_sk, rtol=1e-6, atol=1e-6)

    # KL estimate (will now be extremely close to 0)
    kl_empirical = float(np.mean(ll_sk - ll_t))
    assert abs(kl_empirical) < 1e-6
