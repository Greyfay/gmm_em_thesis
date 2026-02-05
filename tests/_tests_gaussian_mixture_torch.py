# tests/test_torch_gmm_em.py
import sys
import os

# Add parent directory to path so we can import implementation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from implementation._torch_gmm_em import (
    _expectation_step_precchol,
    _maximization_step as _maximization_step_precchol,
    _estimate_log_gaussian_prob_diag_precchol,
    _estimate_log_gaussian_prob_full_precchol,
    _estimate_log_gaussian_prob_tied_precchol,
    _estimate_log_gaussian_prob_spherical_precchol,
    _compute_precisions_cholesky,
    _compute_precisions,
)


def _random_seed():
    """Generate a random seed between 1 and 1000."""
    return np.random.default_rng().integers(1, 1001)


class RandomData:
    """Generate random GMM data for testing with compact covariance storage."""
    def __init__(self, rng, n_samples=200, n_components=2, n_features=2, covariance_type=None):
        self.n_samples = int(n_samples)
        self.n_components = int(n_components)
        self.n_features = int(n_features)

        # weights on simplex
        w = rng.rand(self.n_components).astype(np.float64)
        self.weights = w / w.sum()

        # means spread out
        self.means = (rng.rand(self.n_components, self.n_features).astype(np.float64) * 50.0)

        allowed = ("diag", "full", "spherical", "tied")
        self.covariance_type = rng.choice(allowed) if covariance_type is None else covariance_type
        if self.covariance_type not in allowed:
            raise ValueError(f"covariance_type must be one of {allowed}, got {self.covariance_type}")

        self.cov = self._make_covariance(rng)  # compact form
        self.X = self._generate_samples(rng)

    def _make_covariance(self, rng):
        K, D = self.n_components, self.n_features
        reg = 1e-6

        # variance range ~ [0.25, 2.25)
        def rand_var(shape):
            return ((0.5 + rng.rand(*shape)) ** 2).astype(np.float64)

        if self.covariance_type == "diag":
            return rand_var((K, D))  # (K,D)

        if self.covariance_type == "spherical":
            return rand_var((K,))    # (K,)

        if self.covariance_type == "tied":
            A = rng.randn(D, D).astype(np.float64)
            C = A @ A.T
            C /= (np.trace(C) / D)
            C += reg * np.eye(D)
            return C  # (D,D)

        if self.covariance_type == "full":
            covs = []
            for _ in range(K):
                A = rng.randn(D, D).astype(np.float64)
                C = A @ A.T
                C /= (np.trace(C) / D)
                C += reg * np.eye(D)
                covs.append(C)
            return np.stack(covs, axis=0)  # (K,D,D)

        raise ValueError(self.covariance_type)

    def _generate_samples(self, rng):
        """Generate random samples from the Gaussian mixture using the chosen covariance type."""
        K, D = self.n_components, self.n_features
        samples = []

        # Cholesky factors where needed
        L_tied = None
        if self.covariance_type == "tied":
            L_tied = np.linalg.cholesky(self.cov)  # (D,D)

        L_full = None
        if self.covariance_type == "full":
            L_full = np.linalg.cholesky(self.cov)  # (K,D,D)

        for _ in range(self.n_samples):
            k = rng.choice(K, p=self.weights)
            z = rng.randn(D).astype(np.float64)

            if self.covariance_type == "diag":
                var = self.cov[k]                    # (D,)
                x = self.means[k] + z * np.sqrt(var)

            elif self.covariance_type == "spherical":
                var = self.cov[k]                    # scalar
                x = self.means[k] + z * np.sqrt(var)

            elif self.covariance_type == "tied":
                x = self.means[k] + (L_tied @ z)

            elif self.covariance_type == "full":
                x = self.means[k] + (L_full[k] @ z)

            samples.append(x)

        return np.array(samples, dtype=np.float64)

def log_likelihood_mean(X, means, cov, weights, cov_type):
    """Mean log-likelihood: E_n [ log sum_k pi_k N(x_n | ...) ] computed stably."""
    N, D = X.shape
    K, _ = means.shape

    prec_chol = _compute_precisions_cholesky(cov, cov_type)

    if cov_type == "diag":
        log_prob = _estimate_log_gaussian_prob_diag_precchol(X, means, prec_chol)
    elif cov_type == "spherical":
        log_prob = _estimate_log_gaussian_prob_spherical_precchol(X, means, prec_chol)
    elif cov_type == "tied":
        log_prob = _estimate_log_gaussian_prob_tied_precchol(X, means, prec_chol)
    else:
        log_prob = _estimate_log_gaussian_prob_full_precchol(X, means, prec_chol)

    log_weights = torch.log(weights).unsqueeze(0)  # (1,K)
    return torch.logsumexp(log_prob + log_weights, dim=1).mean()


def _expectation_step(X, means, cov, weights, cov_type):
    """Wrapper that accepts covariances and returns responsibilities."""
    prec_chol = _compute_precisions_cholesky(cov, cov_type)
    _, log_resp = _expectation_step_precchol(X, means, prec_chol, weights, cov_type)
    return log_resp.exp()


def _maximization_step(X, means, cov, weights, resp, cov_type, reg_covar: float = 1e-6):
    """Wrapper that accepts responsibilities and calls log-resp M-step."""
    tiny = torch.finfo(resp.dtype).tiny
    log_resp = torch.log(resp.clamp_min(tiny))
    return _maximization_step_precchol(X, means, cov, weights, log_resp, cov_type, reg_covar=reg_covar)


def compute_precision(cov, covariance_type):
    """Compute precision (inverse covariance) from covariance."""
    prec_chol = _compute_precisions_cholesky(cov, covariance_type)
    return _compute_precisions(prec_chol, covariance_type)


def test_different_seeds_generate_different_samples_printed():
    """Small, printed check that different RNG seeds yield different samples."""
    rng_seeds = np.random.default_rng()
    seed_a, seed_b = rng_seeds.choice(np.arange(1, 1001), size=2, replace=False)
    n_samples, n_components, n_features = 8, 2, 2

    rng_a = np.random.RandomState(int(seed_a))
    rng_b = np.random.RandomState(int(seed_b))

    data_a = RandomData(
        rng_a,
        n_samples=n_samples,
        n_components=n_components,
        n_features=n_features,
        covariance_type="diag",
    )
    data_b = RandomData(
        rng_b,
        n_samples=n_samples,
        n_components=n_components,
        n_features=n_features,
        covariance_type="diag",
    )

    print(f"\n[seed-diff] seed_a={int(seed_a)} samples:\n", data_a.X)
    print(f"[seed-diff] seed_b={int(seed_b)} samples:\n", data_b.X)

    assert not np.allclose(data_a.X, data_b.X), "Different seeds should produce different samples"


@pytest.mark.parametrize("covariance_type", ["diag", "spherical", "tied", "full"])
def test_monotonic_likelihood_all_cov_types(covariance_type):
    """Test that EM does not decrease mean log-likelihood each iteration."""
    rng = np.random.RandomState(_random_seed())
    rand_data = RandomData(rng, n_samples=200, n_components=2, n_features=2, covariance_type=covariance_type)

    X = torch.from_numpy(rand_data.X).float()
    means = torch.from_numpy(rand_data.means).float()
    cov = torch.from_numpy(rand_data.cov).float()
    weights = torch.from_numpy(rand_data.weights).float()

    print(f"\n[{covariance_type}] Initial params:")
    print(f"  means shape: {means.shape}, values:\n{means}")
    print(f"  cov shape: {cov.shape}, values:\n{cov}")
    print(f"  weights: {weights}")

    likelihoods = []
    #compute initial likelihood
    ll = log_likelihood_mean(X, means, cov, weights, cov_type=covariance_type)
    likelihoods.append(ll.item())
    print(f"  Initial likelihood: {ll.item():.6f}")

    for iteration in range(10):
        # E-step
        resp = _expectation_step(X, means, cov, weights, cov_type=covariance_type)

        print(f"\n  Iteration {iteration+1}:")
        print(f"    resp sum per sample (should be 1.0): {resp.sum(dim=1)[:5]}")  # first 5
        print(f"    resp stats: min={resp.min():.6f}, max={resp.max():.6f}")

        # M-step
        means_new, cov_new, weights_new = _maximization_step(X, means, cov, weights, resp, cov_type=covariance_type)
        
        print(f"    means after M-step:\n{means_new}")
        print(f"    cov after M-step:\n{cov_new}")
        print(f"    weights after M-step: {weights_new}")
        print(f"    cov min: {cov_new.min():.6f}, max: {cov_new.max():.6f}")

         # Check for NaN/Inf
        if not torch.isfinite(means_new).all():
            print(f"    ERROR: means has NaN/Inf!")
        if not torch.isfinite(cov_new).all():
            print(f"    ERROR: cov has NaN/Inf!")
        if not torch.isfinite(weights_new).all():
            print(f"    ERROR: weights has NaN/Inf!")
        
        means, cov, weights = means_new, cov_new, weights_new

        # Compute likelihood after update
        ll = log_likelihood_mean(X, means, cov, weights, cov_type=covariance_type)
        likelihoods.append(ll.item())

        print(f"    Likelihood after update: {ll.item():.6f}")

        if iteration == 0 and likelihoods[-1] < likelihoods[-2]:
            print(f"    FAILED on first iteration!")
            break

    for i in range(1, len(likelihoods)):
        assert likelihoods[i] >= likelihoods[i - 1] - 1e-7, \
            f"[{covariance_type}] decreased at iter {i}: {likelihoods[i-1]} -> {likelihoods[i]}"


@pytest.mark.parametrize("covariance_type", ["diag", "spherical", "tied", "full"])
def test_responsibilities_sum_to_one(covariance_type):
    """Test that responsibilities sum to 1 across clusters."""
    rng = np.random.RandomState(_random_seed())
    rand_data = RandomData(rng, n_samples=50, n_components=3, n_features=2, covariance_type=covariance_type)

    X = torch.from_numpy(rand_data.X).float()
    means = torch.from_numpy(rand_data.means).float()
    cov = torch.from_numpy(rand_data.cov).float()
    weights = torch.from_numpy(rand_data.weights).float()

    resp = _expectation_step(X, means, cov, weights, cov_type=covariance_type)
    resp_sums = resp.sum(dim=1)
    assert torch.allclose(resp_sums, torch.ones_like(resp_sums), atol=1e-5)

@pytest.mark.parametrize("covariance_type", ["diag", "spherical", "tied", "full"])
def test_precision_computation(covariance_type):
    """Test that precision (inverse covariance) is computed correctly."""
    rng = np.random.RandomState(_random_seed())
    K, D = 3, 2
    
    # Generate covariance based on type
    if covariance_type == "diag":
        # (K, D) - diagonal variances
        cov = torch.tensor([[2.0, 4.0],
                            [1.0, 3.0],
                            [0.5, 2.5]], dtype=torch.float64)
        expected_prec = torch.tensor([[0.5, 0.25],
                                       [1.0, 0.333333],
                                       [2.0, 0.4]], dtype=torch.float64)
        
    elif covariance_type == "spherical":
        # (K,) - single variance per component
        cov = torch.tensor([2.0, 4.0, 0.5], dtype=torch.float64)
        expected_prec = torch.tensor([0.5, 0.25, 2.0], dtype=torch.float64)
        
    elif covariance_type == "tied":
        # (D, D) - shared full covariance
        cov = torch.tensor([[2.0, 0.5],
                            [0.5, 3.0]], dtype=torch.float64)
        # Manual inverse: 1/(2*3 - 0.5*0.5) * [[3, -0.5], [-0.5, 2]]
        det = 2.0 * 3.0 - 0.5 * 0.5  # = 5.75
        expected_prec = torch.tensor([[3.0/det, -0.5/det],
                                       [-0.5/det, 2.0/det]], dtype=torch.float64)
        
    elif covariance_type == "full":
        # (K, D, D) - per-component full covariance
        cov = torch.tensor([[[2.0, 0.5],
                             [0.5, 3.0]],
                            [[4.0, 0.0],
                             [0.0, 1.0]],
                            [[1.5, -0.3],
                             [-0.3, 2.0]]], dtype=torch.float64)
        
        # Compute expected via torch.linalg.inv
        expected_prec = torch.linalg.inv(cov)
    
    # Compute precision using your implementation
    prec_chol = compute_precision(cov, covariance_type)
    
    # For diagonal/spherical, precision is direct
    if covariance_type in ["diag", "spherical"]:
        computed_prec = prec_chol
    else:
        # For tied/full, we need to reconstruct precision from Cholesky
        # prec = (L^{-1})^T @ L^{-1}
        if covariance_type == "tied":
            L_inv = torch.linalg.inv(torch.linalg.cholesky(cov))
            computed_prec = L_inv.T @ L_inv
        else:  # full
            computed_prec = torch.zeros_like(cov)
            for k in range(K):
                L_inv = torch.linalg.inv(torch.linalg.cholesky(cov[k]))
                computed_prec[k] = L_inv.T @ L_inv
    
    # Check if computed precision matches expected
    assert torch.allclose(computed_prec, expected_prec, atol=1e-5, rtol=1e-4), \
        f"[{covariance_type}] Precision mismatch:\nComputed:\n{computed_prec}\nExpected:\n{expected_prec}"


def test_precision_properties():
    """Test mathematical properties of precision matrices."""
    # Test that Σ @ Σ^{-1} = I
    K, D = 2, 3
    
    # Create a positive definite covariance
    A = torch.randn(D, D, dtype=torch.float64)
    cov_tied = A @ A.T + torch.eye(D, dtype=torch.float64) * 0.1
        
    # Compute precision
    prec_chol = compute_precision(cov_tied, "tied")
    
    # Reconstruct precision matrix
    L_inv = torch.linalg.inv(torch.linalg.cholesky(cov_tied))
    precision = L_inv.T @ L_inv
    
    # Test: Σ @ Σ^{-1} should equal identity
    product = cov_tied @ precision
    identity = torch.eye(D, dtype=torch.float64)
    
    assert torch.allclose(product, identity, atol=1e-5), \
        f"Covariance @ Precision != Identity:\n{product}\nvs\n{identity}"
    
    # Test: precision should be symmetric
    assert torch.allclose(precision, precision.T, atol=1e-10), \
        "Precision matrix is not symmetric"
    
    # Test: precision should be positive definite (all eigenvalues > 0)
    eigenvalues = torch.linalg.eigvalsh(precision)
    assert (eigenvalues > 0).all(), \
        f"Precision has non-positive eigenvalues: {eigenvalues}"
    
def test_single_component():
    """Test K=1 (single Gaussian, no mixture)."""
    rng = np.random.RandomState(_random_seed())
    rand_data = RandomData(rng, n_samples=100, n_components=1, n_features=3, covariance_type="diag")
    
    X = torch.from_numpy(rand_data.X).float()
    means = torch.from_numpy(rand_data.means).float()
    cov = torch.from_numpy(rand_data.cov).float()
    weights = torch.from_numpy(rand_data.weights).float()
    
    # Should still work with K=1
    resp = _expectation_step(X, means, cov, weights, cov_type="diag")
    assert resp.shape == (100, 1)
    assert torch.allclose(resp, torch.ones_like(resp))  # All samples belong to single component


def test_high_dimensional():
    """Test with D >> K (many features, few components)."""
    rng = np.random.RandomState(_random_seed())
    rand_data = RandomData(rng, n_samples=100, n_components=2, n_features=20, covariance_type="diag")
    
    X = torch.from_numpy(rand_data.X).float()
    means = torch.from_numpy(rand_data.means).float()
    cov = torch.from_numpy(rand_data.cov).float()
    weights = torch.from_numpy(rand_data.weights).float()
    
    for _ in range(5):
        resp = _expectation_step(X, means, cov, weights, cov_type="diag")
        means, cov, weights = _maximization_step(X, means, cov, weights, resp, cov_type="diag")
        
        # Should remain finite
        assert torch.isfinite(means).all()
        assert torch.isfinite(cov).all()
        assert (cov > 0).all()


def test_few_samples():
    """Test with N < K (fewer samples than components)."""
    rng = np.random.RandomState(42)
    
    # Create minimal data
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)  # N=2
    means = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)  # K=3
    cov = torch.ones(3, 2, dtype=torch.float32)  # Diagonal
    weights = torch.tensor([0.33, 0.33, 0.34], dtype=torch.float32)
    
    # Should not crash
    resp = _expectation_step(X, means, cov, weights, cov_type="diag")
    means_new, cov_new, weights_new = _maximization_step(X, means, cov, weights, resp, cov_type="diag")
    
    assert torch.isfinite(means_new).all()
    assert torch.isfinite(cov_new).all()


def test_nearly_empty_cluster():
    """Test when one cluster gets very few samples assigned."""
    rng = np.random.RandomState(42)
    
    # Create two well-separated clusters
    X1 = torch.randn(95, 2) + torch.tensor([0.0, 0.0])
    X2 = torch.randn(5, 2) + torch.tensor([20.0, 20.0])  # Far away, few samples
    X = torch.cat([X1, X2], dim=0)
    
    # Initialize with reasonable means
    means = torch.tensor([[0.0, 0.0], [20.0, 20.0]], dtype=torch.float32)
    cov = torch.ones(2, 2, dtype=torch.float32)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
    
    # Run a few iterations
    for _ in range(3):
        resp = _expectation_step(X, means, cov, weights, cov_type="diag")
        means_new, cov_new, weights_new = _maximization_step(X, means, cov, weights, resp, cov_type="diag")
        
        # Check component 1 (small cluster) doesn't collapse
        assert (cov_new[1] > 1e-6).all(), "Small cluster variance collapsed"
        assert weights_new[1] > 0, "Small cluster weight became zero"
        
        means, cov, weights = means_new, cov_new, weights_new


def test_identical_samples():
    """Test with all samples at the same point (zero variance case)."""
    X = torch.ones(50, 2, dtype=torch.float32) * 5.0  # All identical
    means = torch.tensor([[4.0, 4.0], [6.0, 6.0]], dtype=torch.float32)
    cov = torch.ones(2, 2, dtype=torch.float32)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
    
    # Should handle gracefully with regularization
    resp = _expectation_step(X, means, cov, weights, cov_type="diag")
    means_new, cov_new, weights_new = _maximization_step(X, means, cov, weights, resp, cov_type="diag", reg_covar=1e-3)
    
    # Means should converge to data point
    assert torch.allclose(means_new[0], torch.tensor([5.0, 5.0]), atol=0.1)
    
    # Variance should be at regularization floor
    assert (cov_new >= 1e-3).all(), "Variance below regularization threshold"


@pytest.mark.parametrize("covariance_type", ["diag", "spherical", "tied", "full"])
def test_extreme_separations(covariance_type):
    """Test with clusters very far apart."""
    rng = np.random.RandomState(_random_seed())
    
    # Two clusters 1000 units apart
    X1 = torch.randn(100, 2, dtype=torch.float32) * 0.1 + torch.tensor([0.0, 0.0])
    X2 = torch.randn(100, 2, dtype=torch.float32) * 0.1 + torch.tensor([1000.0, 1000.0])
    X = torch.cat([X1, X2], dim=0)
    
    rand_data = RandomData(rng, n_samples=200, n_components=2, n_features=2, covariance_type=covariance_type)
    means = torch.from_numpy(rand_data.means).float()
    cov = torch.from_numpy(rand_data.cov).float()
    weights = torch.from_numpy(rand_data.weights).float()
    
    # Should handle without overflow
    resp = _expectation_step(X, means, cov, weights, cov_type=covariance_type)
    assert torch.isfinite(resp).all(), "Responsibilities contain NaN/Inf"
    assert torch.allclose(resp.sum(dim=1), torch.ones(200), atol=1e-5)
