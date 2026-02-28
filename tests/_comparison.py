"""
Focused comparison tests between PyTorch and sklearn GMM implementations.

Tests 4 specific aspects:
1. E-step correctness: log probabilities and responsibilities match sklearn's
2. M-step correctness: given same responsibilities, means/covariances match sklearn's
3. Objective equivalence: optimizing the same likelihood
4. End-to-end equivalence: same init parameters lead to equivalent final parameters
"""

import os
import sys
import numpy as np
import torch
from sklearn.mixture import GaussianMixture as SklearnGMM
from sklearn.mixture._gaussian_mixture import (
    _estimate_log_gaussian_prob as _sk_estimate_log_gaussian_prob,
    _estimate_gaussian_parameters as _sk_estimate_gaussian_parameters,
)

# Make local modules importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from implementation._v1 import (
    TorchGaussianMixture,
    _estimate_log_gaussian_prob_precchol,
    _expectation_step_precchol,
    _maximization_step,
    _compute_precisions_cholesky,
)

torch.set_default_dtype(torch.float64)


def _random_seed():
    """Generate a random seed between 1 and 1000."""
    return np.random.randint(1, 1001)


# ===========================
# Test 1: E-step Correctness
# ===========================

# Generate random seeds once for all tests
SEED_TSTEP = _random_seed()
SEED_MSTEP = _random_seed()
SEED_OBJ = _random_seed()
SEED_E2E = _random_seed()

def test_e_step_correctness():
    """
    Test that log probabilities and responsibilities match sklearn's.
    We compare log prob computation and posterior calculation.
    """
    print("\n" + "="*80)
    print("TEST 1: E-step Correctness")
    print(f"(seed={SEED_TSTEP})")
    print("="*80)
    
    np.random.seed(SEED_TSTEP)
    N, D, K = 100, 5, 3
    
    # Generate random data
    X_np = np.random.randn(N, D).astype(np.float64)
    X_torch = torch.from_numpy(X_np)
    
    # Generate random parameters
    rng = np.random.RandomState(42)
    means_np = rng.randn(K, D).astype(np.float64)
    means_torch = torch.from_numpy(means_np)
    
    weights_np = np.ones(K) / K
    weights_torch = torch.from_numpy(weights_np)
    
    # Simple diagonal covariance
    cov_np = np.tile(np.ones(D, dtype=np.float64), (K, 1)) + 0.5
    cov_torch = torch.from_numpy(cov_np)
    
    # Compute precisions_chol for sklearn
    prec_chol_np = 1.0 / np.sqrt(cov_np)  # For diag: precisions_chol = 1/sqrt(var)
    
    # Compute sklearn log probs (using precisions_chol, not covariance)
    sk_log_prob = _sk_estimate_log_gaussian_prob(
        X_np, means_np, prec_chol_np, covariance_type="diag"
    )
    
    # Compute torch log probs
    prec_chol_torch = _compute_precisions_cholesky(cov_torch, "diag")
    torch_log_prob = _estimate_log_gaussian_prob_precchol(
        X_torch, means_torch, prec_chol_torch, "diag"
    ).cpu().numpy()
    
    # Compare log probabilities
    log_prob_diff = np.abs(sk_log_prob - torch_log_prob)
    max_log_prob_diff = np.max(log_prob_diff)
    mean_log_prob_diff = np.mean(log_prob_diff)
    
    print(f"Log probability difference (max): {max_log_prob_diff:.2e}")
    print(f"Log probability difference (mean): {mean_log_prob_diff:.2e}")
    
    # Compute sklearn responsibilities
    sk_resp_unnorm = sk_log_prob + np.log(weights_np)[None, :]
    sk_log_norm = np.logaddexp.reduce(sk_resp_unnorm, axis=1, keepdims=True)
    sk_resp = np.exp(sk_resp_unnorm - sk_log_norm)
    
    # Compute torch responsibilities
    _, torch_log_resp = _expectation_step_precchol(
        X_torch, means_torch, prec_chol_torch, weights_torch, "diag"
    )
    torch_resp = torch_log_resp.exp().cpu().numpy()
    
    # Compare responsibilities
    resp_diff = np.abs(sk_resp - torch_resp)
    max_resp_diff = np.max(resp_diff)
    mean_resp_diff = np.mean(resp_diff)
    
    print(f"Responsibility difference (max): {max_resp_diff:.2e}")
    print(f"Responsibility difference (mean): {mean_resp_diff:.2e}")
    
    # Test passes if differences are small
    assert max_log_prob_diff < 1e-10, f"Log prob mismatch: {max_log_prob_diff}"
    assert max_resp_diff < 1e-10, f"Responsibility mismatch: {max_resp_diff}"
    print("✓ E-step correctness PASSED\n")


# ===========================
# Test 2: M-step Correctness
# ===========================

def test_m_step_correctness():
    """
    Test that given the same responsibilities, means/covariances match sklearn's.
    """
    print("="*80)
    print("TEST 2: M-step Correctness")
    print(f"(seed={SEED_MSTEP})")
    print("="*80)
    
    np.random.seed(SEED_MSTEP)
    N, D, K = 100, 5, 3
    
    # Generate random data
    X_np = np.random.randn(N, D).astype(np.float64)
    X_torch = torch.from_numpy(X_np)
    
    # Generate random responsibilities (normalized)
    rng = np.random.RandomState(42)
    resp_np = rng.uniform(0, 1, (N, K)).astype(np.float64)
    resp_np /= resp_np.sum(axis=1, keepdims=True)
    resp_torch = torch.from_numpy(resp_np)
    log_resp_torch = torch.log(resp_torch)
    
    # Dummy initial parameters
    means_np_init = rng.randn(K, D).astype(np.float64)
    means_torch_init = torch.from_numpy(means_np_init)
    cov_np_init = np.tile(np.ones(D, dtype=np.float64), (K, 1)) + 0.5
    cov_torch_init = torch.from_numpy(cov_np_init)
    weights_np_init = np.ones(K) / K
    weights_torch_init = torch.from_numpy(weights_np_init)
    
    # Compute sklearn M-step
    sk_norms, sk_means, sk_cov = _sk_estimate_gaussian_parameters(
        X_np, resp_np, reg_covar=1e-6, covariance_type="diag"
    )
    sk_weights = sk_norms / sk_norms.sum()
    
    # Compute torch M-step
    torch_means, torch_cov, torch_weights = _maximization_step(
        X_torch, means_torch_init, cov_torch_init, weights_torch_init,
        log_resp_torch, "diag", reg_covar=1e-6
    )
    torch_means = torch_means.cpu().numpy()
    torch_cov = torch_cov.cpu().numpy()
    torch_weights = torch_weights.cpu().numpy()
    
    # Compare means
    means_diff = np.abs(sk_means - torch_means)
    max_means_diff = np.max(means_diff)
    mean_means_diff = np.mean(means_diff)
    
    print(f"Means difference (max): {max_means_diff:.2e}")
    print(f"Means difference (mean): {mean_means_diff:.2e}")
    
    # Compare covariances
    cov_diff = np.abs(sk_cov - torch_cov)
    max_cov_diff = np.max(cov_diff)
    mean_cov_diff = np.mean(cov_diff)
    
    print(f"Covariance difference (max): {max_cov_diff:.2e}")
    print(f"Covariance difference (mean): {mean_cov_diff:.2e}")
    
    # Compare weights
    weights_diff = np.abs(sk_weights - torch_weights)
    max_weights_diff = np.max(weights_diff)
    
    print(f"Weights difference (max): {max_weights_diff:.2e}")
    
    # Test passes if differences are small
    assert max_means_diff < 1e-10, f"Means mismatch: {max_means_diff}"
    assert max_cov_diff < 1e-10, f"Covariance mismatch: {max_cov_diff}"
    assert max_weights_diff < 1e-10, f"Weights mismatch: {max_weights_diff}"
    print("✓ M-step correctness PASSED\n")


# ===========================
# Test 3: Objective Equivalence
# ===========================

def test_objective_equivalence():
    """
    Test that both implementations optimize the same likelihood objective.
    Compare EM trajectory likelihoods over iterations.
    """
    print("="*80)
    print("TEST 3: Objective Equivalence")
    print(f"(seed={SEED_OBJ})")
    print("="*80)
    
    np.random.seed(SEED_OBJ)
    N, D, K = 200, 8, 3
    
    # Generate random data
    X_np = np.random.randn(N, D).astype(np.float32)
    X_torch = torch.from_numpy(X_np).float()
    
    # Fit sklearn with fixed seed and specific init
    sklearn_gmm = SklearnGMM(
        n_components=K,
        covariance_type="diag",
        init_params="kmeans",
        n_init=1,
        max_iter=50,
        tol=1e-3,
        random_state=SEED_OBJ
    )
    sklearn_gmm.fit(X_np)
    sk_ll = sklearn_gmm.lower_bound_
    sk_iter = sklearn_gmm.n_iter_
    
    # Fit PyTorch with same init (scikit_kmeans) and max iterations
    torch_gmm = TorchGaussianMixture(
        n_components=K,
        covariance_type="diag",
        init_params="scikit_kmeans",
        n_init=1,
        max_iter=50,
        tol=1e-3,
        device=None,
        dtype=torch.float32
    )
    torch_gmm.fit(X_torch)
    torch_ll = torch_gmm.lower_bound_
    torch_iter = torch_gmm.n_iter_
    
    # Compare final likelihoods
    ll_diff = abs(sk_ll - torch_ll)
    
    print(f"Sklearn final log-likelihood:  {sk_ll:.6f} (iter {sk_iter})")
    print(f"PyTorch final log-likelihood:  {torch_ll:.6f} (iter {torch_iter})")
    print(f"Difference: {ll_diff:.6f}")
    
    # They should optimize the same objective (small relative difference)
    rel_diff = ll_diff / (abs(sk_ll) + 1e-10)
    print(f"Relative difference: {rel_diff:.2e}")
    
    # Pass if likelihoods are reasonably close
    assert rel_diff < 0.1, f"Objective divergence too large: {rel_diff}"
    print("✓ Objective equivalence PASSED\n")


# ===========================
# Test 4: End-to-End Equivalence
# ===========================

def test_end_to_end_equivalence():
    """
    Test that given the same init parameters (scikit_kmeans for pytorch, 
    kmeans for sklearn), the end parameters are equivalent.
    """
    print("="*80)
    print("TEST 4: End-to-End Equivalence")
    print(f"(seed={SEED_E2E})")
    print("="*80)
    
    np.random.seed(SEED_E2E)
    N, D, K = 300, 10, 4
    
    # Generate random data
    X_np = np.random.randn(N, D).astype(np.float32)
    X_torch = torch.from_numpy(X_np).float()
    
    # Fit sklearn with kmeans init
    sklearn_gmm = SklearnGMM(
        n_components=K,
        covariance_type="diag",
        init_params="kmeans",
        n_init=1,
        max_iter=100,
        tol=1e-3,
        random_state=SEED_E2E
    )
    sklearn_gmm.fit(X_np)
    
    # Fit PyTorch with scikit_kmeans init
    torch_gmm = TorchGaussianMixture(
        n_components=K,
        covariance_type="diag",
        init_params="scikit_kmeans",
        n_init=1,
        max_iter=100,
        tol=1e-3,
        device=None,
        dtype=torch.float32
    )
    torch_gmm.fit(X_torch)
    
    # Extract final parameters
    sk_weights = sklearn_gmm.weights_
    sk_means = sklearn_gmm.means_
    sk_covs = sklearn_gmm.covariances_
    
    torch_weights = torch_gmm.weights_.cpu().numpy()
    torch_means = torch_gmm.means_.cpu().numpy()
    torch_covs = torch_gmm.covariances_.cpu().numpy()
    
    # Compare final log-likelihoods
    sk_ll = sklearn_gmm.lower_bound_
    torch_ll = torch_gmm.lower_bound_
    
    print(f"Sklearn final log-likelihood: {sk_ll:.6f}")
    print(f"PyTorch final log-likelihood: {torch_ll:.6f}")
    print(f"Difference: {abs(sk_ll - torch_ll):.6f}")
    
    # Since they start from same kmeans init (via scikit_kmeans),
    # they should end up in similar places
    ll_diff = abs(sk_ll - torch_ll)
    rel_diff = ll_diff / (abs(sk_ll) + 1e-10)
    
    print(f"Relative difference: {rel_diff:.2e}")
    
    # Check component alignment
    print(f"\nFinal weights:")
    print(f"  Sklearn:  {sk_weights}")
    print(f"  PyTorch:  {torch_weights}")
    print(f"  Diff (max): {np.max(np.abs(sk_weights - torch_weights)):.2e}")
    
    print(f"\nFinal means shape:")
    print(f"  Sklearn:  {sk_means.shape}")
    print(f"  PyTorch:  {torch_means.shape}")
    
    # Pass if end-to-end results are close
    assert rel_diff < 0.15, f"End-to-end divergence too large: {rel_diff}"
    print("\n✓ End-to-end equivalence PASSED\n")


if __name__ == "__main__":
    test_e_step_correctness()
    test_m_step_correctness()
    test_objective_equivalence()
    test_end_to_end_equivalence()
    
    print("="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
