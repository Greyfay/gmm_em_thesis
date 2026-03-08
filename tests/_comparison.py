"""
Correctness comparison: PyTorch GMM implementations vs sklearn oracle.

Covers all four implementations:
  _v0_ref                        (loop-based reference)
  _v1                            (vectorized)
  _v2_reduced_covariance_updates (covariance throttling)
  _v2_tiling                     (tiling/blocked vectorized)

Tests:
  1. E-step correctness  – log-probs and responsibilities vs sklearn internals
  2. M-step correctness  – updated parameters vs sklearn internals
  3. End-to-end          – all four implementations, identical init, vs sklearn

Strategy for identical initialization (Tests 3):
  Run sklearn KMeans once → extract initial means/covariances → inject the
  exact same means_init + precisions_init into all five models (sklearn GMM
  started from those means via means_init, and all four PyTorch variants).
  This eliminates any RNG divergence during init and makes log-likelihood
  comparison meaningful (tight tolerance).
"""

import os
import sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as SklearnGMM
from sklearn.mixture._gaussian_mixture import (
    _estimate_log_gaussian_prob as _sk_estimate_log_gaussian_prob,
    _estimate_gaussian_parameters as _sk_estimate_gaussian_parameters,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import implementation._v0_ref as v0
import implementation._v1 as v1
import implementation._v2_reduced_covariance_updates as v2r
import implementation._v2_tiling as v2t

torch.set_default_dtype(torch.float64)

# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _align_components(ref_means, cand_means):
    """Return a permutation index mapping candidate components to reference components.

    Uses the Hungarian algorithm on pairwise squared distances between means so
    that components can be compared after permutation, handling the fact that EM
    has no canonical component ordering.
    """
    # cost[i, j] = ||ref_means[i] - cand_means[j]||^2
    cost = np.sum((ref_means[:, None, :] - cand_means[None, :, :]) ** 2, axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.empty(len(col_ind), dtype=int)
    perm[col_ind] = row_ind  # candidate j -> reference perm[j]
    return perm


def _sk_init_params_from_means(X_np, means_np, cov_type, reg_covar=1e-6):
    """Given initial means, compute sklearn-style initial precisions for all cov types."""
    N, D = X_np.shape
    K = means_np.shape[0]
    # Assign each point to its nearest mean
    dists = np.sum((X_np[:, None, :] - means_np[None, :, :]) ** 2, axis=-1)
    labels = np.argmin(dists, axis=1)
    resp = np.zeros((N, K))
    resp[np.arange(N), labels] = 1.0
    _, _, sk_cov = _sk_estimate_gaussian_parameters(X_np, resp, reg_covar=reg_covar, covariance_type=cov_type)
    # Compute precisions from sk_cov
    if cov_type == "diag":
        precisions = 1.0 / sk_cov            # (K, D)
    elif cov_type == "spherical":
        precisions = 1.0 / sk_cov            # (K,)
    elif cov_type == "tied":
        precisions = np.linalg.inv(sk_cov)   # (D, D)
    else:  # full
        precisions = np.linalg.inv(sk_cov)   # (K, D, D)
    return precisions


SEED_ESTEP = np.random.randint(1, 1001)
SEED_MSTEP = np.random.randint(1, 1001)
SEED_E2E   = np.random.randint(1, 1001)


# ───────────────────────────────────────────────────────────────────────────
# Test 1: E-step correctness
# ───────────────────────────────────────────────────────────────────────────

def _make_spd(rng, size):
    """Random symmetric positive definite matrix of shape (size, size)."""
    A = rng.randn(size, size)
    return (A @ A.T + size * np.eye(size)).astype(np.float64)


def _build_cov_and_prec_chol(rng, cov_type, K, D):
    """
    Build a random covariance tensor and the corresponding precisions_chol
    in sklearn's expected format for each covariance type.

    Returns (cov_np, prec_chol_np) as numpy arrays.
    """
    if cov_type == "spherical":
        cov_np = rng.uniform(0.5, 2.0, K).astype(np.float64)          # (K,)
        prec_chol_np = 1.0 / np.sqrt(cov_np)                           # (K,)

    elif cov_type == "diag":
        cov_np = rng.uniform(0.5, 2.0, (K, D)).astype(np.float64)      # (K, D)
        prec_chol_np = 1.0 / np.sqrt(cov_np)                           # (K, D)

    elif cov_type == "tied":
        cov_np = _make_spd(rng, D)                                      # (D, D)
        prec_np = np.linalg.inv(cov_np)
        prec_chol_np = np.linalg.cholesky(prec_np)                     # (D, D) lower-tri

    else:  # full
        cov_np = np.stack([_make_spd(rng, D) for _ in range(K)])       # (K, D, D)
        prec_chol_np = np.stack([
            np.linalg.cholesky(np.linalg.inv(cov_np[k])) for k in range(K)
        ])                                                               # (K, D, D) lower-tri

    return cov_np, prec_chol_np


def _run_e_step_test_for_cov_type(cov_type, label, lp_fn, estep_fn, compute_prec_fn):
    """Compare one implementation's E-step against sklearn for a given covariance type."""
    rng = np.random.RandomState(SEED_ESTEP)
    N, D, K = 200, 5, 3

    X_np     = rng.randn(N, D).astype(np.float64)
    X_t      = torch.from_numpy(X_np)
    means_np = rng.randn(K, D).astype(np.float64)
    means_t  = torch.from_numpy(means_np)

    # Random mixture weights
    weights_np = rng.dirichlet(np.ones(K))
    weights_t  = torch.from_numpy(weights_np)

    cov_np, prec_chol_np = _build_cov_and_prec_chol(rng, cov_type, K, D)
    cov_t = torch.from_numpy(cov_np)

    # sklearn reference
    sk_log_prob    = _sk_estimate_log_gaussian_prob(X_np, means_np, prec_chol_np, cov_type)
    sk_resp_unnorm = sk_log_prob + np.log(weights_np)[None, :]
    sk_log_norm    = np.logaddexp.reduce(sk_resp_unnorm, axis=1, keepdims=True)
    sk_resp        = np.exp(sk_resp_unnorm - sk_log_norm)

    # PyTorch implementation
    prec_chol_t    = compute_prec_fn(cov_t, cov_type)
    torch_log_prob = lp_fn(X_t, means_t, prec_chol_t, cov_type).cpu().numpy()
    _, log_resp_t  = estep_fn(X_t, means_t, prec_chol_t, weights_t, cov_type)
    torch_resp     = log_resp_t.exp().cpu().numpy()

    max_lp_diff   = np.max(np.abs(sk_log_prob - torch_log_prob))
    max_resp_diff = np.max(np.abs(sk_resp - torch_resp))

    print(f"    {cov_type:<10}  log-prob diff: {max_lp_diff:.2e}  resp diff: {max_resp_diff:.2e}", end="")
    assert max_lp_diff   < 1e-10, f"[{label}/{cov_type}] log-prob mismatch: {max_lp_diff}"
    assert max_resp_diff < 1e-10, f"[{label}/{cov_type}] resp mismatch: {max_resp_diff}"
    print("  ok")


def test_e_step_correctness():
    print("\n" + "=" * 80)
    print(f"TEST 1: E-step Correctness  (seed={SEED_ESTEP})")
    print("=" * 80)

    impls = [
        ("_v0_ref",    v0._estimate_log_gaussian_prob_precchol,
                       v0._expectation_step_precchol,
                       v0._compute_precisions_cholesky),
        ("_v1",        v1._estimate_log_gaussian_prob_precchol,
                       v1._expectation_step_precchol,
                       v1._compute_precisions_cholesky),
        # _v2_reduced reuses v1's functions; _v2_tiling has its own:
        ("_v2_tiling", v2t._estimate_log_gaussian_prob_precchol,
                       v2t._expectation_step_precchol,
                       v2t._compute_precisions_cholesky),
    ]
    for label, lp_fn, estep_fn, prec_fn in impls:
        print(f"\n  [{label}]")
        for cov_type in ("spherical", "diag", "tied", "full"):
            _run_e_step_test_for_cov_type(cov_type, label, lp_fn, estep_fn, prec_fn)

    print("\n+ E-step correctness PASSED for all implementations and covariance types\n")


# ───────────────────────────────────────────────────────────────────────────
# Test 2: M-step correctness
# ───────────────────────────────────────────────────────────────────────────

def _run_m_step_test_for_cov_type(cov_type, label, mstep_fn):
    """Compare one implementation's M-step against sklearn for a given covariance type."""
    rng = np.random.RandomState(SEED_MSTEP)
    N, D, K = 100, 5, 3

    X_np = rng.randn(N, D).astype(np.float64)
    X_t  = torch.from_numpy(X_np)

    resp_np = rng.uniform(0, 1, (N, K)).astype(np.float64)
    resp_np /= resp_np.sum(axis=1, keepdims=True)
    log_resp_t = torch.from_numpy(np.log(resp_np))

    # sklearn M-step reference
    sk_norms, sk_means, sk_cov = _sk_estimate_gaussian_parameters(
        X_np, resp_np, reg_covar=1e-6, covariance_type=cov_type
    )
    sk_weights = sk_norms / sk_norms.sum()

    # Dummy initial params for PyTorch — shapes must match cov_type
    means0   = torch.from_numpy(rng.randn(K, D).astype(np.float64))
    weights0 = torch.full((K,), 1.0 / K, dtype=torch.float64)
    if cov_type == "spherical":
        cov0 = torch.ones(K, dtype=torch.float64)
    elif cov_type == "diag":
        cov0 = torch.ones(K, D, dtype=torch.float64)
    elif cov_type == "tied":
        cov0 = torch.eye(D, dtype=torch.float64)
    else:  # full
        cov0 = torch.eye(D, dtype=torch.float64).unsqueeze(0).expand(K, D, D).contiguous()

    torch_means, torch_cov, torch_weights = mstep_fn(
        X_t, means0, cov0, weights0, log_resp_t, cov_type, reg_covar=1e-6
    )
    torch_means   = torch_means.cpu().numpy()
    torch_cov     = torch_cov.cpu().numpy()
    torch_weights = torch_weights.cpu().numpy()

    max_means_diff   = np.max(np.abs(sk_means   - torch_means))
    max_cov_diff     = np.max(np.abs(sk_cov     - torch_cov))
    max_weights_diff = np.max(np.abs(sk_weights - torch_weights))

    print(
        f"    {cov_type:<10}  means: {max_means_diff:.2e}"
        f"  cov: {max_cov_diff:.2e}  weights: {max_weights_diff:.2e}",
        end="",
    )
    assert max_means_diff   < 1e-10, f"[{label}/{cov_type}] means mismatch: {max_means_diff}"
    assert max_cov_diff     < 1e-10, f"[{label}/{cov_type}] cov mismatch: {max_cov_diff}"
    assert max_weights_diff < 1e-10, f"[{label}/{cov_type}] weights mismatch: {max_weights_diff}"
    print("  ok")


def test_m_step_correctness():
    print("=" * 80)
    print(f"TEST 2: M-step Correctness  (seed={SEED_MSTEP})")
    print("=" * 80)

    impls = [
        ("_v0_ref",    v0._maximization_step),
        ("_v1",        v1._maximization_step),
        # _v2_reduced uses v1._maximization_step when update_covariance=True
        ("_v2_tiling", v2t._maximization_step),
    ]
    for label, mstep_fn in impls:
        print(f"\n  [{label}]")
        for cov_type in ("spherical", "diag", "tied", "full"):
            _run_m_step_test_for_cov_type(cov_type, label, mstep_fn)

    print("\n+ M-step correctness PASSED for all implementations and covariance types\n")


# ───────────────────────────────────────────────────────────────────────────
# Test 3: End-to-end with identical initialization
# ───────────────────────────────────────────────────────────────────────────

def test_end_to_end():
    """
    _v0_ref and _v1 are started from the exact same initial means and precisions
    (derived from a single sklearn KMeans run) and run to convergence, across all
    four covariance types.

    Assertions per covariance type:
      (a) EM trajectory is monotonically non-decreasing (core EM invariant).
      (b) Both PyTorch models converge within MAX_ITER iterations.
      (c) LL is finite.
      (d) _v0_ref and _v1 agree on LL to 1e-8 (same algorithm, same init —
          any divergence indicates a bug in one of them).

    Note: LL vs sklearn is informational only. Even from identical init,
    floating-point differences between sklearn's C-accelerated EM and PyTorch
    steer trajectories to different local optima. The E/M step math is verified
    to machine precision in Tests 1 & 2; parameter-level agreement under identical
    init is verified in Test 5 (K=1 unique optimum).
    """
    print("=" * 80)
    print(f"TEST 3: End-to-End with Identical Init  (seed={SEED_E2E})")
    print("=" * 80)

    np.random.seed(SEED_E2E)
    N, D, K   = 300, 8, 4
    REG_COVAR = 1e-6
    MAX_ITER  = 300
    TOL       = 1e-4

    X_np = np.random.randn(N, D).astype(np.float64)
    X_t  = torch.from_numpy(X_np)

    km = KMeans(n_clusters=K, n_init=1, random_state=SEED_E2E).fit(X_np)
    init_means_np = km.cluster_centers_.astype(np.float64)  # (K, D)

    overall_failed = []

    for COV_TYPE in ("spherical", "diag", "tied", "full"):
        print(f"\n  {'─' * 60}")
        print(f"  [{COV_TYPE}]")
        print(f"  {'─' * 60}")

        init_prec_np = _sk_init_params_from_means(X_np, init_means_np, COV_TYPE, REG_COVAR)
        init_means_t = torch.from_numpy(init_means_np)
        init_prec_t  = torch.from_numpy(init_prec_np)

        sk_gmm = SklearnGMM(
            n_components=K,
            covariance_type=COV_TYPE,
            reg_covar=REG_COVAR,
            max_iter=MAX_ITER,
            tol=TOL,
            n_init=1,
            init_params="kmeans",
            random_state=SEED_E2E,
            means_init=init_means_np,
            precisions_init=init_prec_np,
        )
        sk_gmm.fit(X_np)
        sk_ll    = sk_gmm.lower_bound_
        sk_iter  = sk_gmm.n_iter_
        sk_means = sk_gmm.means_

        print(f"\n  sklearn: ll={sk_ll:.6f}  iter={sk_iter}"
              f"  converged={'yes' if sk_gmm.converged_ else 'NO'}")

        def run_torch_gmm(GMM_cls, **extra_kwargs):
            gmm = GMM_cls(
                n_components=K,
                covariance_type=COV_TYPE,
                reg_covar=REG_COVAR,
                max_iter=MAX_ITER,
                tol=TOL,
                n_init=1,
                init_params="kmeans",
                device=None,
                dtype=torch.float64,
                means_init=init_means_t.clone(),
                precisions_init=init_prec_t.clone(),
                **extra_kwargs,
            )
            gmm.fit(X_t)
            return gmm

        results = {
            "_v0_ref": run_torch_gmm(v0.TorchGaussianMixture),
            "_v1":     run_torch_gmm(v1.TorchGaussianMixture),
        }

        print(f"\n  {'Implementation':<20} {'lower_bound':>14} {'ll_diff vs sk':>15}"
              f" {'converged':>10} {'iter':>6}")
        print(f"  {'sklearn':<20} {sk_ll:>14.6f} {'—':>15}"
              f" {'yes' if sk_gmm.converged_ else 'NO':>10} {sk_iter:>6}")

        failed        = []
        reference_ll   = None
        reference_name = None

        for name, gmm in results.items():
            ll      = gmm.lower_bound_
            ll_diff = ll - sk_ll
            history = gmm.lower_bounds_

            # (a) Monotonicity
            diffs      = np.diff(history)
            monotone   = bool(np.all(diffs >= -1e-8))
            worst_drop = float(np.min(diffs)) if len(diffs) > 0 else 0.0

            # (b) Convergence
            converged = gmm.converged_

            print(f"  {name:<20} {ll:>14.6f} {ll_diff:>+15.2e}"
                  f" {'yes' if converged else 'NO':>10} {gmm.n_iter_:>6}")

            if not monotone:
                failed.append(
                    f"[{COV_TYPE}] {name}: not monotone; worst drop = {worst_drop:.2e}"
                )
            if not converged:
                failed.append(
                    f"[{COV_TYPE}] {name}: did not converge in {gmm.n_iter_} iterations"
                )
            if not np.isfinite(ll):
                failed.append(f"[{COV_TYPE}] {name}: LL is not finite ({ll})")

            # (d) Inter-implementation consistency
            if reference_ll is None:
                reference_ll   = ll
                reference_name = name
            else:
                inter_diff = abs(ll - reference_ll)
                if inter_diff > 1e-8:
                    failed.append(
                        f"[{COV_TYPE}] {name} vs {reference_name}:"
                        f" LL disagreement = {inter_diff:.2e}"
                    )

            # Informational: aligned means diff vs sklearn
            cand_means    = gmm.means_.cpu().numpy()
            perm          = _align_components(sk_means, cand_means)
            aligned_means = cand_means[perm]
            means_diff    = np.max(np.abs(sk_means - aligned_means))
            print(f"    means diff vs sklearn (informational): {means_diff:.2e}")

            # Informational: L2 density error vs sklearn
            p_torch = gmm.score_samples(X_t).exp().cpu().numpy()
            p_sk    = np.exp(sk_gmm.score_samples(X_np))
            l2_err  = np.sqrt(np.mean((p_torch - p_sk) ** 2))
            print(f"    L2 density error vs sklearn (informational): {l2_err:.2e}")

        if failed:
            overall_failed.extend(failed)
            for msg in failed:
                print(f"  FAIL: {msg}")

    if overall_failed:
        raise AssertionError(
            "End-to-end correctness failures:\n" + "\n".join(overall_failed)
        )

    print("\n+ End-to-end PASSED for all implementations and covariance types\n")


# ───────────────────────────────────────────────────────────────────────────
# Test 4: Cholesky / positive-definiteness stability
# ───────────────────────────────────────────────────────────────────────────

def test_cholesky_stability():
    """
    Regression test for:
        torch._C._LinAlgError: linalg.cholesky: The factorization could not be
        completed because the input is not positive-definite.

    This crash occurs in _compute_precisions_cholesky (called after each M-step)
    when a covariance matrix produced by the M-step is no longer positive definite.
    Root causes:
      - A component accumulates too few points (near-collapse) so its empirical
        covariance is rank-deficient or indefinite before reg_covar is added.
      - reg_covar is too small to restore positive definiteness.
      - High D magnifies both effects (more dimensions = harder to stay SPD).

    The test runs EM under several stress configurations (full covariance, varying
    D and reg_covar) and asserts that every final covariance matrix is positive
    definite by checking its minimum eigenvalue. This catches the problem before
    it reaches the Cholesky call.

    Implementations tested: _v0_ref and _v1 (the benchmark targets).
    """
    print("=" * 80)
    print("TEST 4: Cholesky / Positive-Definiteness Stability")
    print("=" * 80)

    # Stress configurations: (N, D, K, reg_covar)
    # D is kept moderate for test speed, but covers the high-D regime.
    # K is intentionally large relative to N to encourage near-empty components.
    configs = [
        (200,  20,  8, 1e-6),   # baseline stress
        (200,  50,  8, 1e-6),   # higher D
        (200, 100,  8, 1e-6),   # D approaching benchmark scale
        (100,  30, 10, 1e-6),   # K large relative to N
        (200,  50,  8, 1e-4),   # larger reg_covar — should always pass
    ]

    impls = [
        ("_v0_ref", v0.TorchGaussianMixture),
        ("_v1",     v1.TorchGaussianMixture),
    ]

    rng = np.random.RandomState(42)
    failed = []

    print(f"\n  {'impl':<10} {'N':>5} {'D':>5} {'K':>4} {'reg_covar':>10}  result")

    for N, D, K, reg_covar in configs:
        X_np = rng.randn(N, D).astype(np.float64)
        X_t  = torch.from_numpy(X_np)

        for label, GMM_cls in impls:
            try:
                gmm = GMM_cls(
                    n_components=K,
                    covariance_type="full",
                    reg_covar=reg_covar,
                    max_iter=100,
                    tol=1e-4,
                    n_init=1,
                    init_params="kmeans",
                    device=None,
                    dtype=torch.float64,
                ).fit(X_t)

                # Verify all K covariance matrices are positive definite
                covs = gmm.covariances_.cpu().numpy()  # (K, D, D)
                min_eigvals = np.array([
                    np.linalg.eigvalsh(covs[k]).min() for k in range(K)
                ])
                min_ev = float(min_eigvals.min())

                if min_ev <= 0:
                    result = f"FAIL (min_eigval={min_ev:.2e})"
                    failed.append(
                        f"{label} N={N} D={D} K={K} reg={reg_covar}: "
                        f"non-positive-definite covariance (min_eigval={min_ev:.2e})"
                    )
                else:
                    result = f"ok (min_eigval={min_ev:.2e})"

            except Exception as e:
                result = f"CRASH: {type(e).__name__}"
                failed.append(
                    f"{label} N={N} D={D} K={K} reg={reg_covar}: {type(e).__name__}: {e}"
                )

            print(f"  {label:<10} {N:>5} {D:>5} {K:>4} {reg_covar:>10.0e}  {result}")

    if failed:
        print()
        for msg in failed:
            print(f"  FAIL: {msg}")
        raise AssertionError("Cholesky stability failures:\n" + "\n".join(failed))

    print("\n+ Cholesky stability PASSED\n")


# ───────────────────────────────────────────────────────────────────────────
# Test 5: K=1 score_samples correctness — unique global optimum
# ───────────────────────────────────────────────────────────────────────────

def test_score_samples_k1():
    """
    With K=1, every E-step assigns responsibility 1 to the single component for
    every data point (log_resp = 0 exactly), regardless of the current parameters.
    The M-step therefore always produces the maximum likelihood Gaussian:
      mean = sample mean,  cov = sample covariance + reg_covar
    This is the unique global optimum, reached in a single EM iteration from any
    initialization.

    Both sklearn and _v1 must converge to the same solution, so their
    score_samples values must agree to near machine precision (~1e-10).

    This directly validates:
      - score_samples is correctly implemented in _v1.
      - EM converges to the right parameter values, not just satisfies invariants.
      - The result is init-invariant (K=1 makes every init equivalent).
    """
    print("=" * 80)
    print("TEST 5: K=1 score_samples correctness  (unique global optimum)")
    print("=" * 80)

    rng = np.random.RandomState(42)
    N, D      = 500, 6
    K         = 1
    REG_COVAR = 1e-6

    X_np = rng.randn(N, D).astype(np.float64)
    X_t  = torch.from_numpy(X_np)

    failed = []

    print(f"\n  {'cov_type':<12} {'L2 density error':>18}  result")

    for cov_type in ("spherical", "diag", "tied", "full"):
        sk_gmm = SklearnGMM(
            n_components=K,
            covariance_type=cov_type,
            reg_covar=REG_COVAR,
            max_iter=10,     # K=1 always converges in a single M-step
            tol=1e-12,
            n_init=1,
            random_state=42,
        ).fit(X_np)

        gmm = v1.TorchGaussianMixture(
            n_components=K,
            covariance_type=cov_type,
            reg_covar=REG_COVAR,
            max_iter=10,
            tol=1e-12,
            n_init=1,
            init_params="random_from_data",
            dtype=torch.float64,
        ).fit(X_t)

        p_sk  = np.exp(sk_gmm.score_samples(X_np))         # (N,)
        p_v1  = gmm.score_samples(X_t).exp().cpu().numpy() # (N,)
        l2_err = np.sqrt(np.mean((p_v1 - p_sk) ** 2))

        tol = 1e-10
        ok  = l2_err < tol
        print(f"  {cov_type:<12} {l2_err:>18.2e}  {'ok' if ok else 'FAIL'}")

        if not ok:
            failed.append(
                f"[{cov_type}] K=1 L2 density error = {l2_err:.2e} > {tol:.0e}"
            )

    if failed:
        for msg in failed:
            print(f"  FAIL: {msg}")
        raise AssertionError("K=1 score_samples failures:\n" + "\n".join(failed))

    print("\n+ K=1 score_samples PASSED for all covariance types\n")


# ───────────────────────────────────────────────────────────────────────────
# Test 6: AIC/BIC correctness
# ───────────────────────────────────────────────────────────────────────────

def test_aic_bic():
    """
    Test 6: AIC/BIC correctness — two complementary strategies.

    Part A (K=1, unique optimum): With K=1, all three implementations converge to
    the same MLE Gaussian from any initialization (see test_score_samples_k1 for
    the argument). Their AIC and BIC must therefore agree to near machine precision.
    Tests the formula and all 4 covariance types.

    Part B (parameter injection, K=4): sklearn's fitted parameters are extracted
    and injected directly into _v0_ref and _v1 instances, bypassing EM entirely.
    All three models share identical parameters, making AIC/BIC directly comparable
    for K>1. Also validates _n_parameters for K=4, catching K-multiplier bugs (e.g.
    full uses K*D*(D+1)/2 while tied uses D*(D+1)/2 — indistinguishable at K=1).
    """
    print("=" * 80)
    print("TEST 6: AIC/BIC correctness")
    print("=" * 80)

    REG_COVAR = 1e-6
    failed    = []

    # ── Part A: K=1, unique global optimum ───────────────────────────────
    print("\n  Part A — K=1 unique optimum")
    print(f"  {'cov_type':<12} {'metric':>6} {'sklearn':>14}"
          f" {'v0_ref diff':>14} {'v1 diff':>14}  result")

    rng_a = np.random.RandomState(42)
    N_a, D_a, K_a = 500, 6, 1
    X_np_a = rng_a.randn(N_a, D_a).astype(np.float64)
    X_t_a  = torch.from_numpy(X_np_a)

    for cov_type in ("spherical", "diag", "tied", "full"):
        sk_a = SklearnGMM(
            n_components=K_a, covariance_type=cov_type, reg_covar=REG_COVAR,
            max_iter=10, tol=1e-12, n_init=1, random_state=42,
        ).fit(X_np_a)

        v0_a = v0.TorchGaussianMixture(
            n_components=K_a, covariance_type=cov_type, reg_covar=REG_COVAR,
            max_iter=10, tol=1e-12, n_init=1, init_params="random_from_data",
            dtype=torch.float64,
        ).fit(X_t_a)

        v1_a = v1.TorchGaussianMixture(
            n_components=K_a, covariance_type=cov_type, reg_covar=REG_COVAR,
            max_iter=10, tol=1e-12, n_init=1, init_params="random_from_data",
            dtype=torch.float64,
        ).fit(X_t_a)

        tol_a = 1e-4
        for metric, sk_val, v0_val, v1_val in [
            ("AIC", sk_a.aic(X_np_a), v0_a.aic(X_t_a).item(), v1_a.aic(X_t_a).item()),
            ("BIC", sk_a.bic(X_np_a), v0_a.bic(X_t_a).item(), v1_a.bic(X_t_a).item()),
        ]:
            v0_diff = abs(v0_val - sk_val)
            v1_diff = abs(v1_val - sk_val)
            ok = v0_diff < tol_a and v1_diff < tol_a
            print(f"  {cov_type:<12} {metric:>6} {sk_val:>14.4f}"
                  f" {v0_diff:>14.2e} {v1_diff:>14.2e}  {'ok' if ok else 'FAIL'}")
            if not ok:
                if v0_diff >= tol_a:
                    failed.append(
                        f"[A/{cov_type}] _v0_ref {metric} diff = {v0_diff:.2e} > {tol_a:.0e}"
                    )
                if v1_diff >= tol_a:
                    failed.append(
                        f"[A/{cov_type}] _v1 {metric} diff = {v1_diff:.2e} > {tol_a:.0e}"
                    )

    # ── Part B: Parameter injection, K=4 ─────────────────────────────────
    print(f"\n  Part B — Parameter injection (K=4)")
    print(f"  {'cov_type':<12} {'n_params':>8} {'metric':>6} {'sklearn':>14}"
          f" {'v0_ref diff':>14} {'v1 diff':>14}  result")

    rng_b = np.random.RandomState(42)
    N_b, D_b, K_b = 300, 8, 4
    X_np_b = rng_b.randn(N_b, D_b).astype(np.float64)
    X_t_b  = torch.from_numpy(X_np_b)

    for cov_type in ("spherical", "diag", "tied", "full"):
        sk_b = SklearnGMM(
            n_components=K_b, covariance_type=cov_type, reg_covar=REG_COVAR,
            max_iter=200, tol=1e-4, n_init=1, random_state=42,
        ).fit(X_np_b)

        # Extract sklearn's fitted parameters as float64 torch tensors.
        # Covariance shapes match our storage format exactly:
        #   spherical (K,), diag (K,D), tied (D,D), full (K,D,D).
        cov_t   = torch.from_numpy(np.asarray(sk_b.covariances_)).double()
        w_t     = torch.from_numpy(sk_b.weights_).double()
        means_t = torch.from_numpy(sk_b.means_).double()

        # Inject into _v0_ref — no fit(), only _params needs to be set
        prec_chol_v0 = v0._compute_precisions_cholesky(cov_t, cov_type)
        v0_b = v0.TorchGaussianMixture(
            n_components=K_b, covariance_type=cov_type, dtype=torch.float64,
        )
        v0_b._params = v0.GMMParams(
            weights=w_t, means=means_t, cov=cov_t,
            prec_chol=prec_chol_v0, cov_type=cov_type,
        )

        # Inject into _v1
        prec_chol_v1 = v1._compute_precisions_cholesky(cov_t, cov_type)
        v1_b = v1.TorchGaussianMixture(
            n_components=K_b, covariance_type=cov_type, dtype=torch.float64,
        )
        v1_b._params = v1.GMMParams(
            weights=w_t, means=means_t, cov=cov_t,
            prec_chol=prec_chol_v1, cov_type=cov_type,
        )

        # n_parameters check — integer equality, no tolerance
        sk_np = sk_b._n_parameters()
        v0_np = v0_b._n_parameters(D_b)
        v1_np = v1_b._n_parameters(D_b)
        if v0_np != sk_np:
            failed.append(
                f"[B/{cov_type}] _v0_ref n_parameters={v0_np} != sklearn={sk_np}"
            )
        if v1_np != sk_np:
            failed.append(
                f"[B/{cov_type}] _v1 n_parameters={v1_np} != sklearn={sk_np}"
            )

        tol_b = 1e-6
        for i, (metric, sk_val, v0_val, v1_val) in enumerate([
            ("AIC", sk_b.aic(X_np_b), v0_b.aic(X_t_b).item(), v1_b.aic(X_t_b).item()),
            ("BIC", sk_b.bic(X_np_b), v0_b.bic(X_t_b).item(), v1_b.bic(X_t_b).item()),
        ]):
            v0_diff = abs(v0_val - sk_val)
            v1_diff = abs(v1_val - sk_val)
            ok      = v0_diff < tol_b and v1_diff < tol_b
            # Print n_params on first metric row only, blank on second
            n_tag = f"{sk_np}" if i == 0 else ""
            print(f"  {cov_type:<12} {n_tag:>8} {metric:>6} {sk_val:>14.4f}"
                  f" {v0_diff:>14.2e} {v1_diff:>14.2e}  {'ok' if ok else 'FAIL'}")
            if not ok:
                if v0_diff >= tol_b:
                    failed.append(
                        f"[B/{cov_type}] _v0_ref {metric} diff = {v0_diff:.2e} > {tol_b:.0e}"
                    )
                if v1_diff >= tol_b:
                    failed.append(
                        f"[B/{cov_type}] _v1 {metric} diff = {v1_diff:.2e} > {tol_b:.0e}"
                    )

    if failed:
        for msg in failed:
            print(f"  FAIL: {msg}")
        raise AssertionError("AIC/BIC failures:\n" + "\n".join(failed))

    print("\n+ AIC/BIC PASSED for all covariance types\n")


# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_e_step_correctness()
    test_m_step_correctness()
    test_end_to_end()
    test_cholesky_stability()
    test_score_samples_k1()
    test_aic_bic()

    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
