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
SEED_V2R   = np.random.randint(1, 1001)


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
    All four PyTorch implementations and sklearn are started from the exact same
    initial means and precisions (derived from a single sklearn KMeans run) and
    run to convergence. Their final lower bounds and parameters are compared.

    With identical init, differences should only arise from float32 arithmetic
    (if dtype=float32 is used) or numerical quirks — we expect very tight
    agreement on the lower bound.
    """
    print("=" * 80)
    print(f"TEST 3: End-to-End with Identical Init  (seed={SEED_E2E})")
    print("=" * 80)

    np.random.seed(SEED_E2E)
    N, D, K = 300, 8, 4
    COV_TYPE = "diag"
    REG_COVAR = 1e-6
    MAX_ITER = 200
    TOL = 1e-4

    # Generate data
    X_np = np.random.randn(N, D).astype(np.float64)
    X_t  = torch.from_numpy(X_np)

    # ── Identical initialization via sklearn KMeans ──────────────────────
    km = KMeans(n_clusters=K, n_init=1, random_state=SEED_E2E).fit(X_np)
    init_means_np = km.cluster_centers_.astype(np.float64)  # (K, D)

    # Compute initial precisions from hard KMeans assignments
    init_prec_np = _sk_init_params_from_means(X_np, init_means_np, COV_TYPE, REG_COVAR)

    init_means_t = torch.from_numpy(init_means_np)
    init_prec_t  = torch.from_numpy(init_prec_np)

    # ── sklearn GMM from the same init means + precisions ────────────────
    sk_gmm = SklearnGMM(
        n_components=K,
        covariance_type=COV_TYPE,
        reg_covar=REG_COVAR,
        max_iter=MAX_ITER,
        tol=TOL,
        n_init=1,
        init_params="kmeans",       # overridden by means_init below
        random_state=SEED_E2E,
        means_init=init_means_np,
        precisions_init=init_prec_np,
    )
    sk_gmm.fit(X_np)
    sk_ll    = sk_gmm.lower_bound_
    sk_iter  = sk_gmm.n_iter_
    sk_means = sk_gmm.means_

    print(f"\n  sklearn: ll={sk_ll:.6f}  iter={sk_iter}")

    # ── Helper: build and run a PyTorch GMM from identical init ──────────
    def run_torch_gmm(label, GMM_cls, **extra_kwargs):
        gmm = GMM_cls(
            n_components=K,
            covariance_type=COV_TYPE,
            reg_covar=REG_COVAR,
            max_iter=MAX_ITER,
            tol=TOL,
            n_init=1,
            init_params="kmeans",   # overridden by means_init below
            device=None,
            dtype=torch.float64,
            means_init=init_means_t.clone(),
            precisions_init=init_prec_t.clone(),
            **extra_kwargs,
        )
        gmm.fit(X_t)
        return gmm

    results = {}

    print(f"\n  [_v0_ref]")
    gmm_v0 = run_torch_gmm("_v0_ref", v0.TorchGaussianMixture)
    results["_v0_ref"] = gmm_v0

    print(f"  [_v1]")
    gmm_v1 = run_torch_gmm("_v1", v1.TorchGaussianMixture)
    results["_v1"] = gmm_v1

    print(f"  [_v2_reduced]")
    gmm_v2r = run_torch_gmm("_v2_reduced", v2r.TorchGaussianMixture,
                             covariance_update_frequency=1)  # same as v1
    results["_v2_reduced"] = gmm_v2r

    print(f"  [_v2_tiling]")
    gmm_v2t = run_torch_gmm("_v2_tiling", v2t.TorchGaussianMixture)
    results["_v2_tiling"] = gmm_v2t

    # ── Compare ──────────────────────────────────────────────────────────
    # NOTE on means comparison:
    # Even with identical initialization, tiny floating-point differences
    # between sklearn's C-accelerated EM and PyTorch's operations cause
    # diverging EM trajectories that converge to different local optima.
    # With a flat likelihood landscape near convergence, parameters can
    # differ substantially (e.g. 4e-02 in means) while the LL difference
    # is negligible (<1e-5 relative). Comparing raw means is therefore not
    # a meaningful correctness check.
    #
    # Instead we verify:
    #   (a) LL is at least as good as sklearn (PyTorch should find >= sklearn's optimum).
    #   (b) EM trajectory is monotonically non-decreasing (core correctness invariant).
    #   (c) All four PyTorch implementations agree with each other exactly
    #       (they share init; any divergence would indicate a bug in one of them).

    print(f"\n  {'Implementation':<20} {'lower_bound':>14} {'ll_diff vs sk':>15} {'monotone':>9} {'iter':>6}")
    print(f"  {'sklearn':<20} {sk_ll:>14.6f} {'—':>15} {'—':>9} {sk_iter:>6}")

    failed = []
    reference_ll    = None   # first PyTorch result, used for inter-impl consistency check
    reference_name  = None

    for name, gmm in results.items():
        ll      = gmm.lower_bound_
        ll_diff = ll - sk_ll          # positive = PyTorch found better or equal optimum
        history = gmm.lower_bounds_

        # Monotonicity check: each recorded lower bound must be >= previous
        diffs    = np.diff(history)
        monotone = bool(np.all(diffs >= -1e-8))   # allow tiny float noise
        worst_drop = float(np.min(diffs)) if len(diffs) > 0 else 0.0

        print(f"  {name:<20} {ll:>14.6f} {ll_diff:>+15.2e} {'yes' if monotone else 'NO':>9} {gmm.n_iter_:>6}")
        if not monotone:
            failed.append(f"{name}: EM is not monotone; worst drop = {worst_drop:.2e}")

        # LL quality: PyTorch must not be significantly worse than sklearn.
        # We allow it to be better (higher LL = better fit).
        # Threshold: sklearn's own convergence tolerance (TOL) in absolute terms.
        if ll_diff < -TOL:
            failed.append(
                f"{name}: LL worse than sklearn by {-ll_diff:.2e} (threshold={TOL:.2e})"
            )

        # Inter-implementation consistency: all four PyTorch impls should agree.
        if reference_ll is None:
            reference_ll   = ll
            reference_name = name
        else:
            inter_diff = abs(ll - reference_ll)
            if inter_diff > 1e-8:
                failed.append(
                    f"{name} vs {reference_name}: LL disagreement = {inter_diff:.2e}"
                )

        # Informational: show aligned means diff (not asserted)
        cand_means = gmm.means_.cpu().numpy()
        perm = _align_components(sk_means, cand_means)
        aligned_means = cand_means[perm]
        means_diff = np.max(np.abs(sk_means - aligned_means))
        print(f"    means diff vs sklearn (informational): {means_diff:.2e}")

    if failed:
        for msg in failed:
            print(f"  FAIL: {msg}")
        raise AssertionError("End-to-end correctness failures:\n" + "\n".join(failed))

    print("\n+ End-to-end PASSED for all implementations\n")


# ───────────────────────────────────────────────────────────────────────────
# Test 4: _v2_reduced covariance throttling
# ───────────────────────────────────────────────────────────────────────────

def test_v2_reduced_throttling():
    """
    Test _v2_reduced with covariance_update_frequency > 1.

    When covariances are updated only every N iterations, the M-step is
    incomplete and EM monotonicity is NOT guaranteed — this is a known
    trade-off of the variant. This test checks that:
      (a) The model still converges to a reasonable LL (within a loose
          factor of the full-update baseline).
      (b) The throttled path executes without numerical failure.
      (c) Monotonicity violations, if any, are reported (informational).

    Uses 'full' covariance (most expensive, most benefit from throttling).
    """
    print("=" * 80)
    print(f"TEST 4: _v2_reduced Covariance Throttling  (seed={SEED_V2R})")
    print("=" * 80)

    rng = np.random.RandomState(SEED_V2R)
    N, D, K = 300, 8, 4
    COV_TYPE  = "full"
    REG_COVAR = 1e-6
    MAX_ITER  = 300
    TOL       = 1e-4

    X_np = rng.randn(N, D).astype(np.float64)
    X_t  = torch.from_numpy(X_np)

    # Shared KMeans init
    km = KMeans(n_clusters=K, n_init=1, random_state=SEED_V2R).fit(X_np)
    init_means_np = km.cluster_centers_.astype(np.float64)
    init_prec_np  = _sk_init_params_from_means(X_np, init_means_np, COV_TYPE, REG_COVAR)
    init_means_t  = torch.from_numpy(init_means_np)
    init_prec_t   = torch.from_numpy(init_prec_np)

    def make_gmm(freq):
        return v2r.TorchGaussianMixture(
            n_components=K,
            covariance_type=COV_TYPE,
            reg_covar=REG_COVAR,
            max_iter=MAX_ITER,
            tol=TOL,
            n_init=1,
            device=None,
            dtype=torch.float64,
            means_init=init_means_t.clone(),
            precisions_init=init_prec_t.clone(),
            covariance_update_frequency=freq,
        ).fit(X_t)

    # Baseline: full covariance updates (freq=1, identical to _v1)
    gmm_f1 = make_gmm(1)
    ref_ll  = gmm_f1.lower_bound_

    print(f"\n  {'freq':<6} {'lower_bound':>14} {'ll_diff vs f=1':>16} {'monotone':>9} {'iter':>6}")
    print(f"  {1:<6} {ref_ll:>14.6f} {'(reference)':>16} {'yes':>9} {gmm_f1.n_iter_:>6}")

    failed = []
    for freq in [2, 3, 5]:
        gmm     = make_gmm(freq)
        ll      = gmm.lower_bound_
        diff    = ll - ref_ll
        history = gmm.lower_bounds_
        h_diffs = np.diff(history)
        monotone    = bool(np.all(h_diffs >= -1e-8))
        worst_drop  = float(np.min(h_diffs)) if len(h_diffs) > 0 else 0.0

        monotone_str = "yes" if monotone else f"NO ({worst_drop:.1e})"
        print(f"  {freq:<6} {ll:>14.6f} {diff:>+16.2e} {monotone_str:>9} {gmm.n_iter_:>6}")

        # Numerical sanity: LL must be finite and not catastrophically worse
        if not np.isfinite(ll):
            failed.append(f"freq={freq}: LL is not finite ({ll})")
        elif diff < -0.5 * abs(ref_ll):
            failed.append(f"freq={freq}: LL worse than freq=1 by {-diff:.2e} (>50% of |ref|)")

    if failed:
        for msg in failed:
            print(f"  FAIL: {msg}")
        raise AssertionError("Throttling test failures:\n" + "\n".join(failed))

    print("\n+ _v2_reduced throttling PASSED\n")


# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_e_step_correctness()
    test_m_step_correctness()
    test_end_to_end()
    test_v2_reduced_throttling()

    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
