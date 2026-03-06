#!/usr/bin/env python3
"""Convergence benchmark: _v1 vs _v2_reduced_covariance_updates.

Two sweeps, combined into a single flat experiment list:

  N sweep  — D fixed at 10 or 100, N in [1e4, 1e5, 1e6].
             Tests how the freq trade-off scales with dataset size.

  D sweep  — N fixed at 10,000, D in [10, 50, 250, 500].
             Tests how the freq trade-off scales with dimensionality.
             N=10,000 keeps each problem well-enough conditioned across all D.

covariance_update_frequency=1 is the v1 baseline (full M-step every iteration).
Higher values skip the covariance recomputation on non-multiple iterations,
trading convergence rate against per-iteration cost.

Measures per run:
  - avg time per EM iteration (E-step + M-step + Cholesky)
  - number of iterations to convergence
  - actual covariance updates performed
  - log-likelihood change: lower_bound[final] - lower_bound[initial]
  - total wall time = sum of all timed iteration times

Results are aggregated as mean ± std across N_RUNS runs.
Outputs a summary table to stdout and a CSV file.
"""

import sys
import os
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from implementation._v1 import (
    GMMParams,
    TorchGaussianMixture as _BaseGMM,
    _expectation_step_precchol,
    _maximization_step,
    _compute_precisions_cholesky,
)
from implementation._v2_reduced_covariance_updates import _maximization_step_reduced

DEVICE    = torch.device("cuda")

K         = 5
N_RUNS    = 30
COV_TYPE  = "full"
REG_COVAR = 1e-4
TOL       = 1e-3
MAX_ITER  = 200
COV_FREQS = [1, 2, 4, 8]   # 1 == v1 baseline

# N sweep: two fixed D values, three N values each
_N_SWEEP_N = [10_000, 100_000, 1_000_000]
_N_SWEEP   = [(10, N) for N in _N_SWEEP_N] + [(100, N) for N in _N_SWEEP_N]

# D sweep: fixed N=10,000, four D values
_D_SWEEP_N = 10_000
_D_SWEEP   = [(D, _D_SWEEP_N) for D in [10, 50, 250, 500]]

# Combined — (10, 10_000) appears in both sweeps; deduplicate while preserving order
_seen      = set()
EXPERIMENTS = []
for pair in _N_SWEEP + _D_SWEEP:
    if pair not in _seen:
        _seen.add(pair)
        EXPERIMENTS.append(pair)


# ---------------------------------------------------------------------------
# Core EM runner
# ---------------------------------------------------------------------------

def _run_em(X_t, p_init, cov_freq):
    """Run EM to convergence using covariance_update_frequency=cov_freq.

    cov_freq=1 reproduces v1 exactly (full M-step every iteration).

    Times only: E-step + M-step + Cholesky per iteration.

    Returns:
        iter_times_ms : list[float]
        n_iter        : int
        cov_updates   : int
        initial_ll    : float
        final_ll      : float
    """
    p = GMMParams(
        weights=p_init.weights.clone(),
        means=p_init.means.clone(),
        cov=p_init.cov.clone(),
        prec_chol=p_init.prec_chol.clone(),
        cov_type=COV_TYPE,
    )

    prev_lower  = float("-inf")
    iter_times  = []
    initial_ll  = None
    final_ll    = None
    n_iter      = MAX_ITER
    cov_updates = 0

    for it in range(MAX_ITER):
        update_cov = (it % cov_freq == 0)

        # --- timed block: E-step + M-step + Cholesky ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        lower, log_resp = _expectation_step_precchol(
            X_t, p.means, p.prec_chol, p.weights, COV_TYPE
        )
        means, cov, weights = _maximization_step_reduced(
            X_t, p.means, p.cov, p.weights, log_resp, COV_TYPE,
            reg_covar=REG_COVAR, update_covariance=update_cov,
        )
        prec_chol = _compute_precisions_cholesky(cov, COV_TYPE)

        torch.cuda.synchronize()
        iter_times.append((time.perf_counter() - t0) * 1e3)
        # --- end timed block ---

        if update_cov:
            cov_updates += 1

        p = GMMParams(
            weights=weights, means=means, cov=cov,
            prec_chol=prec_chol, cov_type=COV_TYPE,
        )

        lower_f = float(lower.item())
        if initial_ll is None:
            initial_ll = lower_f
        final_ll = lower_f

        if abs(lower_f - prev_lower) < TOL:
            n_iter = it + 1
            break
        prev_lower = lower_f

    return iter_times, n_iter, cov_updates, initial_ll, final_ll


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_table(all_rows):
    W   = 114
    sep = "-" * W

    print()
    print("=" * W)
    print(
        f"  REDUCED COV BENCHMARK   K={K}  COV_TYPE={COV_TYPE}  "
        f"tol={TOL}  max_iter={MAX_ITER}  runs={N_RUNS}"
    )
    print("=" * W)
    header = (
        f"  {'D':>4}  {'N':>10}  {'freq':>6}  "
        f"{'ms/iter':>16}  "
        f"{'n_iter':>12}  "
        f"{'cov_updates':>12}  "
        f"{'total ms':>12}  "
        f"{'ΔLL':>16}"
    )
    print(header)

    prev_key = None
    for row in all_rows:
        key = (row["D"], row["N"])
        if key != prev_key:
            print(sep)
            prev_key = key

        t_s   = f"{row['iter_time_mean_ms']:.3f} ± {row['iter_time_std_ms']:.3f}"
        n_s   = f"{row['n_iter_mean']:.1f} ± {row['n_iter_std']:.1f}"
        cu_s  = f"{row['cov_updates_mean']:.1f} ± {row['cov_updates_std']:.1f}"
        tot_s = f"{row['total_time_mean_ms']:.1f} ± {row['total_time_std_ms']:.1f}"
        ll_s  = f"{row['ll_change_mean']:.3f} ± {row['ll_change_std']:.3f}"
        freq_label = f"{row['cov_freq']}" + (" (v1)" if row["cov_freq"] == 1 else "")

        print(
            f"  {row['D']:>4}  {row['N']:>10,}  {freq_label:>6}  "
            f"{t_s:>16}  "
            f"{n_s:>12}  "
            f"{cu_s:>12}  "
            f"{tot_s:>12}  "
            f"{ll_s:>16}"
        )

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — this benchmark requires a GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    all_rows = []

    for exp_idx, (D, N) in enumerate(EXPERIMENTS):
        print(
            f"\n[{exp_idx+1}/{len(EXPERIMENTS)}  D={D}  N={N:,}]  "
            f"K={K}  —  {N_RUNS} runs × {len(COV_FREQS)} frequencies",
            flush=True,
        )

        # Warmup: a few untimed iterations on a small dummy dataset
        Xw   = torch.randn(min(N, 1000), D, dtype=torch.float32, device=DEVICE)
        p_w  = _BaseGMM(
            n_components=K, covariance_type=COV_TYPE,
            reg_covar=REG_COVAR, init_params="random_from_data",
        )._initialize(Xw)
        for it_w in range(4):
            _, lr = _expectation_step_precchol(Xw, p_w.means, p_w.prec_chol, p_w.weights, COV_TYPE)
            mn, cv, wt = _maximization_step_reduced(
                Xw, p_w.means, p_w.cov, p_w.weights, lr, COV_TYPE,
                reg_covar=REG_COVAR, update_covariance=(it_w % 2 == 0),
            )
            _compute_precisions_cholesky(cv, COV_TYPE)

        # Accumulators keyed by freq
        avg_times_all   = {f: [] for f in COV_FREQS}
        n_iters_all     = {f: [] for f in COV_FREQS}
        cov_updates_all = {f: [] for f in COV_FREQS}
        ll_changes_all  = {f: [] for f in COV_FREQS}
        total_times_all = {f: [] for f in COV_FREQS}

        for run_idx in range(N_RUNS):
            # All frequencies share the same dataset and initialisation for this run
            X_np   = rng.standard_normal((N, D)).astype(np.float32)
            X_t    = torch.from_numpy(X_np).to(DEVICE)
            p_init = _BaseGMM(
                n_components=K, covariance_type=COV_TYPE,
                reg_covar=REG_COVAR, init_params="random_from_data",
            )._initialize(X_t)

            run_summary = []
            for freq in COV_FREQS:
                times, n_iter, cov_upd, ll0, llf = _run_em(X_t, p_init, freq)
                avg_times_all[freq].append(float(np.mean(times)))
                n_iters_all[freq].append(n_iter)
                cov_updates_all[freq].append(cov_upd)
                ll_changes_all[freq].append(llf - ll0)
                total_times_all[freq].append(float(np.sum(times)))
                run_summary.append(
                    f"freq={freq}: {n_iter}it {float(np.sum(times)):.1f}ms total"
                )

            print(
                f"  run {run_idx+1:2d}/{N_RUNS}  " + "  |  ".join(run_summary),
                flush=True,
            )

        for freq in COV_FREQS:
            row = {
                "D":                    D,
                "N":                    N,
                "K":                    K,
                "cov_freq":             freq,
                "iter_time_mean_ms":    round(float(np.mean(avg_times_all[freq])),   4),
                "iter_time_std_ms":     round(float(np.std(avg_times_all[freq])),    4),
                "n_iter_mean":          round(float(np.mean(n_iters_all[freq])),     2),
                "n_iter_std":           round(float(np.std(n_iters_all[freq])),      2),
                "cov_updates_mean":     round(float(np.mean(cov_updates_all[freq])), 2),
                "cov_updates_std":      round(float(np.std(cov_updates_all[freq])),  2),
                "total_time_mean_ms":   round(float(np.mean(total_times_all[freq])), 2),
                "total_time_std_ms":    round(float(np.std(total_times_all[freq])),  2),
                "ll_change_mean":       round(float(np.mean(ll_changes_all[freq])),  4),
                "ll_change_std":        round(float(np.std(ll_changes_all[freq])),   4),
            }
            all_rows.append(row)

        print(f"[D={D}  N={N:,}]  done.", flush=True)

    _print_table(all_rows)

    df = pd.DataFrame(all_rows)
    output_path = os.path.join(os.path.dirname(__file__), "convergence_v1_v2_reduced.csv")
    df.to_csv(output_path, index=False)
    print(f"CSV saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
