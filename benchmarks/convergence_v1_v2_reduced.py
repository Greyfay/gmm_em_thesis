#!/usr/bin/env python3
"""Convergence benchmark: _v1 vs _v2_reduced_covariance_updates.

Fixed: K=5, D=100, COV_TYPE='full'. N in [1e4, 1e5, 1e6].
N_RUNS independent runs per (N, covariance_update_frequency) configuration.

covariance_update_frequency=1 is the v1 baseline (full M-step every iteration).
Higher values skip the covariance recomputation on non-multiple iterations,
trading convergence rate against per-iteration cost.

Measures per run:
  - avg time per EM iteration (E-step + M-step + Cholesky)
  - number of iterations to convergence
  - actual covariance updates performed
  - log-likelihood change: lower_bound[final] - lower_bound[initial]
  - total wall time = sum of all timed iteration times (derived)

Results are aggregated as mean ± std across runs.
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

K           = 5
D           = 100
N_VALUES    = [10_000, 100_000, 1_000_000]
N_RUNS      = 30
COV_TYPE    = "full"
REG_COVAR   = 1e-4
TOL         = 1e-3
MAX_ITER    = 200
COV_FREQS   = [1, 2, 4, 8]   # 1 == v1 baseline


# ---------------------------------------------------------------------------
# Core EM runner
# ---------------------------------------------------------------------------

def _run_em(X_t, p_init, cov_freq):
    """Run EM to convergence from p_init using covariance_update_frequency=cov_freq.

    cov_freq=1 reproduces v1 exactly (full M-step every iteration).
    cov_freq>1 skips covariance recomputation on non-multiple iterations.

    Times only: E-step + M-step + Cholesky per iteration.

    Returns:
        iter_times_ms  : list[float]  per-iteration wall times in ms
        n_iter         : int          iterations until convergence
        cov_updates    : int          how many times covariance was actually updated
        initial_ll     : float        lower bound at iteration 0
        final_ll       : float        lower bound at convergence
    """
    p = GMMParams(
        weights=p_init.weights.clone(),
        means=p_init.means.clone(),
        cov=p_init.cov.clone(),
        prec_chol=p_init.prec_chol.clone(),
        cov_type=COV_TYPE,
    )

    prev_lower   = float("-inf")
    iter_times   = []
    initial_ll   = None
    final_ll     = None
    n_iter       = MAX_ITER
    cov_updates  = 0

    for it in range(MAX_ITER):
        update_cov = (it % cov_freq == 0)

        # --- timed block: E-step + M-step + Cholesky ---
        t0 = time.perf_counter()

        lower, log_resp = _expectation_step_precchol(
            X_t, p.means, p.prec_chol, p.weights, COV_TYPE
        )

        means, cov, weights = _maximization_step_reduced(
            X_t, p.means, p.cov, p.weights, log_resp, COV_TYPE,
            reg_covar=REG_COVAR,
            update_covariance=update_cov,
        )

        prec_chol = _compute_precisions_cholesky(cov, COV_TYPE)

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
    W   = 112
    sep = "-" * W

    print()
    print("=" * W)
    print(
        f"  CONVERGENCE BENCHMARK (reduced cov updates)   K={K}  D={D}  "
        f"COV_TYPE={COV_TYPE}  tol={TOL}  max_iter={MAX_ITER}  runs={N_RUNS}"
    )
    print("=" * W)
    header = (
        f"  {'N':>10}  {'freq':>5}  "
        f"{'ms/iter':>16}  "
        f"{'n_iter':>12}  "
        f"{'cov_updates':>12}  "
        f"{'total ms':>12}  "
        f"{'ΔLL':>16}"
    )
    print(header)

    prev_N = None
    for row in all_rows:
        if row["N"] != prev_N:
            print(sep)
            prev_N = row["N"]

        t_s   = f"{row['iter_time_mean_ms']:.3f} ± {row['iter_time_std_ms']:.3f}"
        n_s   = f"{row['n_iter_mean']:.1f} ± {row['n_iter_std']:.1f}"
        cu_s  = f"{row['cov_updates_mean']:.1f} ± {row['cov_updates_std']:.1f}"
        tot_s = f"{row['total_time_mean_ms']:.1f} ± {row['total_time_std_ms']:.1f}"
        ll_s  = f"{row['ll_change_mean']:.3f} ± {row['ll_change_std']:.3f}"

        freq_label = f"{row['cov_freq']}" + (" (v1)" if row["cov_freq"] == 1 else "")
        print(
            f"  {row['N']:>10,}  {freq_label:>5}  "
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
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    all_rows = []

    for i, N in enumerate(N_VALUES):
        print(
            f"\n[{i+1}/{len(N_VALUES)}] N={N:,}  K={K}  D={D}  —  "
            f"{N_RUNS} runs × {len(COV_FREQS)} frequencies",
            flush=True,
        )

        # Warmup: a few untimed iterations on a small dummy dataset
        Xw    = torch.randn(min(N, 1000), D, dtype=torch.float32)
        p_w   = _BaseGMM(
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

        # Pre-generate datasets for all runs
        datasets = []
        for _ in range(N_RUNS):
            X_np = rng.standard_normal((N, D)).astype(np.float32)
            datasets.append(torch.from_numpy(X_np))

        for freq in COV_FREQS:
            freq_label = f"freq={freq}" + (" [v1 baseline]" if freq == 1 else "")
            print(f"  {freq_label}", flush=True)

            avg_times_per_run  = []
            n_iters_per_run    = []
            cov_updates_per_run = []
            ll_changes_per_run = []
            total_times_per_run = []

            for run_idx, X_t in enumerate(datasets):
                # Fresh shared initialization for each run (same across frequencies)
                p_init = _BaseGMM(
                    n_components=K, covariance_type=COV_TYPE,
                    reg_covar=REG_COVAR, init_params="random_from_data",
                )._initialize(X_t)

                times, n_iter, cov_upd, ll0, llf = _run_em(X_t, p_init, freq)

                avg_t    = float(np.mean(times))
                total_t  = float(np.sum(times))
                ll_delta = llf - ll0

                avg_times_per_run.append(avg_t)
                n_iters_per_run.append(n_iter)
                cov_updates_per_run.append(cov_upd)
                ll_changes_per_run.append(ll_delta)
                total_times_per_run.append(total_t)

                print(
                    f"    run {run_idx+1:2d}/{N_RUNS}  "
                    f"{n_iter:3d} iters  "
                    f"cov_updates={cov_upd:3d}  "
                    f"avg {avg_t:8.2f} ms/iter  "
                    f"total {total_t:8.1f} ms  "
                    f"ΔLL={ll_delta:+.3f}",
                    flush=True,
                )

            row = {
                "N":                    N,
                "K":                    K,
                "D":                    D,
                "cov_freq":             freq,
                "iter_time_mean_ms":    round(float(np.mean(avg_times_per_run)),   4),
                "iter_time_std_ms":     round(float(np.std(avg_times_per_run)),    4),
                "n_iter_mean":          round(float(np.mean(n_iters_per_run)),     2),
                "n_iter_std":           round(float(np.std(n_iters_per_run)),      2),
                "cov_updates_mean":     round(float(np.mean(cov_updates_per_run)), 2),
                "cov_updates_std":      round(float(np.std(cov_updates_per_run)),  2),
                "total_time_mean_ms":   round(float(np.mean(total_times_per_run)), 2),
                "total_time_std_ms":    round(float(np.std(total_times_per_run)),  2),
                "ll_change_mean":       round(float(np.mean(ll_changes_per_run)),  4),
                "ll_change_std":        round(float(np.std(ll_changes_per_run)),   4),
            }
            all_rows.append(row)

        print(f"[{i+1}/{len(N_VALUES)}] N={N:,}  done.", flush=True)

    _print_table(all_rows)

    df = pd.DataFrame(all_rows)
    output_path = os.path.join(os.path.dirname(__file__), "convergence_v1_v2_reduced.csv")
    df.to_csv(output_path, index=False)
    print(f"CSV saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
