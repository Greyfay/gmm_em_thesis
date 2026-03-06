#!/usr/bin/env python3
"""Convergence benchmark: _v1 vs _v2_tiling.

Fixed: K=5, D=20, COV_TYPE='full'. N in [1e4, 1e5, 1e6].
N_RUNS independent runs per configuration (each with fresh random data and shared init).

Measures per run:
  - avg time per EM iteration (E-step + M-step + Cholesky factorization)
  - number of iterations to convergence
  - log-likelihood change: lower_bound[final] - lower_bound[initial]

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

from implementation import _v1, _v2_tiling

K         = 5
D         = 20
N_VALUES  = [10_000, 100_000, 1_000_000]
N_RUNS    = 30
COV_TYPE  = "full"
REG_COVAR = 1e-4
TOL       = 1e-3
MAX_ITER  = 200
V2_CHUNK  = 1024   # chunk_N (tile_B) passed to _v2_tiling._maximization_step


# ---------------------------------------------------------------------------
# Core EM runner
# ---------------------------------------------------------------------------

def _run_em(X_t, p_init, module, chunk_N=None):
    """Run EM to convergence from p_init using the given module.

    p_init is a GMMParams-like object (weights/means/cov/prec_chol/cov_type).
    Times only: E-step + M-step + Cholesky per iteration.

    Returns:
        iter_times_ms  : list[float]  per-iteration wall times in ms
        n_iter         : int          iterations until convergence (capped at MAX_ITER)
        initial_ll     : float        lower bound at iteration 0
        final_ll       : float        lower bound at convergence
    """
    p = module.GMMParams(
        weights=p_init.weights.clone(),
        means=p_init.means.clone(),
        cov=p_init.cov.clone(),
        prec_chol=p_init.prec_chol.clone(),
        cov_type=COV_TYPE,
    )

    prev_lower = float("-inf")
    iter_times = []
    initial_ll = None
    final_ll   = None
    n_iter     = MAX_ITER

    for it in range(MAX_ITER):
        # --- timed block: E-step + M-step + Cholesky ---
        t0 = time.perf_counter()

        lower, log_resp = module._expectation_step_precchol(
            X_t, p.means, p.prec_chol, p.weights, COV_TYPE
        )

        if chunk_N is not None:
            means, cov, weights = module._maximization_step(
                X_t, p.means, p.cov, p.weights, log_resp, COV_TYPE,
                reg_covar=REG_COVAR, tile_B=chunk_N,
            )
        else:
            means, cov, weights = module._maximization_step(
                X_t, p.means, p.cov, p.weights, log_resp, COV_TYPE,
                reg_covar=REG_COVAR,
            )

        prec_chol = module._compute_precisions_cholesky(cov, COV_TYPE)

        iter_times.append((time.perf_counter() - t0) * 1e3)
        # --- end timed block ---

        p = module.GMMParams(
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

    return iter_times, n_iter, initial_ll, final_ll


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_table(all_rows):
    W   = 108
    sep = "-" * W

    print()
    print("=" * W)
    print(
        f"  CONVERGENCE BENCHMARK   K={K}  D={D}  COV_TYPE={COV_TYPE}  "
        f"tol={TOL}  max_iter={MAX_ITER}  runs={N_RUNS}  v2_chunk={V2_CHUNK}"
    )
    print("=" * W)
    header = (
        f"  {'N':>10}  "
        f"{'v1 ms/iter':>16}  "
        f"{'v2 ms/iter':>16}  "
        f"{'speedup':>8}  "
        f"{'v1 n_iter':>12}  "
        f"{'v2 n_iter':>12}  "
        f"{'v1 ΔLL':>14}  "
        f"{'v2 ΔLL':>14}"
    )
    print(header)
    print(sep)

    for row in all_rows:
        speedup = (
            row["v1_iter_time_mean_ms"] / row["v2_iter_time_mean_ms"]
            if row["v2_iter_time_mean_ms"] > 0 else float("nan")
        )
        v1_t  = f"{row['v1_iter_time_mean_ms']:.3f} ± {row['v1_iter_time_std_ms']:.3f}"
        v2_t  = f"{row['v2_iter_time_mean_ms']:.3f} ± {row['v2_iter_time_std_ms']:.3f}"
        v1_n  = f"{row['v1_n_iter_mean']:.1f} ± {row['v1_n_iter_std']:.1f}"
        v2_n  = f"{row['v2_n_iter_mean']:.1f} ± {row['v2_n_iter_std']:.1f}"
        v1_ll = f"{row['v1_ll_change_mean']:.3f} ± {row['v1_ll_change_std']:.3f}"
        v2_ll = f"{row['v2_ll_change_mean']:.3f} ± {row['v2_ll_change_std']:.3f}"
        print(
            f"  {row['N']:>10,}  "
            f"{v1_t:>16}  "
            f"{v2_t:>16}  "
            f"{speedup:>7.3f}x  "
            f"{v1_n:>12}  "
            f"{v2_n:>12}  "
            f"{v1_ll:>14}  "
            f"{v2_ll:>14}"
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
            f"\n[{i+1}/{len(N_VALUES)}] N={N:,}  K={K}  D={D}  —  {N_RUNS} runs",
            flush=True,
        )

        v1_avg_times  = []
        v2_avg_times  = []
        v1_n_iters    = []
        v2_n_iters    = []
        v1_ll_changes = []
        v2_ll_changes = []

        # Warmup: one untimed run of 2 iterations on a small dummy dataset
        Xw = torch.randn(min(N, 1000), D, dtype=torch.float32)
        p_warm = _v1.TorchGaussianMixture(
            n_components=K, covariance_type=COV_TYPE,
            reg_covar=REG_COVAR, init_params="random_from_data",
        )._initialize(Xw)
        for _ in range(2):
            _, lr = _v1._expectation_step_precchol(Xw, p_warm.means, p_warm.prec_chol, p_warm.weights, COV_TYPE)
            mn, cv, wt = _v1._maximization_step(Xw, p_warm.means, p_warm.cov, p_warm.weights, lr, COV_TYPE, reg_covar=REG_COVAR)
            _v1._compute_precisions_cholesky(cv, COV_TYPE)
        for _ in range(2):
            _, lr = _v2_tiling._expectation_step_precchol(Xw, p_warm.means, p_warm.prec_chol, p_warm.weights, COV_TYPE)
            mn, cv, wt = _v2_tiling._maximization_step(Xw, p_warm.means, p_warm.cov, p_warm.weights, lr, COV_TYPE, reg_covar=REG_COVAR, tile_B=V2_CHUNK)
            _v2_tiling._compute_precisions_cholesky(cv, COV_TYPE)

        for run_idx in range(N_RUNS):
            print(f"  run {run_idx+1:2d}/{N_RUNS}  generating data...", end=" ", flush=True)

            X_np = rng.standard_normal((N, D)).astype(np.float32)
            X_t  = torch.from_numpy(X_np)

            # Shared initialization (v1's _initialize; params are pure tensors)
            p_init = _v1.TorchGaussianMixture(
                n_components=K, covariance_type=COV_TYPE,
                reg_covar=REG_COVAR, init_params="random_from_data",
            )._initialize(X_t)

            # --- v1 ---
            v1_times, v1_n, v1_ll0, v1_llf = _run_em(X_t, p_init, _v1, chunk_N=None)
            v1_avg_times.append(float(np.mean(v1_times)))
            v1_n_iters.append(v1_n)
            v1_ll_changes.append(v1_llf - v1_ll0)

            # --- v2_tiling ---
            v2_times, v2_n, v2_ll0, v2_llf = _run_em(X_t, p_init, _v2_tiling, chunk_N=V2_CHUNK)
            v2_avg_times.append(float(np.mean(v2_times)))
            v2_n_iters.append(v2_n)
            v2_ll_changes.append(v2_llf - v2_ll0)

            print(
                f"v1: {v1_n:3d} iters @ {np.mean(v1_times):8.2f} ms/iter  ΔLL={v1_llf - v1_ll0:+.3f}  |  "
                f"v2: {v2_n:3d} iters @ {np.mean(v2_times):8.2f} ms/iter  ΔLL={v2_llf - v2_ll0:+.3f}",
                flush=True,
            )

        row = {
            "N":                    N,
            "K":                    K,
            "D":                    D,
            "v1_iter_time_mean_ms": round(float(np.mean(v1_avg_times)),  4),
            "v1_iter_time_std_ms":  round(float(np.std(v1_avg_times)),   4),
            "v1_n_iter_mean":       round(float(np.mean(v1_n_iters)),    2),
            "v1_n_iter_std":        round(float(np.std(v1_n_iters)),     2),
            "v1_ll_change_mean":    round(float(np.mean(v1_ll_changes)), 4),
            "v1_ll_change_std":     round(float(np.std(v1_ll_changes)),  4),
            "v2_iter_time_mean_ms": round(float(np.mean(v2_avg_times)),  4),
            "v2_iter_time_std_ms":  round(float(np.std(v2_avg_times)),   4),
            "v2_n_iter_mean":       round(float(np.mean(v2_n_iters)),    2),
            "v2_n_iter_std":        round(float(np.std(v2_n_iters)),     2),
            "v2_ll_change_mean":    round(float(np.mean(v2_ll_changes)), 4),
            "v2_ll_change_std":     round(float(np.std(v2_ll_changes)),  4),
            "speedup_x":            round(
                float(np.mean(v1_avg_times)) / float(np.mean(v2_avg_times))
                if float(np.mean(v2_avg_times)) > 0 else float("nan"), 4
            ),
        }
        all_rows.append(row)
        print(f"[{i+1}/{len(N_VALUES)}] N={N:,}  done.", flush=True)

    _print_table(all_rows)

    df = pd.DataFrame(all_rows)
    output_path = os.path.join(os.path.dirname(__file__), "convergence_v1_v2_tiling.csv")
    df.to_csv(output_path, index=False)
    print(f"CSV saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
