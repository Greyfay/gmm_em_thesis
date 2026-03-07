#!/usr/bin/env python3
"""Convergence benchmark: _v1 vs _v2_tiling across tile sizes.

Outer sweep: D in [10, 20, 80].
Fixed: N=100,000.
Per D: v1 baseline + v2_tiling with each tile size in TILE_SIZES.

All configurations share the same initialisation within each run so that
differences in ms/iter, n_iter, and ΔLL are attributable purely to the
M-step implementation (einsum vs chunked bmm) and chunk size.

Measures per run:
  - avg time per EM iteration (E-step + M-step + Cholesky)
  - number of iterations to convergence
  - log-likelihood change: lower_bound[final] - lower_bound[initial]

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

from implementation import _v1, _v2_tiling

DEVICE     = torch.device("cuda")

K          = 5
N          = 100_000
D_VALUES   = [10, 20, 80]
N_RUNS     = 30
COV_TYPE   = "full"
REG_COVAR  = 1e-4
TOL        = 1e-3
MAX_ITER   = 200
TILE_SIZES = [128, 256, 512, 1024, 2048, 4096]


# ---------------------------------------------------------------------------
# Core EM runner
# ---------------------------------------------------------------------------

def _run_em(X_t, p_init, module, chunk_N=None):
    """Run EM to convergence from p_init using the given module.

    chunk_N=None  → v1 (no tile_B argument passed to _maximization_step)
    chunk_N=int   → v2_tiling with that chunk size

    Times only: E-step + M-step + Cholesky per iteration.

    Returns:
        iter_times_ms : list[float]  per-iteration wall times in ms
        n_iter        : int          iterations until convergence
        initial_ll    : float        lower bound at iteration 0
        final_ll      : float        lower bound at convergence
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
        torch.cuda.synchronize()
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

        torch.cuda.synchronize()
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

    if n_iter == MAX_ITER:
        print(f"  WARNING: did not converge within {MAX_ITER} iterations", flush=True)

    return iter_times, n_iter, initial_ll, final_ll


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_table(all_rows):
    W   = 106
    sep = "-" * W

    print()
    print("=" * W)
    print(
        f"  TILING BENCHMARK   K={K}  COV_TYPE={COV_TYPE}  "
        f"tol={TOL}  max_iter={MAX_ITER}  runs={N_RUNS}"
    )
    print("=" * W)
    header = (
        f"  {'D':>4}  {'config':>12}  "
        f"{'ms/iter':>16}  "
        f"{'speedup':>8}  "
        f"{'n_iter':>12}  "
        f"{'ΔLL':>16}"
    )
    print(header)

    prev_D = None
    for row in all_rows:
        if row["D"] != prev_D:
            print(sep)
            prev_D = row["D"]

        t_s  = f"{row['iter_time_mean_ms']:.3f} ± {row['iter_time_std_ms']:.3f}"
        n_s  = f"{row['n_iter_mean']:.1f} ± {row['n_iter_std']:.1f}"
        ll_s = f"{row['ll_change_mean']:.3f} ± {row['ll_change_std']:.3f}"
        spd  = f"{row['speedup_x']:.3f}x" if not np.isnan(row["speedup_x"]) else "  —  "

        print(
            f"  {row['D']:>4}  {row['config']:>12}  "
            f"{t_s:>16}  "
            f"{spd:>8}  "
            f"{n_s:>12}  "
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

    for D in D_VALUES:
        print(
            f"\n[D={D}  N={N:,}]  "
            f"K={K}  —  {N_RUNS} runs × {1 + len(TILE_SIZES)} configs",
            flush=True,
        )

        # Warmup: a few untimed iterations on a small dummy dataset
        Xw     = torch.randn(min(N, 1000), D, dtype=torch.float32, device=DEVICE)
        p_warm = _v1.TorchGaussianMixture(
            n_components=K, covariance_type=COV_TYPE,
            reg_covar=REG_COVAR, init_params="random_from_data",
        )._initialize(Xw)
        for _ in range(2):
            _, lr = _v1._expectation_step_precchol(
                Xw, p_warm.means, p_warm.prec_chol, p_warm.weights, COV_TYPE)
            mn, cv, wt = _v1._maximization_step(
                Xw, p_warm.means, p_warm.cov, p_warm.weights, lr, COV_TYPE,
                reg_covar=REG_COVAR)
            _v1._compute_precisions_cholesky(cv, COV_TYPE)
        for tile in TILE_SIZES:
            for _ in range(2):
                _, lr = _v2_tiling._expectation_step_precchol(
                    Xw, p_warm.means, p_warm.prec_chol, p_warm.weights, COV_TYPE)
                mn, cv, wt = _v2_tiling._maximization_step(
                    Xw, p_warm.means, p_warm.cov, p_warm.weights, lr, COV_TYPE,
                    reg_covar=REG_COVAR, tile_B=tile)
                _v2_tiling._compute_precisions_cholesky(cv, COV_TYPE)

        # Per-run accumulators: config label → list of per-run avg times / n_iter / ll_delta
        configs    = ["v1"] + [f"tile={t}" for t in TILE_SIZES]
        avg_times  = {c: [] for c in configs}
        n_iters    = {c: [] for c in configs}
        ll_changes = {c: [] for c in configs}

        for run_idx in range(N_RUNS):
            X_np   = rng.standard_normal((N, D)).astype(np.float32)
            X_t    = torch.from_numpy(X_np).to(DEVICE)

            # Shared initialisation across all configs for this run
            p_init = _v1.TorchGaussianMixture(
                n_components=K, covariance_type=COV_TYPE,
                reg_covar=REG_COVAR, init_params="random_from_data",
            )._initialize(X_t)

            # v1 baseline
            times, n, ll0, llf = _run_em(X_t, p_init, _v1, chunk_N=None)
            avg_times["v1"].append(float(np.mean(times)))
            n_iters["v1"].append(n)
            ll_changes["v1"].append(llf - ll0)

            # v2_tiling with each tile size
            for tile in TILE_SIZES:
                label = f"tile={tile}"
                times, n, ll0, llf = _run_em(X_t, p_init, _v2_tiling, chunk_N=tile)
                avg_times[label].append(float(np.mean(times)))
                n_iters[label].append(n)
                ll_changes[label].append(llf - ll0)

            v1_t      = avg_times["v1"][-1]
            best_tile = min(TILE_SIZES, key=lambda t: avg_times[f"tile={t}"][-1])
            best_t    = avg_times[f"tile={best_tile}"][-1]
            print(
                f"  run {run_idx+1:2d}/{N_RUNS}  "
                f"v1={v1_t:.2f}ms  "
                f"best tile={best_tile} @ {best_t:.2f}ms  "
                f"({v1_t/best_t:.2f}x)",
                flush=True,
            )

            del X_t
            torch.cuda.empty_cache()

        # v1 baseline reference for speedup computation
        v1_mean = float(np.mean(avg_times["v1"]))

        for config in configs:
            c_mean = float(np.mean(avg_times[config]))
            row = {
                "D":                    D,
                "N":                    N,
                "K":                    K,
                "config":               config,
                "iter_time_mean_ms":    round(c_mean, 4),
                "iter_time_std_ms":     round(float(np.std(avg_times[config])), 4),
                "n_iter_mean":          round(float(np.mean(n_iters[config])), 2),
                "n_iter_std":           round(float(np.std(n_iters[config])), 2),
                "ll_change_mean":       round(float(np.mean(ll_changes[config])), 4),
                "ll_change_std":        round(float(np.std(ll_changes[config])), 4),
                "speedup_x":            round(v1_mean / c_mean if c_mean > 0 else float("nan"), 4),
            }
            all_rows.append(row)

        print(f"[D={D}  N={N:,}]  done.", flush=True)
        torch.cuda.empty_cache()

    _print_table(all_rows)

    df = pd.DataFrame(all_rows)
    output_path = os.path.join(os.path.dirname(__file__), "convergence_v1_v2_tiling.csv")
    df.to_csv(output_path, index=False)
    print(f"CSV saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
