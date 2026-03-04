"""
Performance comparison test for _v1.py vs _v2.py implementations.

This test measures (CUDA-accurate):
1. Total fit runtime (seconds)            [includes E+M steps, excludes data generation]
2. Time spent in covariance update (sec)  [ONLY the covariance part of M-step]
3. Final log likelihood (lower_bound_)
4. Iterations to converge (n_iter_)

Design notes:
- Uses CUDA synchronize around timed regions (GPU kernels are async).
- Uses multiple runs with warmup and reports mean/std.
- Measures covariance time by monkeypatching _maximization_step and timing ONLY the
  covariance-update region, reproducing the exact covariance path for each impl:
    * v1: monolithic einsum full-cov update
    * v2: tiled full-cov update + upper-triangular tiling + mirroring (symmetry)

IMPORTANT:
- We reset covariance stats AFTER warmup so totals correspond to the timed runs only.
- We run everything under torch.no_grad() for clean benchmarking.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from implementation import _v1 as v1
from implementation import _v2 as v2


# -----------------------
# Data generation
# -----------------------

def generate_test_data(
    N: int,
    D: int,
    random_seed: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate synthetic test data for benchmarking."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    return torch.randn(N, D, device=device, dtype=dtype)


# -----------------------
# Timing helpers
# -----------------------

def _cuda_sync_if_needed(x: torch.Tensor) -> None:
    if x.is_cuda:
        torch.cuda.synchronize()


def measure_fit_runtime(
    make_model_fn,
    X: torch.Tensor,
    n_runs: int = 5,
    warmup: int = 2,
    on_after_warmup=None,
) -> Tuple[float, float, Any]:
    """
    Measure mean/std of model.fit time (CUDA-accurate).
    Returns: (mean_s, std_s, last_model)

    NOTE: If provided, on_after_warmup() is called exactly once after warmup.
    """
    # Warmup
    model = None
    for _ in range(warmup):
        model = make_model_fn()
        _cuda_sync_if_needed(X)
        model.fit(X)  # do NOT clone; fit should not mutate X
        _cuda_sync_if_needed(X)

    if on_after_warmup is not None:
        on_after_warmup()

    # Timed runs
    times: List[float] = []
    last_model = None
    for _ in range(n_runs):
        model = make_model_fn()
        _cuda_sync_if_needed(X)
        t0 = time.perf_counter()
        model.fit(X)
        _cuda_sync_if_needed(X)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_model = model

    return float(np.mean(times)), float(np.std(times)), last_model


# -----------------------
# Covariance-only timing (implementation-faithful)
# -----------------------

def make_cov_timer_v1() -> Tuple[Dict[str, float], Any]:
    """
    Monkeypatch v1._maximization_step so we can accumulate covariance-only time
    while still returning the same outputs as the original function.

    We time ONLY the full-covariance update path for v1.
    """
    original = v1._maximization_step
    stats: Dict[str, float] = {"cov_time_s": 0.0, "count": 0.0}

    _nk_eps = v1._nk_eps

    def wrapped(
        X: torch.Tensor,
        means: torch.Tensor,
        cov: torch.Tensor,
        weights: torch.Tensor,
        log_resp: torch.Tensor,
        cov_type: str,
        reg_covar: float = 1e-6,
    ):
        if cov_type != "full":
            return original(X, means, cov, weights, log_resp, cov_type, reg_covar=reg_covar)

        # Shared intermediates (NOT timed)
        resp = log_resp.exp()
        nk = resp.sum(dim=0) + _nk_eps(resp.dtype)
        new_weights = nk / nk.sum()
        new_means = (resp.T @ X) / nk.unsqueeze(1)

        # Time ONLY covariance update portion (CUDA accurate)
        _cuda_sync_if_needed(X)
        t0 = time.perf_counter()

        diff = X.unsqueeze(1) - new_means.unsqueeze(0)  # (N,K,D)
        cov_sum = torch.einsum("nk,nkd,nke->kde", resp, diff, diff)  # (K,D,D)
        new_cov = cov_sum / nk.unsqueeze(1).unsqueeze(2)
        eye = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
        new_cov = new_cov + reg_covar * eye.unsqueeze(0)

        _cuda_sync_if_needed(X)
        t1 = time.perf_counter()

        stats["cov_time_s"] += (t1 - t0)
        stats["count"] += 1.0

        return new_means, new_cov, new_weights

    v1._maximization_step = wrapped
    return stats, original


def make_cov_timer_v2() -> Tuple[Dict[str, float], Any]:
    """
    Monkeypatch v2._maximization_step so we can accumulate covariance-only time
    for the tiled + upper-triangular full-cov path (respects tile_B).

    This matches your current v2 covariance branch:

      - tile over i blocks
      - for each i block, iterate j from i..D (upper triangle)
      - write cov_block into [i,j]
      - if j != i, mirror transpose into [j,i]
    """
    original = v2._maximization_step
    stats: Dict[str, float] = {"cov_time_s": 0.0, "count": 0.0}

    _nk_eps = v2._nk_eps

    def wrapped(
        X: torch.Tensor,
        means: torch.Tensor,
        cov: torch.Tensor,
        weights: torch.Tensor,
        log_resp: torch.Tensor,
        cov_type: str,
        reg_covar: float = 1e-6,
        tile_B: Optional[int] = None,
    ):
        if cov_type != "full":
            return original(
                X, means, cov, weights, log_resp, cov_type, reg_covar=reg_covar, tile_B=tile_B
            )

        # Shared intermediates (NOT timed)
        resp = log_resp.exp()
        nk = resp.sum(dim=0) + _nk_eps(resp.dtype)
        new_weights = nk / nk.sum()
        new_means = (resp.T @ X) / nk.unsqueeze(1)

        # Time ONLY the tiled covariance update (CUDA accurate)
        B = int(tile_B) if tile_B is not None else 64
        if B <= 0:
            raise ValueError(f"tile_B must be positive, got {B}")

        K = new_means.shape[0]
        D = X.shape[1]

        _cuda_sync_if_needed(X)
        t0 = time.perf_counter()

        cov_sum = torch.zeros((K, D, D), device=X.device, dtype=X.dtype)
        nk_ = nk.unsqueeze(1).unsqueeze(2)  # (K,1,1)
        eye = torch.eye(D, device=X.device, dtype=X.dtype)

        # Tiling over feature dimensions (upper-triangular blocks)
        for i in range(0, D, B):
            i2 = min(i + B, D)

            Xi = X[:, i:i2].unsqueeze(1)           # (N,1,Bi)
            Mi = new_means[:, i:i2].unsqueeze(0)   # (1,K,Bi)
            diff_i = Xi - Mi                        # (N,K,Bi)

            for j in range(i, D, B):
                j2 = min(j + B, D)

                Xj = X[:, j:j2].unsqueeze(1)        # (N,1,Bj)
                Mj = new_means[:, j:j2].unsqueeze(0)# (1,K,Bj)
                diff_j = Xj - Mj                     # (N,K,Bj)

                cov_block = torch.einsum("nk,nkb,nkc->kbc", resp, diff_i, diff_j)

                cov_sum[:, i:i2, j:j2] = cov_block
                if j != i:
                    cov_sum[:, j:j2, i:i2] = cov_block.transpose(-1, -2)

        new_cov = cov_sum / nk_
        new_cov = new_cov + reg_covar * eye.unsqueeze(0)

        _cuda_sync_if_needed(X)
        t1 = time.perf_counter()

        stats["cov_time_s"] += (t1 - t0)
        stats["count"] += 1.0

        return new_means, new_cov, new_weights

    v2._maximization_step = wrapped
    return stats, original


# -----------------------
# Tests
# -----------------------

@torch.no_grad()
def test_v1_full_covariance() -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("TEST: v1 (full covariance)")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    device = "cuda"
    dtype = torch.float64
    cov_type = "full"

    test_configs = [
        (1000, 32, 4, 50),
        (1000, 64, 4, 50),
        (1000, 128, 4, 50),
        (1000, 256, 4, 50),
        (1000, 512, 4, 50),
    ]

    for N, D, K, max_iter in test_configs:
        print(f"\n--- Config: N={N}, D={D}, K={K}, max_iter={max_iter} ---")

        X = generate_test_data(N, D, device=device, dtype=dtype)

        cov_stats, original = make_cov_timer_v1()
        try:
            def make_model():
                torch.manual_seed(42)
                np.random.seed(42)
                return v1.TorchGaussianMixture(
                    n_components=K,
                    covariance_type=cov_type,
                    max_iter=max_iter,
                    n_init=1,
                    init_params="random",
                    device=device,
                    dtype=dtype,
                )

            def reset_cov_stats_after_warmup():
                cov_stats["cov_time_s"] = 0.0
                cov_stats["count"] = 0.0

            mean_fit, std_fit, model = measure_fit_runtime(
                make_model, X, n_runs=5, warmup=2, on_after_warmup=reset_cov_stats_after_warmup
            )

            final_ll = float(model.lower_bound_)
            n_iter = int(model.n_iter_)

            cov_time = float(cov_stats["cov_time_s"])
            cov_count = int(cov_stats["count"])
            avg_cov_ms = (cov_time / max(cov_count, 1)) * 1000.0

            print(
                f"✓ fit={mean_fit:.4f}±{std_fit:.4f}s, "
                f"cov={cov_time:.4f}s over {cov_count} calls (avg {avg_cov_ms:.3f} ms), "
                f"ll={final_ll:.4f}, iters={n_iter}"
            )

            results.append({
                "Implementation": "v1",
                "Tiling Size": np.nan,
                "N": N,
                "D": D,
                "K": K,
                "Max Iterations": max_iter,
                "Fit Time Mean (s)": mean_fit,
                "Fit Time Std (s)": std_fit,
                "Cov Time Total (s)": cov_time,
                "Cov Calls": cov_count,
                "Avg Cov Time (ms)": avg_cov_ms,
                "Final Log-Likelihood": final_ll,
                "Iterations to Converge": n_iter,
            })
        finally:
            v1._maximization_step = original

    return results


@torch.no_grad()
def test_v2_tiling_sweep() -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("TEST: v2 (full covariance) tiling_size sweep")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    device = "cuda"
    dtype = torch.float64
    cov_type = "full"

    test_configs = [
        (1000, 32, 4, 50),
        (1000, 64, 4, 50),
        (1000, 128, 4, 50),
        (1000, 256, 4, 50),
        (1000, 512, 4, 50),
    ]

    tiling_sizes = [4, 16, 32, 64, 128]

    for N, D, K, max_iter in test_configs:
        print(f"\n--- Config: N={N}, D={D}, K={K}, max_iter={max_iter} ---")

        X = generate_test_data(N, D, device=device, dtype=dtype)

        for B in tiling_sizes:
            print(f"  tiling_size={B}...", end=" ", flush=True)

            cov_stats, original = make_cov_timer_v2()
            try:
                def make_model():
                    torch.manual_seed(42)
                    np.random.seed(42)
                    return v2.TorchGaussianMixture(
                        n_components=K,
                        covariance_type=cov_type,
                        max_iter=max_iter,
                        n_init=1,
                        init_params="random",
                        device=device,
                        dtype=dtype,
                        tiling_size=B,
                    )

                def reset_cov_stats_after_warmup():
                    cov_stats["cov_time_s"] = 0.0
                    cov_stats["count"] = 0.0

                mean_fit, std_fit, model = measure_fit_runtime(
                    make_model, X, n_runs=5, warmup=2, on_after_warmup=reset_cov_stats_after_warmup
                )

                final_ll = float(model.lower_bound_)
                n_iter = int(model.n_iter_)

                cov_time = float(cov_stats["cov_time_s"])
                cov_count = int(cov_stats["count"])
                avg_cov_ms = (cov_time / max(cov_count, 1)) * 1000.0

                print(
                    f"✓ fit={mean_fit:.4f}±{std_fit:.4f}s, "
                    f"cov={cov_time:.4f}s over {cov_count} calls (avg {avg_cov_ms:.3f} ms), "
                    f"ll={final_ll:.4f}, iters={n_iter}"
                )

                results.append({
                    "Implementation": "v2",
                    "Tiling Size": B,
                    "N": N,
                    "D": D,
                    "K": K,
                    "Max Iterations": max_iter,
                    "Fit Time Mean (s)": mean_fit,
                    "Fit Time Std (s)": std_fit,
                    "Cov Time Total (s)": cov_time,
                    "Cov Calls": cov_count,
                    "Avg Cov Time (ms)": avg_cov_ms,
                    "Final Log-Likelihood": final_ll,
                    "Iterations to Converge": n_iter,
                })
            finally:
                v2._maximization_step = original

    return results


def main():
    if not torch.cuda.is_available():
        print("ERROR: This test requires CUDA (GPU). No CUDA device found.")
        sys.exit(1)

    # Optional: stabilize/accelerate cuDNN heuristics on fixed shapes
    torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("PERFORMANCE TEST: v1 vs v2 (covariance timing + tiling sweep)")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version (torch): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Device: CUDA")
    print(f"CUDA available: {torch.cuda.is_available()}\n")

    torch.set_default_dtype(torch.float64)

    v1_results = test_v1_full_covariance()
    v2_results = test_v2_tiling_sweep()

    combined: List[Dict[str, Any]] = []
    combined.extend(v1_results)
    combined.extend(v2_results)

    if combined:
        df = pd.DataFrame(combined)
        out = os.path.join(os.path.dirname(__file__), "v1_v2_performance_results.csv")
        df.to_csv(out, index=False)
        print(f"\n✓ Results saved to: {out}")

        # Compute v1 baseline (avg cov time per call) by D
        df_v1 = df[df["Implementation"] == "v1"]
        v1_baseline: Dict[int, float] = {}
        for D in sorted(df_v1["D"].unique()):
            v1_baseline[int(D)] = float(df_v1[df_v1["D"] == D]["Avg Cov Time (ms)"].mean())

        print("\n" + "=" * 80)
        print("SUMMARY: Avg Covariance Time (ms) vs Tiling Size for each D")
        print("=" * 80)

        df_v2 = df[df["Implementation"] == "v2"]
        for D in sorted(df_v2["D"].unique()):
            D_int = int(D)
            df_d = df_v2[df_v2["D"] == D]
            base = v1_baseline.get(D_int, float("nan"))
            print(f"\nD={D_int}:")
            print(f"  v1 baseline: {base:.3f} ms")
            print(f"\n  B (tiling_size) | Avg Cov Time (ms) | Ratio to v1")
            print(f"  " + "-" * 50)
            for _, row in df_d.sort_values("Tiling Size").iterrows():
                B = int(row["Tiling Size"])
                avg_ms = float(row["Avg Cov Time (ms)"])
                ratio = avg_ms / base if base == base else float("nan")  # base==base checks not-NaN
                print(f"  {B:4d}             | {avg_ms:17.3f} | {ratio:7.3f}x")

        print("\n" + "=" * 80)
        print("SUMMARY: Ratio to v1 (lower is faster)")
        print("=" * 80)
        for D_int, base in v1_baseline.items():
            df_d = df_v2[df_v2["D"] == D_int]
            if len(df_d) == 0:
                continue
            print(f"\nD={D_int}:")
            for _, row in df_d.sort_values("Tiling Size").iterrows():
                B = int(row["Tiling Size"])
                avg_ms = float(row["Avg Cov Time (ms)"])
                ratio = avg_ms / base
                speedup = "faster" if ratio < 1.0 else "slower"
                pct_diff = abs(ratio - 1.0) * 100.0
                print(f"  B={B:3d}: {ratio:.3f}x ({pct_diff:.1f}% {speedup})")


if __name__ == "__main__":
    main()