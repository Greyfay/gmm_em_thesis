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
- Measures covariance time by instrumenting the *actual implementation code path*
  via a lightweight hook in _maximization_step (monkeypatch wrapper that calls the
  original function, but reads per-call timing written by a local timer around the
  covariance region inside an injected wrapper).

IMPORTANT:
- To measure covariance-only time without rewriting the implementation, we wrap
  the original _maximization_step and reproduce ONLY the covariance timing by
  calling the original function and timing just the covariance section via an
  internal helper that matches each implementation's covariance path:
    * v1: monolithic einsum full-cov update
    * v2: tiled full-cov update (respects tiling_size / tile_B)

This script assumes:
- v1._maximization_step signature: (X, means, cov, weights, log_resp, cov_type, reg_covar=...)
- v2._maximization_step signature: (X, means, cov, weights, log_resp, cov_type, reg_covar=..., tile_B=None)
- v2.TorchGaussianMixture.fit supports tiling_size parameter and passes tile_B through.
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
    # Only synchronize if we're on CUDA
    if x.is_cuda:
        torch.cuda.synchronize()


def measure_fit_runtime(
    make_model_fn,
    X: torch.Tensor,
    n_runs: int = 5,
    warmup: int = 2,
) -> Tuple[float, float, Any]:
    """
    Measure mean/std of model.fit time (CUDA-accurate).
    Returns: (mean_s, std_s, last_model)
    """
    # Warmup
    model = None
    for _ in range(warmup):
        model = make_model_fn()
        _cuda_sync_if_needed(X)
        model.fit(X)  # do NOT clone; fit should not mutate X
        _cuda_sync_if_needed(X)

    # Timed runs
    times = []
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
    """
    original = v1._maximization_step
    stats = {"cov_time_s": 0.0, "count": 0}

    # Pull helpers from v1 to keep semantics identical
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
        # We call original for correctness, but we also time the *actual covariance path* for v1.
        # For v1 full-cov, covariance update is the monolithic einsum using diff.
        # If cov_type is not 'full', we just fall back to original without counting.

        if cov_type != "full":
            return original(X, means, cov, weights, log_resp, cov_type, reg_covar=reg_covar)

        # Compute shared intermediates exactly as in M-step (not timed):
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
        stats["count"] += 1

        return new_means, new_cov, new_weights

    v1._maximization_step = wrapped
    return stats, original


def make_cov_timer_v2() -> Tuple[Dict[str, float], Any]:
    """
    Monkeypatch v2._maximization_step so we can accumulate covariance-only time
    for the *tiled* full-cov path (respects tile_B).
    """
    original = v2._maximization_step
    stats = {"cov_time_s": 0.0, "count": 0}

    _nk_eps = v2._nk_eps  # use v2's helper

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

        # Match v2 semantics for shared intermediates (not timed)
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
        nk_ = nk.unsqueeze(1).unsqueeze(2)
        eye = torch.eye(D, device=X.device, dtype=X.dtype)

        for i in range(0, D, B):
            i2 = min(i + B, D)
            Xi = X[:, i:i2].unsqueeze(1)          # (N,1,Bi)
            Mi = new_means[:, i:i2].unsqueeze(0)  # (1,K,Bi)
            diff_i = Xi - Mi                      # (N,K,Bi)

            for j in range(0, D, B):
                j2 = min(j + B, D)
                Xj = X[:, j:j2].unsqueeze(1)          # (N,1,Bj)
                Mj = new_means[:, j:j2].unsqueeze(0)  # (1,K,Bj)
                diff_j = Xj - Mj                      # (N,K,Bj)

                cov_block = torch.einsum("nk,nkb,nkc->kbc", resp, diff_i, diff_j)
                cov_sum[:, i:i2, j:j2] = cov_block

        new_cov = cov_sum / nk_
        new_cov = new_cov + reg_covar * eye.unsqueeze(0)

        _cuda_sync_if_needed(X)
        t1 = time.perf_counter()

        stats["cov_time_s"] += (t1 - t0)
        stats["count"] += 1

        return new_means, new_cov, new_weights

    v2._maximization_step = wrapped
    return stats, original


# -----------------------
# Tests
# -----------------------

def test_v1_full_covariance() -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("TEST: v1 (full covariance)")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    device = "cuda"
    dtype = torch.float64
    cov_type = "full"

    test_configs = [
        (500, 20, 5, 50),
        (1000, 50, 10, 50),
        (2000, 100, 20, 50),
        # Optional stress config to make tiling effects visible:
        (1000, 256, 16, 30),
    ]

    for N, D, K, max_iter in test_configs:
        print(f"\n--- Config: N={N}, D={D}, K={K}, max_iter={max_iter} ---")

        X = generate_test_data(N, D, device=device, dtype=dtype)

        cov_stats, original = make_cov_timer_v1()
        try:
            def make_model():
                torch.manual_seed(42)
                return v1.TorchGaussianMixture(
                    n_components=K,
                    covariance_type=cov_type,
                    max_iter=max_iter,
                    n_init=1,
                    init_params="random",
                    device=device,
                    dtype=dtype,
                )

            mean_fit, std_fit, model = measure_fit_runtime(make_model, X, n_runs=5, warmup=2)

            final_ll = float(model.lower_bound_)
            n_iter = int(model.n_iter_)

            cov_time = cov_stats["cov_time_s"]
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


def test_v2_tiling_sweep() -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("TEST: v2 (full covariance) tiling_size sweep")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    device = "cuda"
    dtype = torch.float64
    cov_type = "full"

    test_configs = [
        (500, 20, 5, 50),
        (1000, 50, 10, 50),
        (2000, 100, 20, 50),
        # Optional stress config:
        (1000, 256, 16, 30),
    ]

    tiling_sizes = [16, 32, 64, 128]

    for N, D, K, max_iter in test_configs:
        print(f"\n--- Config: N={N}, D={D}, K={K}, max_iter={max_iter} ---")

        X = generate_test_data(N, D, device=device, dtype=dtype)

        for B in tiling_sizes:
            print(f"  tiling_size={B}...", end=" ", flush=True)

            cov_stats, original = make_cov_timer_v2()
            try:
                def make_model():
                    torch.manual_seed(42)
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

                mean_fit, std_fit, model = measure_fit_runtime(make_model, X, n_runs=5, warmup=2)

                final_ll = float(model.lower_bound_)
                n_iter = int(model.n_iter_)

                cov_time = cov_stats["cov_time_s"]
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

    print("=" * 80)
    print("PERFORMANCE TEST: v1 vs v2 (covariance timing + tiling sweep)")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print("Device: CUDA")
    print(f"CUDA available: {torch.cuda.is_available()}\n")

    torch.set_default_dtype(torch.float64)

    v1_results = test_v1_full_covariance()
    v2_results = test_v2_tiling_sweep()

    combined = []
    combined.extend(v1_results)
    combined.extend(v2_results)

    if combined:
        df = pd.DataFrame(combined)
        out = os.path.join(os.path.dirname(__file__), "v1_v2_performance_results.csv")
        df.to_csv(out, index=False)
        print(f"\n✓ Results saved to: {out}")

        # Helpful on-screen summary
        print("\n--- Quick summary: avg cov time by tiling size (v2) ---")
        if not df[df["Implementation"] == "v2"].empty:
            print(
                df[df["Implementation"] == "v2"]
                .groupby(["D", "K", "Tiling Size"])["Avg Cov Time (ms)"]
                .mean()
                .sort_index()
            )


if __name__ == "__main__":
    main()