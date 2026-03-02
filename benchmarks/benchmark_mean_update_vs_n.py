#!/usr/bin/env python3
"""Benchmark mean update runtime vs. N (data size) for fixed K and D.

This script measures the mean update computation runtime (part of M-step) of the v1
implementation as a function of data size N, keeping K (clusters) and D (dimensions)
fixed. This helps understand the O(N) scaling behavior of the mean update.

The mean update is: new_means = (resp.T @ X) / nk
where resp is the responsibility matrix (N, K) and X is the data (N, D).
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation._v1 import (
    TorchGaussianMixture,
    _nk_eps,
)


def compute_mean_update(X: torch.Tensor, log_resp: torch.Tensor) -> torch.Tensor:
    """Compute only the mean update from responsibilities.
    
    Args:
        X: Data matrix (N, D)
        log_resp: Log responsibilities (N, K)
    
    Returns:
        new_means: Updated means (K, D)
    """
    resp = log_resp.exp()  # (N, K)
    nk = resp.sum(dim=0) + _nk_eps(resp.dtype)  # (K,)
    new_means = (resp.T @ X) / nk.unsqueeze(1)  # (K, D)
    return new_means

# Output directories
RESULTS_DIR = Path("results/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("results")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def timer(func, *args, n_runs: int = 10, warmup: int = 2, **kwargs) -> Tuple[float, float]:
    """Time a function with warmup runs.
    
    Returns:
        (mean_time, std_time) in milliseconds
    """
    # Warmup runs
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)


def generate_test_data(
    N: int,
    D: int,
    K: int,
    device: str = "cuda",
    dtype=torch.float64,
):
    """Generate synthetic test data for mean update benchmarking."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    X = torch.randn(N, D, device=device, dtype=dtype)
    
    # Generate log responsibilities (output from E-step)
    log_resp = torch.randn(N, K, device=device, dtype=dtype)
    log_resp = torch.log_softmax(log_resp, dim=1)
    
    return {
        "X": X,
        "log_resp": log_resp,
    }


def benchmark_mean_update_vs_n(
    K: int = 5,
    D: int = 50,
    N_values: list = None,
    device: str = "cuda",
    n_runs: int = 10,
):
    """Benchmark mean update runtime vs. N for fixed K and D.
    
    Args:
        K: Number of clusters (fixed)
        D: Number of dimensions (fixed)
        N_values: List of N values to test
        device: "cuda" (GPU only)
        n_runs: Number of timing runs per configuration
    """
    if N_values is None:
        N_values = [100, 1000, 10000, 100000, 1000000]
    
    print("=" * 80)
    print(f"BENCHMARK: Mean Update Runtime vs. N")
    print(f"Configuration: K={K}, D={D}, device={device}")
    print("=" * 80)
    print()
    
    results = []
    
    for N in N_values:
        print(f"Testing N={N}...", end=" ", flush=True)
        
        # Generate test data
        data = generate_test_data(N, D, K, device)
        
        # Time mean update
        mean_time, std_time = timer(
            compute_mean_update,
            data["X"],
            data["log_resp"],
            n_runs=n_runs,
        )
        
        print(f"{mean_time:.3f} ± {std_time:.3f} ms")
        
        results.append({
            "N": N,
            "D": D,
            "K": K,
            "Device": device,
            "Mean Time (ms)": mean_time,
            "Std Time (ms)": std_time,
            "Time per Sample (μs)": (mean_time * 1000) / N,  # microseconds per sample
        })
    
    df = pd.DataFrame(results)
    
    print()
    print("Summary:")
    print(df.to_string(index=False))
    print()
    
    return df


def plot_results(df: pd.DataFrame, K: int, D: int, device: str):
    """Create plots of mean update runtime vs. N."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    N = df["N"].values
    mean_time = df["Mean Time (ms)"].values
    std_time = df["Std Time (ms)"].values
    
    # Plot 1: Absolute runtime
    ax1.errorbar(N, mean_time, yerr=std_time, marker='o', capsize=5, capthick=2)
    ax1.set_xlabel("N (Number of Samples)", fontsize=12)
    ax1.set_ylabel("Mean Update Runtime (ms)", fontsize=12)
    ax1.set_title(f"Mean Update Runtime vs. N\n(K={K}, D={D}, {device})", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Time per sample (should be relatively constant for O(N) scaling)
    time_per_sample = df["Time per Sample (μs)"].values
    ax2.plot(N, time_per_sample, marker='s', linewidth=2)
    ax2.set_xlabel("N (Number of Samples)", fontsize=12)
    ax2.set_ylabel("Time per Sample (μs)", fontsize=12)
    ax2.set_title(f"Mean Update Time per Sample vs. N\n(K={K}, D={D}, {device})", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.axhline(y=time_per_sample.mean(), color='r', linestyle='--', 
                label=f'Mean: {time_per_sample.mean():.2f} μs', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    filename = f"mean_update_vs_n_K{K}_D{D}_{device}.png"
    filepath = RESULTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    plt.close()


def run_benchmark():
    """Run benchmark for mean update vs N."""
    # Configuration
    K = 5
    D = 50
    
    # Force GPU usage - will fail if not available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
    device = "cuda"
    
    # Test N values in factors of 10
    N_values = [100, 1000, 10000, 100000, 1000000]
    
    print("\n" + "=" * 80)
    print("Testing Mean Update vs. N")
    print("=" * 80 + "\n")
    
    df = benchmark_mean_update_vs_n(
        K=K,
        D=D,
        N_values=N_values,
        device=device,
        n_runs=20,  # More runs on GPU for stability
    )
    
    # Create plots
    plot_results(df, K, D, device)
    
    # Save to CSV
    csv_path = DATA_DIR / f"mean_update_vs_n_K{K}_D{D}_{device}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    return df


if __name__ == "__main__":
    print("=" * 80)
    print("MEAN UPDATE BENCHMARK: Runtime vs. N (Sample Size)")
    print("=" * 80)
    print()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        print("Please run on a machine with GPU support.")
        sys.exit(1)
    
    device = "cuda"
    print(f"Using device: {device}")
    print()
    
    # Run benchmark
    results = run_benchmark()
    
    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nResults and plots saved to: {RESULTS_DIR}")
