#!/usr/bin/env python3
"""Benchmark covariance update runtime for full covariance matrices.

This script measures the covariance update computation runtime (part of M-step) for
full covariance matrices. It tests scaling in two dimensions:
1. vs. N (sample size) with fixed K and D
2. vs. D (dimensionality) with fixed K and N

The full covariance update is:
    diff = X.unsqueeze(1) - means.unsqueeze(0)  # (N, K, D)
    cov_sum = torch.einsum('nk,nkd,nke->kde', resp, diff, diff)  # (K, D, D)
    new_cov = cov_sum / nk.unsqueeze(1).unsqueeze(2) + reg * eye
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

# Output directories
RESULTS_DIR = Path("results/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("results")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def compute_covariance_update(
    X: torch.Tensor,
    means: torch.Tensor,
    log_resp: torch.Tensor,
    reg_covar: float = 1e-6,
) -> torch.Tensor:
    """Compute only the full covariance update from responsibilities.
    
    Args:
        X: Data matrix (N, D)
        means: Current means (K, D)
        log_resp: Log responsibilities (N, K)
        reg_covar: Regularization term
    
    Returns:
        new_cov: Updated full covariances (K, D, D)
    """
    N, D = X.shape
    K, D2 = means.shape
    assert D == D2
    
    resp = log_resp.exp()  # (N, K)
    nk = resp.sum(dim=0) + _nk_eps(resp.dtype)  # (K,)
    
    diff = X.unsqueeze(1) - means.unsqueeze(0)  # (N, K, D)
    
    # Compute full covariance for all components in one einsum operation
    cov_sum = torch.einsum('nk,nkd,nke->kde', resp, diff, diff)  # (K, D, D)
    new_cov = cov_sum / nk.unsqueeze(1).unsqueeze(2)  # (K, D, D)
    
    eye = torch.eye(D, device=X.device, dtype=X.dtype)
    new_cov = new_cov + reg_covar * eye.unsqueeze(0)
    
    return new_cov


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
    """Generate synthetic test data for covariance update benchmarking."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    X = torch.randn(N, D, device=device, dtype=dtype)
    means = torch.randn(K, D, device=device, dtype=dtype)
    
    # Generate log responsibilities (output from E-step)
    log_resp = torch.randn(N, K, device=device, dtype=dtype)
    log_resp = torch.log_softmax(log_resp, dim=1)
    
    return {
        "X": X,
        "means": means,
        "log_resp": log_resp,
    }


def benchmark_covariance_vs_n(
    K: int = 5,
    D: int = 50,
    N_values: list = None,
    device: str = "cuda",
    n_runs: int = 20,
):
    """Benchmark covariance update runtime vs. N for fixed K and D.
    
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
    print(f"BENCHMARK: Covariance Update Runtime vs. N (Full Covariance)")
    print(f"Configuration: K={K}, D={D}, device={device}")
    print("=" * 80)
    print()
    
    results = []
    
    for N in N_values:
        print(f"Testing N={N}...", end=" ", flush=True)
        
        try:
            # Generate test data
            data = generate_test_data(N, D, K, device)
            
            # Time covariance update
            mean_time, std_time = timer(
                compute_covariance_update,
                data["X"],
                data["means"],
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
        except RuntimeError as e:
            print(f"SKIPPED (OOM or error): {e}")
            continue
    
    df = pd.DataFrame(results)
    
    print()
    print("Summary:")
    print(df.to_string(index=False))
    print()
    
    return df


def benchmark_covariance_vs_d(
    N: int = 5000,
    K: int = 5,
    D_values: list = None,
    device: str = "cuda",
    n_runs: int = 20,
):
    """Benchmark covariance update runtime vs. D for fixed N and K.
    
    Args:
        N: Number of samples (fixed)
        K: Number of clusters (fixed)
        D_values: List of D values to test
        device: "cuda" (GPU only)
        n_runs: Number of timing runs per configuration
    """
    if D_values is None:
        D_values = [3, 5, 10, 50, 100, 500]
    
    print("=" * 80)
    print(f"BENCHMARK: Covariance Update Runtime vs. D (Full Covariance)")
    print(f"Configuration: N={N}, K={K}, device={device}")
    print("=" * 80)
    print()
    
    results = []
    
    for D in D_values:
        print(f"Testing D={D}...", end=" ", flush=True)
        
        try:
            # Generate test data
            data = generate_test_data(N, D, K, device)
            
            # Time covariance update
            mean_time, std_time = timer(
                compute_covariance_update,
                data["X"],
                data["means"],
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
        except RuntimeError as e:
            print(f"SKIPPED (OOM or error): {e}")
            continue
    
    df = pd.DataFrame(results)
    
    print()
    print("Summary:")
    print(df.to_string(index=False))
    print()
    
    return df


def plot_results_vs_n(df: pd.DataFrame, K: int, D: int, device: str):
    """Create plots of covariance update runtime vs. N."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    N = df["N"].values
    mean_time = df["Mean Time (ms)"].values
    std_time = df["Std Time (ms)"].values
    
    # Plot 1: Absolute runtime
    ax1.errorbar(N, mean_time, yerr=std_time, marker='o', capsize=5, capthick=2)
    ax1.set_xlabel("N (Number of Samples)", fontsize=12)
    ax1.set_ylabel("Covariance Update Runtime (ms)", fontsize=12)
    ax1.set_title(f"Covariance Update Runtime vs. N (Full)\n(K={K}, D={D}, {device})", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Time per sample
    time_per_sample = df["Time per Sample (μs)"].values
    ax2.plot(N, time_per_sample, marker='s', linewidth=2)
    ax2.set_xlabel("N (Number of Samples)", fontsize=12)
    ax2.set_ylabel("Time per Sample (μs)", fontsize=12)
    ax2.set_title(f"Covariance Update Time per Sample vs. N (Full)\n(K={K}, D={D}, {device})", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.axhline(y=time_per_sample.mean(), color='r', linestyle='--', 
                label=f'Mean: {time_per_sample.mean():.2f} μs', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    filename = f"cov_update_vs_n_K{K}_D{D}_{device}.png"
    filepath = RESULTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    plt.close()


def plot_results_vs_d(df: pd.DataFrame, N: int, K: int, device: str):
    """Create plots of covariance update runtime vs. D."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    D = df["D"].values
    mean_time = df["Mean Time (ms)"].values
    std_time = df["Std Time (ms)"].values
    
    # Plot 1: Absolute runtime
    ax1.errorbar(D, mean_time, yerr=std_time, marker='o', capsize=5, capthick=2)
    ax1.set_xlabel("D (Dimensionality)", fontsize=12)
    ax1.set_ylabel("Covariance Update Runtime (ms)", fontsize=12)
    ax1.set_title(f"Covariance Update Runtime vs. D (Full)\n(N={N}, K={K}, {device})", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Time per sample (should increase with D²)
    time_per_sample = df["Time per Sample (μs)"].values
    ax2.plot(D, time_per_sample, marker='s', linewidth=2)
    ax2.set_xlabel("D (Dimensionality)", fontsize=12)
    ax2.set_ylabel("Time per Sample (μs)", fontsize=12)
    ax2.set_title(f"Covariance Update Time per Sample vs. D (Full)\n(N={N}, K={K}, {device})", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save figure
    filename = f"cov_update_vs_d_N{N}_K{K}_{device}.png"
    filepath = RESULTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    plt.close()


def run_all_benchmarks():
    """Run both N and D scaling benchmarks."""
    # Force GPU usage - will fail if not available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
    device = "cuda"
    
    # Benchmark 1: vs N (fixed K=5, D=50)
    print("\n" + "=" * 80)
    print("PART 1: Covariance Update vs. N")
    print("=" * 80 + "\n")
    
    K = 5
    D = 50
    N_values = [100, 1000, 10000, 100000, 1000000]
    
    df_n = benchmark_covariance_vs_n(
        K=K,
        D=D,
        N_values=N_values,
        device=device,
        n_runs=20,
    )
    
    if len(df_n) > 0:
        plot_results_vs_n(df_n, K, D, device)
        csv_path = DATA_DIR / f"cov_update_vs_n_K{K}_D{D}_{device}.csv"
        df_n.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
    
    # Benchmark 2: vs D (fixed N=5000, K=5)
    print("\n" + "=" * 80)
    print("PART 2: Covariance Update vs. D")
    print("=" * 80 + "\n")
    
    N = 5000
    K = 5
    D_values = [3, 5, 10, 50, 100, 500]
    
    df_d = benchmark_covariance_vs_d(
        N=N,
        K=K,
        D_values=D_values,
        device=device,
        n_runs=20,
    )
    
    if len(df_d) > 0:
        plot_results_vs_d(df_d, N, K, device)
        csv_path = DATA_DIR / f"cov_update_vs_d_N{N}_K{K}_{device}.csv"
        df_d.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
    
    return df_n, df_d


if __name__ == "__main__":
    print("=" * 80)
    print("COVARIANCE UPDATE BENCHMARK: Full Covariance Matrices")
    print("=" * 80)
    print()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        print("Please run on a machine with GPU support.")
        sys.exit(1)
    
    device = "cuda"
    print(f"Using device: {device}")
    print()
    
    # Run all benchmarks
    df_n, df_d = run_all_benchmarks()
    
    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nResults and plots saved to: {RESULTS_DIR}")
