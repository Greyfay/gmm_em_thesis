#!/usr/bin/env python3
"""Benchmark E-Step runtime vs. D (dimensionality) for full covariance with fixed N and K.

This script measures the E-Step (_expectation_step_precchol) runtime of the v1
implementation as a function of dimensionality D, keeping N (samples) and K (clusters)
fixed. This helps understand the O(D) or O(D²) scaling behavior of the E-step.
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
    _expectation_step_precchol,
    _compute_precisions_cholesky,
)

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
    """Generate synthetic test data for E-step benchmarking."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    X = torch.randn(N, D, device=device, dtype=dtype)
    means = torch.randn(K, D, device=device, dtype=dtype)
    weights = torch.softmax(torch.randn(K, device=device, dtype=dtype), dim=0)
    
    # Generate full covariances
    cov = torch.stack([
        torch.eye(D, device=device, dtype=dtype) + 
        0.1 * torch.randn(D, D, device=device, dtype=dtype)
        for _ in range(K)
    ])
    # Make positive definite
    for k in range(K):
        A = cov[k]
        cov[k] = A @ A.T + 0.1 * torch.eye(D, device=device, dtype=dtype)
    
    # Compute precision cholesky
    prec_chol = _compute_precisions_cholesky(cov, "full")
    
    return {
        "X": X,
        "means": means,
        "weights": weights,
        "cov": cov,
        "prec_chol": prec_chol,
    }


def benchmark_estep_vs_d(
    N: int = 10000,
    K: int = 5,
    D_values: list = None,
    device: str = "cuda",
    n_runs: int = 20,
):
    """Benchmark E-step runtime vs. D for fixed N and K with full covariance.
    
    Args:
        N: Number of samples (fixed)
        K: Number of clusters (fixed)
        D_values: List of D values to test
        device: "cuda" (GPU only)
        n_runs: Number of timing runs per configuration
    """
    if D_values is None:
        D_values = [10, 100, 1000, 10000, 100000, 1000000]
    
    print("=" * 80)
    print(f"BENCHMARK: E-Step Runtime vs. D (Full Covariance)")
    print(f"Configuration: N={N}, K={K}, device={device}")
    print("=" * 80)
    print()
    
    results = []
    
    for D in D_values:
        print(f"Testing D={D}...", end=" ", flush=True)
        
        try:
            # Generate test data
            data = generate_test_data(N, D, K, device)
            
            # Time E-step
            mean_time, std_time = timer(
                _expectation_step_precchol,
                data["X"],
                data["means"],
                data["prec_chol"],
                data["weights"],
                "full",
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


def plot_results(df: pd.DataFrame, N: int, K: int, device: str):
    """Create plots of E-step runtime vs. D."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    D = df["D"].values
    mean_time = df["Mean Time (ms)"].values
    std_time = df["Std Time (ms)"].values
    
    # Plot 1: Absolute runtime
    ax1.errorbar(D, mean_time, yerr=std_time, marker='o', capsize=5, capthick=2)
    ax1.set_xlabel("D (Dimensionality)", fontsize=12)
    ax1.set_ylabel("E-Step Runtime (ms)", fontsize=12)
    ax1.set_title(f"E-Step Runtime vs. D (Full Covariance)\n(N={N}, K={K}, {device})", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Time per sample (should increase with D)
    time_per_sample = df["Time per Sample (μs)"].values
    ax2.plot(D, time_per_sample, marker='s', linewidth=2)
    ax2.set_xlabel("D (Dimensionality)", fontsize=12)
    ax2.set_ylabel("Time per Sample (μs)", fontsize=12)
    ax2.set_title(f"E-Step Time per Sample vs. D (Full Covariance)\n(N={N}, K={K}, {device})", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save figure
    filename = f"estep_vs_d_N{N}_K{K}_full_{device}.png"
    filepath = RESULTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    plt.close()


def run_benchmark():
    """Run benchmark for E-step vs D with full covariance."""
    # Configuration
    N = 10000
    K = 5
    
    # Force GPU usage - will fail if not available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
    device = "cuda"
    
    # Test D values in factors of 10
    D_values = [10, 100, 1000, 10000, 100000, 1000000]
    
    print("\n" + "=" * 80)
    print("Testing E-Step vs. D (Full Covariance)")
    print("=" * 80 + "\n")
    
    df = benchmark_estep_vs_d(
        N=N,
        K=K,
        D_values=D_values,
        device=device,
        n_runs=20,
    )
    
    # Create plots
    if len(df) > 0:
        plot_results(df, N, K, device)
    
    # Save to CSV
    csv_path = DATA_DIR / f"estep_vs_d_N{N}_K{K}_{device}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    return df


if __name__ == "__main__":
    print("=" * 80)
    print("E-STEP BENCHMARK: Runtime vs. D (Dimensionality)")
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
