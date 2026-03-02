#!/usr/bin/env python3
"""Benchmark E-Step runtime vs. N (data size) for fixed K and D.

This script measures the E-Step (_expectation_step_precchol) runtime of the v1
implementation as a function of data size N, keeping K (clusters) and D (dimensions)
fixed. This helps understand the O(N) scaling behavior of the E-step.
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
    cov_type: str = "full",
    device: str = "cpu",
    dtype=torch.float64,
):
    """Generate synthetic test data for E-step benchmarking."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    X = torch.randn(N, D, device=device, dtype=dtype)
    means = torch.randn(K, D, device=device, dtype=dtype)
    weights = torch.softmax(torch.randn(K, device=device, dtype=dtype), dim=0)
    
    # Generate covariances based on type
    if cov_type == "diag":
        cov = torch.rand(K, D, device=device, dtype=dtype) + 0.5
    elif cov_type == "spherical":
        cov = torch.rand(K, device=device, dtype=dtype) + 0.5
    elif cov_type == "tied":
        A = torch.randn(D, D, device=device, dtype=dtype)
        cov = A @ A.T + torch.eye(D, device=device, dtype=dtype)
    else:  # full
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
    prec_chol = _compute_precisions_cholesky(cov, cov_type)
    
    return {
        "X": X,
        "means": means,
        "weights": weights,
        "cov": cov,
        "prec_chol": prec_chol,
    }


def benchmark_estep_vs_n(
    K: int = 5,
    D: int = 50,
    cov_type: str = "full",
    N_values: list = None,
    device: str = "cuda",
    n_runs: int = 10,
):
    """Benchmark E-step runtime vs. N for fixed K and D.
    
    Args:
        K: Number of clusters (fixed)
        D: Number of dimensions (fixed)
        cov_type: Covariance type to test
        N_values: List of N values to test
        device: "cpu" or "cuda"
        n_runs: Number of timing runs per configuration
    """
    if N_values is None:
        N_values = [100, 1000, 10000, 100000, 1000000]
    
    print("=" * 80)
    print(f"BENCHMARK: E-Step Runtime vs. N")
    print(f"Configuration: K={K}, D={D}, cov_type={cov_type}, device={device}")
    print("=" * 80)
    print()
    
    results = []
    
    for N in N_values:
        print(f"Testing N={N}...", end=" ", flush=True)
        
        # Generate test data
        data = generate_test_data(N, D, K, cov_type, device)
        
        # Time E-step
        mean_time, std_time = timer(
            _expectation_step_precchol,
            data["X"],
            data["means"],
            data["prec_chol"],
            data["weights"],
            cov_type,
            n_runs=n_runs,
        )
        
        print(f"{mean_time:.3f} ± {std_time:.3f} ms")
        
        results.append({
            "N": N,
            "D": D,
            "K": K,
            "Covariance Type": cov_type,
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


def plot_results(df: pd.DataFrame, cov_type: str, K: int, D: int, device: str):
    """Create plots of E-step runtime vs. N."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    N = df["N"].values
    mean_time = df["Mean Time (ms)"].values
    std_time = df["Std Time (ms)"].values
    
    # Plot 1: Absolute runtime
    ax1.errorbar(N, mean_time, yerr=std_time, marker='o', capsize=5, capthick=2)
    ax1.set_xlabel("N (Number of Samples)", fontsize=12)
    ax1.set_ylabel("E-Step Runtime (ms)", fontsize=12)
    ax1.set_title(f"E-Step Runtime vs. N\n(K={K}, D={D}, {cov_type}, {device})", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Time per sample (should be relatively constant for O(N) scaling)
    time_per_sample = df["Time per Sample (μs)"].values
    ax2.plot(N, time_per_sample, marker='s', linewidth=2)
    ax2.set_xlabel("N (Number of Samples)", fontsize=12)
    ax2.set_ylabel("Time per Sample (μs)", fontsize=12)
    ax2.set_title(f"E-Step Time per Sample vs. N\n(K={K}, D={D}, {cov_type}, {device})", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.axhline(y=time_per_sample.mean(), color='r', linestyle='--', 
                label=f'Mean: {time_per_sample.mean():.2f} μs', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    filename = f"estep_vs_n_K{K}_D{D}_{cov_type}_{device}.png"
    filepath = RESULTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    plt.close()


def run_all_benchmarks():
    """Run benchmarks for all covariance types."""
    # Configuration
    K = 5
    D = 50
    
    # Force GPU usage - will fail if not available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
    device = "cuda"
    
    # Test N values in factors of 10
    N_values = [100, 1000, 10000, 100000, 1000000]
    
    all_results = []
    
    for cov_type in ["diag", "spherical", "tied", "full"]:
        print("\n" + "=" * 80)
        print(f"Testing covariance type: {cov_type}")
        print("=" * 80 + "\n")
        
        df = benchmark_estep_vs_n(
            K=K,
            D=D,
            cov_type=cov_type,
            N_values=N_values,
            device=device,
            n_runs=20,  # More runs on GPU for stability
        )
        
        all_results.append(df)
        
        # Create plots
        plot_results(df, cov_type, K, D, device)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV
    csv_path = DATA_DIR / f"estep_vs_n_K{K}_D{D}_{device}.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create comparison plot
    create_comparison_plot(all_results, K, D, device)
    
    return combined_df


def create_comparison_plot(dfs: list, K: int, D: int, device: str):
    """Create a comparison plot for all covariance types."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    cov_types = ["diag", "spherical", "tied", "full"]
    colors = ['blue', 'green', 'orange', 'red']
    markers = ['o', 's', '^', 'D']
    
    for df, cov_type, color, marker in zip(dfs, cov_types, colors, markers):
        N = df["N"].values
        mean_time = df["Mean Time (ms)"].values
        std_time = df["Std Time (ms)"].values
        time_per_sample = df["Time per Sample (μs)"].values
        
        # Plot 1: Absolute runtime
        ax1.errorbar(N, mean_time, yerr=std_time, marker=marker, label=cov_type,
                    capsize=4, capthick=1.5, linewidth=2, color=color, alpha=0.8)
        
        # Plot 2: Time per sample
        ax2.plot(N, time_per_sample, marker=marker, label=cov_type,
                linewidth=2, color=color, alpha=0.8)
    
    # Configure plot 1
    ax1.set_xlabel("N (Number of Samples)", fontsize=13)
    ax1.set_ylabel("E-Step Runtime (ms)", fontsize=13)
    ax1.set_title(f"E-Step Runtime vs. N - All Covariance Types\n(K={K}, D={D}, {device})", 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='best', fontsize=11)
    
    # Configure plot 2
    ax2.set_xlabel("N (Number of Samples)", fontsize=13)
    ax2.set_ylabel("Time per Sample (μs)", fontsize=13)
    ax2.set_title(f"E-Step Time per Sample vs. N - All Covariance Types\n(K={K}, D={D}, {device})", 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"estep_vs_n_comparison_K{K}_D{D}_{device}.png"
    filepath = RESULTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {filepath}")
    plt.close()


if __name__ == "__main__":
    print("=" * 80)
    print("E-STEP BENCHMARK: Runtime vs. N (Sample Size)")
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
    results = run_all_benchmarks()
    
    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nResults and plots saved to: {RESULTS_DIR}")
