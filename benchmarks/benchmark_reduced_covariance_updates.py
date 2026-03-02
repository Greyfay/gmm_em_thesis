#!/usr/bin/env python3
"""Benchmark comparing likelihood progression with reduced covariance update frequency.

This script compares the likelihood per iteration between:
- _v1.py: Standard EM (covariance updated every iteration)
- _v2.py: Reduced-frequency covariance updates (frequency = 2, 5, 10, 20)

The goal is to understand how reducing covariance update frequency affects:
1. Convergence speed (iterations to converge)
2. Likelihood progression over iterations
3. Final likelihood achieved
4. Total runtime
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add parent directory to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Import implementations
from implementation._v1 import TorchGaussianMixture as TorchGaussianMixture_v1
from implementation._v2 import TorchGaussianMixture as TorchGaussianMixture_v2

# Output directories
RESULTS_DIR = Path("results/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("results")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Enforce GPU-only execution
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This benchmark requires a GPU to run.\n"
        "Please run on a machine with CUDA-capable GPU."
    )

DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")


def generate_synthetic_gmm_data(
    N: int = 2000,
    D: int = 10,
    K: int = 5,
    seed: int = 42,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Generate synthetic data from a Gaussian mixture model.
    
    Args:
        N: Number of samples
        D: Dimensionality
        K: Number of mixture components
        seed: Random seed
        device: Device to use
        dtype: Data type
    
    Returns:
        X: Generated data (N, D)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = DEVICE if device is None else device
    
    # Generate cluster centers
    means = torch.randn(K, D, device=device, dtype=dtype) * 5
    
    # Generate samples per cluster
    samples_per_cluster = N // K
    X_list = []
    
    for k in range(K):
        # Generate covariance: add some correlation structure
        A = torch.randn(D, D, device=device, dtype=dtype)
        cov = (A @ A.T) / D + torch.eye(D, device=device, dtype=dtype)
        
        # Sample from multivariate normal
        n_k = samples_per_cluster + (N % K if k == 0 else 0)
        mvn = torch.distributions.MultivariateNormal(means[k], cov)
        X_k = mvn.sample((n_k,))
        X_list.append(X_k)
    
    X = torch.cat(X_list, dim=0)
    # Shuffle
    perm = torch.randperm(X.shape[0], device=device)
    X = X[perm]
    
    return X


def run_single_experiment(
    X: torch.Tensor,
    n_components: int,
    covariance_type: str,
    max_iter: int,
    covariance_update_frequency: int,
    model_class,
    seed: int = 42,
) -> Dict:
    """Run a single GMM fitting experiment and return results.
    
    Args:
        X: Data
        n_components: Number of components
        covariance_type: Type of covariance
        max_iter: Maximum iterations
        covariance_update_frequency: How often to update covariance
        model_class: TorchGaussianMixture class (_v1 or _v2)
        seed: Random seed
    
    Returns:
        Dictionary with results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    if model_class == TorchGaussianMixture_v1:
        model = model_class(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            tol=1e-6,  # Lower tolerance to see more iterations
            n_init=1,
            init_params="kmeans",
            device=X.device,
            dtype=X.dtype,
        )
    else:  # _v2
        model = model_class(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            tol=1e-6,
            n_init=1,
            init_params="kmeans",
            covariance_update_frequency=covariance_update_frequency,
            device=X.device,
            dtype=X.dtype,
        )
    
    # Time the fitting
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    model.fit(X)
    
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    runtime = (end_time - start_time) * 1000  # Convert to ms
    
    return {
        "model": model,
        "lower_bounds": model.lower_bounds_,
        "n_iter": model.n_iter_,
        "converged": model.converged_,
        "final_lower_bound": model.lower_bound_,
        "runtime_ms": runtime,
        "covariance_updates": getattr(model, "covariance_updates_", model.n_iter_),
    }


def benchmark_likelihood_progression(
    N: int = 2000,
    D: int = 10,
    K: int = 5,
    max_iter: int = 100,
    covariance_type: str = "full",
    seed: int = 42,
) -> pd.DataFrame:
    """Benchmark likelihood progression for different covariance update frequencies.
    
    Args:
        N: Number of samples
        D: Dimensionality
        K: Number of components
        max_iter: Maximum iterations
        covariance_type: Type of covariance
        seed: Random seed
    
    Returns:
        DataFrame with results
    """
    print("="*80)
    print(f"BENCHMARK: Likelihood Progression (N={N}, D={D}, K={K}, cov_type={covariance_type})")
    print("="*80)
    
    # Generate data
    X = generate_synthetic_gmm_data(N, D, K, seed=seed)
    
    # Test configurations
    configs = [
        ("v1 (baseline)", TorchGaussianMixture_v1, 1),
        ("v2 (freq=2)", TorchGaussianMixture_v2, 2),
        ("v2 (freq=5)", TorchGaussianMixture_v2, 5),
        ("v2 (freq=10)", TorchGaussianMixture_v2, 10),
        ("v2 (freq=20)", TorchGaussianMixture_v2, 20),
    ]
    
    results = []
    all_lower_bounds = {}
    
    for name, model_class, freq in configs:
        print(f"\nRunning: {name}")
        result = run_single_experiment(
            X, K, covariance_type, max_iter, freq, model_class, seed=seed
        )
        
        print(f"  Iterations: {result['n_iter']}")
        print(f"  Converged: {result['converged']}")
        print(f"  Final log-likelihood: {result['final_lower_bound']:.4f}")
        print(f"  Runtime: {result['runtime_ms']:.2f} ms")
        print(f"  Covariance updates: {result['covariance_updates']}/{result['n_iter']}")
        
        all_lower_bounds[name] = result['lower_bounds']
        
        results.append({
            "Configuration": name,
            "Model": "v1" if model_class == TorchGaussianMixture_v1 else "v2",
            "Cov Update Freq": freq,
            "N": N,
            "D": D,
            "K": K,
            "Cov Type": covariance_type,
            "Iterations": result['n_iter'],
            "Converged": result['converged'],
            "Final Log-Likelihood": result['final_lower_bound'],
            "Runtime (ms)": result['runtime_ms'],
            "Covariance Updates": result['covariance_updates'],
            "Runtime per Iteration (ms)": result['runtime_ms'] / result['n_iter'],
        })
    
    df = pd.DataFrame(results)
    
    # Plot likelihood progression
    plot_likelihood_curves(
        all_lower_bounds,
        title=f"Likelihood Progression (N={N}, D={D}, K={K}, cov={covariance_type})",
        filename=f"likelihood_progression_N{N}_D{D}_K{K}_{covariance_type}.png"
    )
    
    return df


def plot_likelihood_curves(
    lower_bounds_dict: Dict[str, List[float]],
    title: str = "Likelihood Progression",
    filename: str = "likelihood_progression.png",
) -> None:
    """Plot likelihood curves for different configurations.
    
    Args:
        lower_bounds_dict: Dictionary mapping config name to list of lower bounds
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '--', '-.', ':', (0, (5, 2, 1, 2))]
    
    for i, (name, lower_bounds) in enumerate(lower_bounds_dict.items()):
        iterations = list(range(1, len(lower_bounds) + 1))
        plt.plot(
            iterations,
            lower_bounds,
            label=name,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=2,
            marker='o' if len(lower_bounds) < 30 else None,
            markersize=4,
            markevery=max(1, len(lower_bounds) // 20),
        )
    
    plt.xlabel("EM Iteration", fontsize=12)
    plt.ylabel("Log-Likelihood", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Plot saved to: {output_path}")
    plt.close()


def plot_convergence_comparison(df: pd.DataFrame, filename: str = "convergence_comparison.png") -> None:
    """Plot comparison of iterations vs runtime for different configurations.
    
    Args:
        df: Results DataFrame
        filename: Output filename
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Iterations to convergence
    ax = axes[0]
    configs = df['Configuration'].values
    iterations = df['Iterations'].values
    colors_map = {'v1 (baseline)': '#1f77b4', 'v2 (freq=2)': '#ff7f0e', 
                  'v2 (freq=5)': '#2ca02c', 'v2 (freq=10)': '#d62728', 
                  'v2 (freq=20)': '#9467bd'}
    colors = [colors_map.get(c, '#888888') for c in configs]
    
    bars = ax.bar(range(len(configs)), iterations, color=colors, alpha=0.8)
    ax.set_xlabel("Configuration", fontsize=11)
    ax.set_ylabel("Iterations", fontsize=11)
    ax.set_title("Iterations to Convergence", fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, iterations)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Runtime comparison
    ax = axes[1]
    runtime = df['Runtime (ms)'].values
    bars = ax.bar(range(len(configs)), runtime, color=colors, alpha=0.8)
    ax.set_xlabel("Configuration", fontsize=11)
    ax.set_ylabel("Runtime (ms)", fontsize=11)
    ax.set_title("Total Runtime", fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, runtime)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved to: {output_path}")
    plt.close()


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across different configurations."""
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK: Reduced Covariance Update Frequency")
    print("="*80 + "\n")
    
    all_results = []
    
    # Test configurations: (N, D, K, cov_type, max_iter)
    # Only testing full covariance type
    test_configs = [
        (1000, 5, 3, "full", 100),
        (2000, 10, 5, "full", 100),
        (2000, 20, 5, "full", 100),
        (3000, 15, 5, "full", 100),
    ]
    
    for i, (N, D, K, cov_type, max_iter) in enumerate(test_configs):
        print(f"\n{'='*80}")
        print(f"Test {i+1}/{len(test_configs)}")
        print(f"{'='*80}")
        
        df = benchmark_likelihood_progression(
            N=N, D=D, K=K, max_iter=max_iter, covariance_type=cov_type, seed=42+i
        )
        all_results.append(df)
        
        # Plot convergence comparison
        plot_convergence_comparison(
            df, 
            filename=f"convergence_comparison_N{N}_D{D}_K{K}_{cov_type}.png"
        )
        
        # Save CSV
        csv_path = DATA_DIR / f"results_N{N}_D{D}_K{K}_{cov_type}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Results saved to: {csv_path}")
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_path = DATA_DIR / "benchmark_reduced_covariance_updates_all.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\n{'='*80}")
    print(f"All results saved to: {combined_path}")
    print(f"{'='*80}\n")
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("="*80)
    print(combined_df.groupby('Configuration').agg({
        'Iterations': 'mean',
        'Runtime (ms)': 'mean',
        'Final Log-Likelihood': 'mean',
        'Covariance Updates': 'mean',
    }).round(2))
    print("="*80 + "\n")
    
    return combined_df


if __name__ == "__main__":
    print("Starting benchmark: Reduced Covariance Update Frequency (GPU-only)")
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Default dtype: {torch.get_default_dtype()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print()
    
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {DATA_DIR}")
    print(f"Figures saved to: {RESULTS_DIR}")
