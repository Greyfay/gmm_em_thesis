#!/usr/bin/env python3
"""Benchmark comparing loop-based vs parallelized implementations.

This script compares the performance of individual functions between:
- _torch_gmm_em_old.py (loop-based, sklearn-like implementation)
- _torch_gmm_em.py (parallelized/vectorized implementation)

The goal is to quantify the speedup achieved through parallelization.
"""

import sys
import os
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Import old (loop-based) implementation
from implementation import _torch_gmm_em_old as old_impl

# Import new (parallelized) implementation
from implementation import _torch_gmm_em as new_impl


def timer(func: Callable, *args, n_runs: int = 10, warmup: int = 2, **kwargs) -> Tuple[float, float]:
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
    N: int = 1000,
    D: int = 50,
    K: int = 5,
    device: str = "cpu",
    dtype=torch.float64,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic test data for benchmarking."""
    random_seed = np.random.randint(1, 1001)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    X = torch.randn(N, D, device=device, dtype=dtype)
    means = torch.randn(K, D, device=device, dtype=dtype)
    weights = torch.softmax(torch.randn(K, device=device, dtype=dtype), dim=0)
    
    # Generate covariances for different types
    cov_full = torch.stack([
        torch.eye(D, device=device, dtype=dtype) + 0.1 * torch.randn(D, D, device=device, dtype=dtype)
        for _ in range(K)
    ])
    # Make them positive definite
    cov_full = torch.bmm(cov_full, cov_full.transpose(-1, -2))
    
    cov_tied = torch.eye(D, device=device, dtype=dtype) + 0.1 * torch.randn(D, D, device=device, dtype=dtype)
    cov_tied = cov_tied @ cov_tied.T
    
    cov_diag = torch.rand(K, D, device=device, dtype=dtype) + 0.5
    cov_spherical = torch.rand(K, device=device, dtype=dtype) + 0.5
    
    # Generate responsibilities
    log_resp = torch.randn(N, K, device=device, dtype=dtype)
    log_resp = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True)
    
    return {
        "X": X,
        "means": means,
        "weights": weights,
        "cov_full": cov_full,
        "cov_tied": cov_tied,
        "cov_diag": cov_diag,
        "cov_spherical": cov_spherical,
        "log_resp": log_resp,
    }


def benchmark_precisions_cholesky():
    """Benchmark _compute_precisions_cholesky for full covariance."""
    print("\n" + "="*80)
    print("BENCHMARK: _compute_precisions_cholesky (full covariance)")
    print("="*80)
    
    results = []
    for N, D, K in [(500, 20, 5), (1000, 50, 10), (2000, 100, 5)]:
        data = generate_test_data(N, D, K)
        cov = data["cov_full"]
        
        old_time, old_std = timer(old_impl._compute_precisions_cholesky, cov, "full")
        new_time, new_std = timer(new_impl._compute_precisions_cholesky, cov, "full")
        
        speedup = old_time / new_time
        print(f"N={N}, D={D}, K={K}:")
        print(f"  Old: {old_time:.3f} ± {old_std:.3f} ms")
        print(f"  New: {new_time:.3f} ± {new_std:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            "Function": "_compute_precisions_cholesky",
            "Covariance Type": "full",
            "N": N,
            "D": D,
            "K": K,
            "Old Time (ms)": old_time,
            "Old Std (ms)": old_std,
            "New Time (ms)": new_time,
            "New Std (ms)": new_std,
            "Speedup": speedup,
        })
    
    return results


def benchmark_precisions():
    """Benchmark _compute_precisions for full covariance."""
    print("\n" + "="*80)
    print("BENCHMARK: _compute_precisions (full covariance)")
    print("="*80)
    
    results = []
    for N, D, K in [(500, 20, 5), (1000, 50, 10), (2000, 100, 5)]:
        data = generate_test_data(N, D, K)
        cov = data["cov_full"]
        prec_chol = old_impl._compute_precisions_cholesky(cov, "full")
        
        old_time, old_std = timer(old_impl._compute_precisions, prec_chol, "full")
        new_time, new_std = timer(new_impl._compute_precisions, prec_chol, "full")
        
        speedup = old_time / new_time
        print(f"N={N}, D={D}, K={K}:")
        print(f"  Old: {old_time:.3f} ± {old_std:.3f} ms")
        print(f"  New: {new_time:.3f} ± {new_std:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            "Function": "_compute_precisions",
            "Covariance Type": "full",
            "N": N,
            "D": D,
            "K": K,
            "Old Time (ms)": old_time,
            "Old Std (ms)": old_std,
            "New Time (ms)": new_time,
            "New Std (ms)": new_std,
            "Speedup": speedup,
        })
    
    return results


def benchmark_log_prob_tied():
    """Benchmark tied log probability computation."""
    print("\n" + "="*80)
    print("BENCHMARK: _estimate_log_gaussian_prob_tied_precchol")
    print("="*80)
    
    results = []
    for N, D, K in [(500, 20, 5), (1000, 50, 10), (2000, 100, 20)]:
        data = generate_test_data(N, D, K)
        X = data["X"]
        means = data["means"]
        prec_chol = old_impl._compute_precisions_cholesky(data["cov_tied"], "tied")
        
        old_time, old_std = timer(old_impl._estimate_log_gaussian_prob_tied_precchol, X, means, prec_chol)
        new_time, new_std = timer(new_impl._estimate_log_gaussian_prob_tied_precchol, X, means, prec_chol)
        
        speedup = old_time / new_time
        print(f"N={N}, D={D}, K={K}:")
        print(f"  Old: {old_time:.3f} ± {old_std:.3f} ms")
        print(f"  New: {new_time:.3f} ± {new_std:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            "Function": "_estimate_log_gaussian_prob_tied_precchol",
            "Covariance Type": "tied",
            "N": N,
            "D": D,
            "K": K,
            "Old Time (ms)": old_time,
            "Old Std (ms)": old_std,
            "New Time (ms)": new_time,
            "New Std (ms)": new_std,
            "Speedup": speedup,
        })
    
    return results


def benchmark_log_prob_full():
    """Benchmark full log probability computation."""
    print("\n" + "="*80)
    print("BENCHMARK: _estimate_log_gaussian_prob_full_precchol")
    print("="*80)
    
    results = []
    for N, D, K in [(500, 20, 5), (1000, 50, 10), (2000, 100, 5)]:
        data = generate_test_data(N, D, K)
        X = data["X"]
        means = data["means"]
        prec_chol = old_impl._compute_precisions_cholesky(data["cov_full"], "full")
        
        old_time, old_std = timer(old_impl._estimate_log_gaussian_prob_full_precchol, X, means, prec_chol)
        new_time, new_std = timer(new_impl._estimate_log_gaussian_prob_full_precchol, X, means, prec_chol)
        
        speedup = old_time / new_time
        print(f"N={N}, D={D}, K={K}:")
        print(f"  Old: {old_time:.3f} ± {old_std:.3f} ms")
        print(f"  New: {new_time:.3f} ± {new_std:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            "Function": "_estimate_log_gaussian_prob_full_precchol",
            "Covariance Type": "full",
            "N": N,
            "D": D,
            "K": K,
            "Old Time (ms)": old_time,
            "Old Std (ms)": old_std,
            "New Time (ms)": new_time,
            "New Std (ms)": new_std,
            "Speedup": speedup,
        })
    
    return results


def benchmark_m_step_diag():
    """Benchmark M-step for diagonal covariance."""
    print("\n" + "="*80)
    print("BENCHMARK: _maximization_step (diag covariance)")
    print("="*80)
    
    results = []
    for N, D, K in [(500, 20, 5), (1000, 50, 10), (2000, 100, 20)]:
        data = generate_test_data(N, D, K)
        X = data["X"]
        means = data["means"]
        cov = data["cov_diag"]
        weights = data["weights"]
        log_resp = data["log_resp"]
        
        old_time, old_std = timer(old_impl._maximization_step, X, means, cov, weights, log_resp, "diag")
        new_time, new_std = timer(new_impl._maximization_step, X, means, cov, weights, log_resp, "diag")
        
        speedup = old_time / new_time
        print(f"N={N}, D={D}, K={K}:")
        print(f"  Old: {old_time:.3f} ± {old_std:.3f} ms")
        print(f"  New: {new_time:.3f} ± {new_std:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            "Function": "_maximization_step",
            "Covariance Type": "diag",
            "N": N,
            "D": D,
            "K": K,
            "Old Time (ms)": old_time,
            "Old Std (ms)": old_std,
            "New Time (ms)": new_time,
            "New Std (ms)": new_std,
            "Speedup": speedup,
        })
    
    return results


def benchmark_m_step_tied():
    """Benchmark M-step for tied covariance."""
    print("\n" + "="*80)
    print("BENCHMARK: _maximization_step (tied covariance)")
    print("="*80)
    
    results = []
    for N, D, K in [(500, 20, 5), (1000, 50, 10), (2000, 100, 20)]:
        data = generate_test_data(N, D, K)
        X = data["X"]
        means = data["means"]
        cov = data["cov_tied"]
        weights = data["weights"]
        log_resp = data["log_resp"]
        
        old_time, old_std = timer(old_impl._maximization_step, X, means, cov, weights, log_resp, "tied")
        new_time, new_std = timer(new_impl._maximization_step, X, means, cov, weights, log_resp, "tied")
        
        speedup = old_time / new_time
        print(f"N={N}, D={D}, K={K}:")
        print(f"  Old: {old_time:.3f} ± {old_std:.3f} ms")
        print(f"  New: {new_time:.3f} ± {new_std:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            "Function": "_maximization_step",
            "Covariance Type": "tied",
            "N": N,
            "D": D,
            "K": K,
            "Old Time (ms)": old_time,
            "Old Std (ms)": old_std,
            "New Time (ms)": new_time,
            "New Std (ms)": new_std,
            "Speedup": speedup,
        })
    
    return results


def benchmark_m_step_full():
    """Benchmark M-step for full covariance."""
    print("\n" + "="*80)
    print("BENCHMARK: _maximization_step (full covariance)")
    print("="*80)
    
    results = []
    for N, D, K in [(500, 20, 5), (1000, 50, 10), (2000, 100, 5)]:
        data = generate_test_data(N, D, K)
        X = data["X"]
        means = data["means"]
        cov = data["cov_full"]
        weights = data["weights"]
        log_resp = data["log_resp"]
        
        old_time, old_std = timer(old_impl._maximization_step, X, means, cov, weights, log_resp, "full")
        new_time, new_std = timer(new_impl._maximization_step, X, means, cov, weights, log_resp, "full")
        
        speedup = old_time / new_time
        print(f"N={N}, D={D}, K={K}:")
        print(f"  Old: {old_time:.3f} ± {old_std:.3f} ms")
        print(f"  New: {new_time:.3f} ± {new_std:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            "Function": "_maximization_step",
            "Covariance Type": "full",
            "N": N,
            "D": D,
            "K": K,
            "Old Time (ms)": old_time,
            "Old Std (ms)": old_std,
            "New Time (ms)": new_time,
            "New Std (ms)": new_std,
            "Speedup": speedup,
        })
    
    return results


def benchmark_kmeans_lloyd():
    """Benchmark KMeans Lloyd iterations."""
    print("\n" + "="*80)
    print("BENCHMARK: _kmeans_lloyd_with_init")
    print("="*80)
    
    results = []
    for N, D, K in [(500, 20, 5), (1000, 50, 10), (2000, 100, 20)]:
        data = generate_test_data(N, D, K)
        X = data["X"]
        centroids = data["means"].clone()
        
        old_time, old_std = timer(old_impl._kmeans_lloyd_with_init, X, centroids.clone(), n_iter=10, n_runs=5)
        new_time, new_std = timer(new_impl._kmeans_lloyd_with_init, X, centroids.clone(), n_iter=10, n_runs=5)
        
        speedup = old_time / new_time
        print(f"N={N}, D={D}, K={K}:")
        print(f"  Old: {old_time:.3f} ± {old_std:.3f} ms")
        print(f"  New: {new_time:.3f} ± {new_std:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            "Function": "_kmeans_lloyd_with_init",
            "Covariance Type": "N/A",
            "N": N,
            "D": D,
            "K": K,
            "Old Time (ms)": old_time,
            "Old Std (ms)": old_std,
            "New Time (ms)": new_time,
            "New Std (ms)": new_std,
            "Speedup": speedup,
        })
    
    return results


def benchmark_full_fit():
    """Benchmark full GMM fit (end-to-end)."""
    print("\n" + "="*80)
    print("BENCHMARK: TorchGaussianMixture.fit() - END-TO-END")
    print("="*80)
    
    results = []
    for cov_type in ["diag", "tied", "full"]:
        print(f"\n--- Covariance type: {cov_type} ---")
        for N, D, K in [(500, 20, 5), (1000, 50, 5)]:
            data = generate_test_data(N, D, K)
            X = data["X"]
            
            def fit_old():
                model = old_impl.TorchGaussianMixture(
                    n_components=K,
                    covariance_type=cov_type,
                    max_iter=20,
                    n_init=1,
                    init_params="random",
                )
                torch.manual_seed(42)
                model.fit(X)
                return model
            
            def fit_new():
                model = new_impl.TorchGaussianMixture(
                    n_components=K,
                    covariance_type=cov_type,
                    max_iter=20,
                    n_init=1,
                    init_params="random",
                )
                torch.manual_seed(42)
                model.fit(X)
                return model
            
            old_time, old_std = timer(fit_old, n_runs=3, warmup=1)
            new_time, new_std = timer(fit_new, n_runs=3, warmup=1)
            
            speedup = old_time / new_time
            print(f"N={N}, D={D}, K={K}:")
            print(f"  Old: {old_time:.3f} ± {old_std:.3f} ms")
            print(f"  New: {new_time:.3f} ± {new_std:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            results.append({
                "Function": "TorchGaussianMixture.fit",
                "Covariance Type": cov_type,
                "N": N,
                "D": D,
                "K": K,
                "Old Time (ms)": old_time,
                "Old Std (ms)": old_std,
                "New Time (ms)": new_time,
                "New Std (ms)": new_std,
                "Speedup": speedup,
            })
    
    return results


def main():
    """Run all benchmarks."""
    print("="*80)
    print("GMM PARALLELIZATION SPEEDUP BENCHMARKS")
    print("Comparing loop-based vs vectorized implementations")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Collect all results
    all_results = []
    
    # Run all benchmarks
    all_results.extend(benchmark_precisions_cholesky())
    all_results.extend(benchmark_precisions())
    all_results.extend(benchmark_log_prob_tied())
    all_results.extend(benchmark_log_prob_full())
    all_results.extend(benchmark_m_step_diag())
    all_results.extend(benchmark_m_step_tied())
    all_results.extend(benchmark_m_step_full())
    all_results.extend(benchmark_kmeans_lloyd())
    all_results.extend(benchmark_full_fit())
    
    print("\n" + "="*80)
    print("BENCHMARKS COMPLETE")
    print("="*80)
    
    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    column_order = [
        "Function",
        "Covariance Type",
        "N",
        "D",
        "K",
        "Old Time (ms)",
        "Old Std (ms)",
        "New Time (ms)",
        "New Std (ms)",
        "Speedup",
    ]
    df = df[column_order]
    
    # Save to Excel in the same directory as this script
    output_file = os.path.join(os.path.dirname(__file__), "speedup_comparison.xlsx")
    df.to_excel(output_file, index=False, sheet_name="Speedup Results")
    
    print(f"\n✓ Results exported to: {output_file}")
    print(f"  Total benchmarks: {len(df)}")
    print(f"  Average speedup: {df['Speedup'].mean():.2f}x")
    print(f"  Max speedup: {df['Speedup'].max():.2f}x")
    print(f"  Min speedup: {df['Speedup'].min():.2f}x")


if __name__ == "__main__":
    main()
