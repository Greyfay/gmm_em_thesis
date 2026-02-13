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


def benchmark_memory():
    """Benchmark peak memory usage during operations.
    
    Uses torch.cuda.memory_allocated() for GPU or estimates for CPU.
    """
    print("\n" + "="*80)
    print("BENCHMARK: MEMORY USAGE")
    print("="*80)
    
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test configurations: (N, D, K)
    # Testing D = [20, 50, 100] × N = [1k, 10k, 100k]
    test_configs = [
        (1000, 20, 5),      # N=1k, D=20, K=5
        (10000, 20, 5),     # N=10k, D=20, K=5
        (100000, 20, 5),    # N=100k, D=20, K=5
        (1000, 50, 5),      # N=1k, D=50, K=5
        (10000, 50, 5),     # N=10k, D=50, K=5
        (100000, 50, 5),    # N=100k, D=50, K=5
        (1000, 100, 5),     # N=1k, D=100, K=5
        (10000, 100, 5),    # N=10k, D=100, K=5
        (100000, 100, 5),   # N=100k, D=100, K=5
    ]
    
    for N, D, K in test_configs:
        for cov_type in ["spherical", "diag", "tied", "full"]:
            print(f"\n--- Memory: {cov_type}, N={N}, D={D}, K={K} ---")
            
            data = generate_test_data(N, D, K, device=device)
            X = data["X"]
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Measure old implementation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            model_old = old_impl.TorchGaussianMixture(
                n_components=K,
                covariance_type=cov_type,
                max_iter=1,
                n_init=1,
                init_params="random",
            )
            torch.manual_seed(42)
            model_old.fit(X)
            
            old_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Measure new implementation
            model_new = new_impl.TorchGaussianMixture(
                n_components=K,
                covariance_type=cov_type,
                max_iter=1,
                n_init=1,
                init_params="random",
            )
            torch.manual_seed(42)
            model_new.fit(X)
            
            new_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            
            # Calculate memory reduction
            mem_reduction = (((old_mem - new_mem) / old_mem) * 100) if old_mem > 0 else 0
            
            old_mem_mb = old_mem / (1024**2) if torch.cuda.is_available() else 0
            new_mem_mb = new_mem / (1024**2) if torch.cuda.is_available() else 0
            
            print(f"  Old: {old_mem_mb:.2f} MB")
            print(f"  New: {new_mem_mb:.2f} MB")
            print(f"  Reduction: {mem_reduction:.1f}%")
            
            results.append({
                "Function": "TorchGaussianMixture.fit",
                "Metric": "Peak Memory (MB)",
                "Covariance Type": cov_type,
                "N": N,
                "D": D,
                "K": K,
                "Old Value": old_mem_mb,
                "New Value": new_mem_mb,
                "Reduction %": mem_reduction,
                "Device": device,
            })
    
    return results


def benchmark_bandwidth():
    """Benchmark memory bandwidth utilization.
    
    Estimates bytes read/written per second during operations.
    For GPU, uses torch profiling if available, otherwise estimates from timing.
    """
    print("\n" + "="*80)
    print("BENCHMARK: MEMORY BANDWIDTH")
    print("="*80)
    
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test configurations: (N, D, K)
    # Testing D = [20, 50, 100] × N = [1k, 10k, 100k]
    test_configs = [
        (1000, 20, 5),      # N=1k, D=20, K=5
        (10000, 20, 5),     # N=10k, D=20, K=5
        (100000, 20, 5),    # N=100k, D=20, K=5
        (1000, 50, 5),      # N=1k, D=50, K=5
        (10000, 50, 5),     # N=10k, D=50, K=5
        (100000, 50, 5),    # N=100k, D=50, K=5
        (1000, 100, 5),     # N=1k, D=100, K=5
        (10000, 100, 5),    # N=10k, D=100, K=5
        (100000, 100, 5),   # N=100k, D=100, K=5
    ]
    
    def estimate_flops_and_bytes(N, D, K, cov_type):
        """Estimate FLOPs and bytes for M-step (most compute-heavy operation)."""
        # Log probability computation: N*K*D operations
        log_prob_flops = N * K * D * 2
        
        # M-step updates
        if cov_type == "spherical":
            m_step_flops = N * K + K * D
        elif cov_type == "diag":
            m_step_flops = N * K * D + K * D
        elif cov_type == "tied":
            m_step_flops = N * K * D + D * D
        else:  # full
            m_step_flops = N * K * D + K * D * D
        
        total_flops = log_prob_flops + m_step_flops
        
        # Estimate bytes: read X, means, covariances; write intermediate results
        bytes_read = N * D * 8 + K * D * 8  # X and means
        if cov_type == "spherical":
            bytes_read += K * 8
            bytes_write = K * (D + 1) * 8
        elif cov_type == "diag":
            bytes_read += K * D * 8
            bytes_write = K * D * 8
        elif cov_type == "tied":
            bytes_read += D * D * 8
            bytes_write = D * D * 8 + K * 8
        else:  # full
            bytes_read += K * D * D * 8
            bytes_write = K * D * D * 8 + K * 8
        
        total_bytes = bytes_read + bytes_write
        return total_flops, total_bytes
    
    for N, D, K in test_configs:
        for cov_type in ["spherical", "diag", "tied", "full"]:
            print(f"\n--- Bandwidth: {cov_type}, N={N}, D={D}, K={K} ---")
            
            data = generate_test_data(N, D, K, device=device)
            X = data["X"]
            means = data["means"].clone()
            weights = data["weights"].clone()
            log_resp = data["log_resp"].clone()
            
            if cov_type == "spherical":
                cov = data["cov_spherical"].clone()
            elif cov_type == "diag":
                cov = data["cov_diag"].clone()
            elif cov_type == "tied":
                cov = data["cov_tied"].clone()
            else:  # full
                cov = data["cov_full"].clone()
            
            # Measure time for M-step (compute-heavy operation)
            old_time_ms, _ = timer(
                old_impl._maximization_step,
                X, means.clone(), cov.clone(), weights.clone(), log_resp.clone(), cov_type,
                n_runs=5, warmup=1
            )
            
            new_time_ms, _ = timer(
                new_impl._maximization_step,
                X, means.clone(), cov.clone(), weights.clone(), log_resp.clone(), cov_type,
                n_runs=5, warmup=1
            )
            
            # Estimate theoretical bytes
            _, est_bytes = estimate_flops_and_bytes(N, D, K, cov_type)
            
            # Calculate bandwidth (GB/s)
            old_bw = (est_bytes / (1024**3)) / (old_time_ms / 1000)
            new_bw = (est_bytes / (1024**3)) / (new_time_ms / 1000)
            
            bw_improvement = (((new_bw - old_bw) / old_bw) * 100) if old_bw > 0 else 0
            
            print(f"  Old: {old_bw:.2f} GB/s (time: {old_time_ms:.3f} ms)")
            print(f"  New: {new_bw:.2f} GB/s (time: {new_time_ms:.3f} ms)")
            print(f"  Improvement: {bw_improvement:.1f}%")
            print(f"  Est. bytes: {est_bytes / (1024**2):.1f} MB")
            
            results.append({
                "Function": "_maximization_step",
                "Metric": "Bandwidth (GB/s)",
                "Covariance Type": cov_type,
                "N": N,
                "D": D,
                "K": K,
                "Old Value": old_bw,
                "New Value": new_bw,
                "Improvement %": bw_improvement,
                "Estimated Bytes (MB)": est_bytes / (1024**2),
                "Device": device,
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
    speedup_results = []
    memory_results = []
    bandwidth_results = []
    
    # Run all benchmarks
    speedup_results.extend(benchmark_precisions_cholesky())
    speedup_results.extend(benchmark_precisions())
    speedup_results.extend(benchmark_log_prob_tied())
    speedup_results.extend(benchmark_log_prob_full())
    speedup_results.extend(benchmark_m_step_diag())
    speedup_results.extend(benchmark_m_step_tied())
    speedup_results.extend(benchmark_m_step_full())
    speedup_results.extend(benchmark_kmeans_lloyd())
    speedup_results.extend(benchmark_full_fit())
    
    # Run memory and bandwidth benchmarks
    memory_results.extend(benchmark_memory())
    bandwidth_results.extend(benchmark_bandwidth())
    
    print("\n" + "="*80)
    print("BENCHMARKS COMPLETE")
    print("="*80)
    
    # Save speedup results
    if speedup_results:
        df_speedup = pd.DataFrame(speedup_results)
        
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
        df_speedup = df_speedup[column_order]
        
        # Sort by covariance type (spherical, diag, tied, full, N/A), then N, then D
        cov_type_order = {"spherical": 0, "diag": 1, "tied": 2, "full": 3, "N/A": 4}
        df_speedup["_cov_order"] = df_speedup["Covariance Type"].map(cov_type_order)
        df_speedup = df_speedup.sort_values(["_cov_order", "N", "D"]).drop("_cov_order", axis=1)
        
        # Save speedup results to Excel
        output_file = os.path.join(os.path.dirname(__file__), "speedup_comparison.xlsx")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_speedup.to_excel(writer, sheet_name="Speedup Results", index=False)
        
        print(f"\n✓ Speedup results exported to: {output_file}")
        print(f"  Total speedup benchmarks: {len(df_speedup)}")
        print(f"  Average speedup: {df_speedup['Speedup'].mean():.2f}x")
        print(f"  Max speedup: {df_speedup['Speedup'].max():.2f}x")
        print(f"  Min speedup: {df_speedup['Speedup'].min():.2f}x")
    
    # Save memory results
    if memory_results:
        df_memory = pd.DataFrame(memory_results)
        column_order_mem = [
            "Function",
            "Metric",
            "Covariance Type",
            "N",
            "D",
            "K",
            "Old Value",
            "New Value",
            "Reduction %",
            "Device",
        ]
        df_memory = df_memory[column_order_mem]
        
        # Rename columns to include units
        df_memory = df_memory.rename(columns={
            "Old Value": "Old Memory (MB)",
            "New Value": "New Memory (MB)",
        })
        
        # Sort by covariance type (spherical, diag, tied, full), then N, then D
        cov_type_order = {"spherical": 0, "diag": 1, "tied": 2, "full": 3}
        df_memory["_cov_order"] = df_memory["Covariance Type"].map(cov_type_order)
        df_memory = df_memory.sort_values(["_cov_order", "N", "D"]).drop("_cov_order", axis=1)
        
        output_file = os.path.join(os.path.dirname(__file__), "memory_comparison.xlsx")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_memory.to_excel(writer, sheet_name="Memory Results", index=False)
        
        print(f"\n✓ Memory results exported to: {output_file}")
        print(f"  Total memory benchmarks: {len(df_memory)}")
        print(f"  Average memory reduction: {df_memory['Reduction %'].mean():.1f}%")
        if df_memory['Reduction %'].max() > 0:
            print(f"  Max memory reduction: {df_memory['Reduction %'].max():.1f}%")
        if df_memory['Reduction %'].min() < 0:
            print(f"  Max memory increase: {-df_memory['Reduction %'].min():.1f}%")
    
    # Save bandwidth results
    if bandwidth_results:
        df_bandwidth = pd.DataFrame(bandwidth_results)
        column_order_bw = [
            "Function",
            "Metric",
            "Covariance Type",
            "N",
            "D",
            "K",
            "Old Value",
            "New Value",
            "Improvement %",
            "Estimated Bytes (MB)",
            "Device",
        ]
        df_bandwidth = df_bandwidth[column_order_bw]
        
        # Rename columns to include units
        df_bandwidth = df_bandwidth.rename(columns={
            "Old Value": "Old Bandwidth (GB/s)",
            "New Value": "New Bandwidth (GB/s)",
        })
        
        # Sort by covariance type (spherical, diag, tied, full), then N, then D
        cov_type_order = {"spherical": 0, "diag": 1, "tied": 2, "full": 3}
        df_bandwidth["_cov_order"] = df_bandwidth["Covariance Type"].map(cov_type_order)
        df_bandwidth = df_bandwidth.sort_values(["_cov_order", "N", "D"]).drop("_cov_order", axis=1)
        
        output_file = os.path.join(os.path.dirname(__file__), "bandwidth_comparison.xlsx")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_bandwidth.to_excel(writer, sheet_name="Bandwidth Results", index=False)
        
        print(f"\n✓ Bandwidth results exported to: {output_file}")
        print(f"  Total bandwidth benchmarks: {len(df_bandwidth)}")
        print(f"  Average bandwidth improvement: {df_bandwidth['Improvement %'].mean():.1f}%")
        print(f"  Max bandwidth improvement: {df_bandwidth['Improvement %'].max():.1f}%")
        print(f"  Min bandwidth improvement: {df_bandwidth['Improvement %'].min():.1f}%")


if __name__ == "__main__":
    main()
