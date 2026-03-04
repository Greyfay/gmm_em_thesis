"""
Performance comparison test for _v1.py vs _v2.py implementations.

This test measures:
1. Total fit runtime (seconds)
2. Time spent in M-step covariance update (seconds)
3. Final log likelihood (lower_bound_)
4. Iterations to converge (n_iter_)

_v1.py is tested with default config types (spherical, diag, tied, full).
_v2.py is tested with tiling_size sweep (16, 32, 64, 128).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple

# Add parent directory to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from implementation import _v1 as v1
from implementation import _v2 as v2


def generate_test_data(
    N: int = 1000,
    D: int = 20,
    K: int = 5,
    random_seed: int = 42,
    device: str = "cpu",
    dtype=torch.float64,
) -> torch.Tensor:
    """Generate synthetic test data for benchmarking."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    X = torch.randn(N, D, device=device, dtype=dtype)
    return X


def measure_fit_runtime(
    model: torch.nn.Module,
    X: torch.Tensor,
    n_runs: int = 3,
    warmup: int = 1,
) -> Tuple[float, float]:
    """Measure fit runtime with warmup.
    
    Returns:
        (mean_time_seconds, std_time_seconds)
    """
    # Warmup runs
    for _ in range(warmup):
        model.fit(X.clone())
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        model.fit(X.clone())
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times)


def instrument_maximization_step_timing(implementation):
    """Create wrapper that measures M-step execution time.
    
    This temporarily replaces _maximization_step to track timing.
    """
    original_m_step = implementation._maximization_step
    total_m_step_time = {'time': 0.0, 'count': 0}
    
    def timed_m_step(*args, **kwargs):
        start = time.perf_counter()
        result = original_m_step(*args, **kwargs)
        elapsed = time.perf_counter() - start
        total_m_step_time['time'] += elapsed
        total_m_step_time['count'] += 1
        return result
    
    # Replace the function
    implementation._maximization_step = timed_m_step
    
    return total_m_step_time, lambda: total_m_step_time


def test_v1_default_configs():
    """Test _v1 with default covariance types.
    
    Default config types: spherical, diag, tied, full
    """
    print("\n" + "="*80)
    print("TEST: _v1.py with Default Covariance Types")
    print("="*80)
    
    results = []
    device = "cuda"
    
    # Test configurations: (N, D, K, max_iter)
    test_configs = [
        (500, 20, 5, 50),      # Small dataset
        (1000, 50, 10, 50),    # Medium dataset
        (2000, 100, 20, 50),   # Larger dataset
    ]
    
    default_cov_types = ["spherical", "diag", "tied", "full"]
    
    for N, D, K, max_iter in test_configs:
        print(f"\n--- Config: N={N}, D={D}, K={K}, max_iter={max_iter} ---")
        
        X = generate_test_data(N, D, K, device=device)
        
        for cov_type in default_cov_types:
            print(f"  Testing {cov_type}...", end=" ", flush=True)
            
            # Reset M-step timing tracker
            total_m_step_time = {'time': 0.0, 'count': 0}
            original_m_step = v1._maximization_step
            
            def timed_m_step(*args, **kwargs):
                nonlocal total_m_step_time
                start = time.perf_counter()
                result = original_m_step(*args, **kwargs)
                elapsed = time.perf_counter() - start
                total_m_step_time['time'] += elapsed
                total_m_step_time['count'] += 1
                return result
            
            # Replace the function
            v1._maximization_step = timed_m_step
            
            try:
                # Create model and measure runtime
                model = v1.TorchGaussianMixture(
                    n_components=K,
                    covariance_type=cov_type,
                    max_iter=max_iter,
                    n_init=1,
                    init_params="random",
                    device=device,
                    dtype=torch.float64,
                )
                
                # Single run (not multiple warmup runs for simplicity)
                torch.manual_seed(42)
                start = time.perf_counter()
                model.fit(X.clone())
                fit_time = time.perf_counter() - start
                
                # Extract metrics
                final_ll = model.lower_bound_
                n_iter = model.n_iter_
                m_step_time = total_m_step_time['time']
                
                print(f"✓ fit_time={fit_time:.4f}s, m_step={m_step_time:.4f}s, ll={final_ll:.4f}, iters={n_iter}")
                
                results.append({
                    "Implementation": "v1",
                    "Covariance Type": cov_type,
                    "N": N,
                    "D": D,
                    "K": K,
                    "Max Iterations": max_iter,
                    "Total Fit Time (s)": fit_time,
                    "M-step Time (s)": m_step_time,
                    "M-step Count": total_m_step_time['count'],
                    "Avg M-step Time (ms)": (m_step_time / max(total_m_step_time['count'], 1)) * 1000 if total_m_step_time['count'] > 0 else 0,
                    "Final Log-Likelihood": final_ll,
                    "Iterations to Converge": n_iter,
                })
            
            finally:
                # Restore original function
                v1._maximization_step = original_m_step
    
    return results


def test_v2_tiling_size_sweep():
    """Test _v2 with tiling_size sweep (16, 32, 64, 128).
    
    _v2 has an optional tiling_size parameter for memory tiling optimizations.
    """
    print("\n" + "="*80)
    print("TEST: _v2.py with Tiling Size Sweep")
    print("="*80)
    
    results = []
    device = "cuda"
    
    # Test configurations: (N, D, K, max_iter)
    test_configs = [
        (500, 20, 5, 50),      # Small dataset
        (1000, 50, 10, 50),    # Medium dataset
        (2000, 100, 20, 50),   # Larger dataset
    ]
    
    tiling_sizes = [16, 32, 64, 128]
    default_cov_type = "full"  # Use full for tiling tests
    
    for N, D, K, max_iter in test_configs:
        print(f"\n--- Config: N={N}, D={D}, K={K}, max_iter={max_iter} ---")
        
        X = generate_test_data(N, D, K, device=device)
        
        for tiling_size in tiling_sizes:
            print(f"  Testing tiling_size={tiling_size}...", end=" ", flush=True)
            
            # Reset M-step timing tracker
            total_m_step_time = {'time': 0.0, 'count': 0}
            original_m_step = v2._maximization_step
            
            def timed_m_step(*args, **kwargs):
                nonlocal total_m_step_time
                start = time.perf_counter()
                result = original_m_step(*args, **kwargs)
                elapsed = time.perf_counter() - start
                total_m_step_time['time'] += elapsed
                total_m_step_time['count'] += 1
                return result
            
            # Replace the function
            v2._maximization_step = timed_m_step
            
            try:
                # Create model and measure runtime
                model = v2.TorchGaussianMixture(
                    n_components=K,
                    covariance_type=default_cov_type,
                    max_iter=max_iter,
                    n_init=1,
                    init_params="random",
                    device=device,
                    dtype=torch.float64,
                    tiling_size=tiling_size,
                )
                
                # Single run (not multiple warmup runs for simplicity)
                torch.manual_seed(42)
                start = time.perf_counter()
                model.fit(X.clone())
                fit_time = time.perf_counter() - start
                
                # Extract metrics
                final_ll = model.lower_bound_
                n_iter = model.n_iter_
                m_step_time = total_m_step_time['time']
                
                print(f"✓ fit_time={fit_time:.4f}s, m_step={m_step_time:.4f}s, ll={final_ll:.4f}, iters={n_iter}")
                
                results.append({
                    "Implementation": "v2",
                    "Tiling Size": tiling_size,
                    "Covariance Type": default_cov_type,
                    "N": N,
                    "D": D,
                    "K": K,
                    "Max Iterations": max_iter,
                    "Total Fit Time (s)": fit_time,
                    "M-step Time (s)": m_step_time,
                    "M-step Count": total_m_step_time['count'],
                    "Avg M-step Time (ms)": (m_step_time / max(total_m_step_time['count'], 1)) * 1000 if total_m_step_time['count'] > 0 else 0,
                    "Final Log-Likelihood": final_ll,
                    "Iterations to Converge": n_iter,
                })
            
            finally:
                # Restore original function
                v2._maximization_step = original_m_step
    
    return results


def main():
    """Run all performance tests."""
    # Only run on CUDA (GPU)
    if not torch.cuda.is_available():
        print("ERROR: This test requires CUDA (GPU). No CUDA device found.")
        print("Please run this script on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    
    print("="*80)
    print("PERFORMANCE TEST: v1 vs v2")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: CUDA")
    print(f"CUDA available: {torch.cuda.is_available()}\n")
    
    # Run tests
    v1_results = test_v1_default_configs()
    v2_results = test_v2_tiling_size_sweep()
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create DataFrames
    if v1_results:
        df_v1 = pd.DataFrame(v1_results)
        print(f"\n_v1 Results ({len(df_v1)} runs):")
        print(df_v1.to_string())
        
        # Save to CSV
        output_file = os.path.join(os.path.dirname(__file__), "v1_performance_results.csv")
        df_v1.to_csv(output_file, index=False)
        print(f"\n✓ _v1 results saved to: {output_file}")
    
    if v2_results:
        df_v2 = pd.DataFrame(v2_results)
        print(f"\n_v2 Results ({len(df_v2)} runs):")
        print(df_v2.to_string())
        
        # Save to CSV
        output_file = os.path.join(os.path.dirname(__file__), "v2_performance_results.csv")
        df_v2.to_csv(output_file, index=False)
        print(f"\n✓ _v2 results saved to: {output_file}")
    
    # Combine and save summary
    if v1_results and v2_results:
        # Create summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        print("\n_v1 Summary (by covariance type):")
        v1_grouped = df_v1.groupby("Covariance Type").agg({
            "Total Fit Time (s)": ["mean", "min", "max"],
            "M-step Time (s)": ["mean"],
            "Final Log-Likelihood": ["mean"],
            "Iterations to Converge": ["mean"],
        })
        print(v1_grouped)
        
        print("\n_v2 Summary (by tiling size):")
        v2_grouped = df_v2.groupby("Tiling Size").agg({
            "Total Fit Time (s)": ["mean", "min", "max"],
            "M-step Time (s)": ["mean"],
            "Final Log-Likelihood": ["mean"],
            "Iterations to Converge": ["mean"],
        })
        print(v2_grouped)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
