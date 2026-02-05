"""Profile scikit-learn GaussianMixture for figure data (10 runs across configurations)."""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture

OUTDIR = Path("profiles")
OUTDIR.mkdir(exist_ok=True)

NUM_RUNS = 10

# Test configurations
SAMPLE_SIZES = [10000, 100000, 1000000]
DIM_COMPONENTS = [(100, 5), (20, 20), (5, 50)]
COV_TYPES = ["diag", "spherical", "tied", "full"]

def measure_fit_multiple(X, n_components=5, cov_type="full",
                         max_iter=20, n_init=1, init_params="kmeans", 
                         num_runs=NUM_RUNS):
    """Measure wall-time across multiple runs."""
    
    times = []
    all_means = []
    all_weights = []
    
    for run in range(num_runs):
        # Generate NEW random data for each run
        np.random.seed(42 + run)
        X_run = np.random.randn(X.shape[0], X.shape[1]).astype(np.float32)
        
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=cov_type,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            verbose=0,
            random_state=42 + run,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            gmm.fit(X_run)
            t1 = time.perf_counter()

        times.append(t1 - t0)
        all_means.append(gmm.means_.copy())
        all_weights.append(gmm.weights_.copy())
        
        # Print convergence info for this run
        print(f"\n    Run {run+1}: converged={gmm.converged_}, n_iter={gmm.n_iter_}, time={times[-1]:.6f}s")
        print(f"    Weights: {gmm.weights_}")
        if n_components <= 5:  # Only print means for small K
            print(f"    Means (first component): {gmm.means_[0][:5]}...")  # First 5 dims only
    
    # Compute statistics across runs
    all_means = np.array(all_means)  # shape: (num_runs, K, D)
    all_weights = np.array(all_weights)  # shape: (num_runs, K)
    
    mean_weights = np.mean(all_weights, axis=0)
    std_weights = np.std(all_weights, axis=0)
    mean_means = np.mean(all_means, axis=0)  # shape: (K, D)
    std_means = np.std(all_means, axis=0)  # shape: (K, D)
    
    print(f"\n    === Summary across {num_runs} runs ===")
    print(f"    Mean weights: {mean_weights}")
    print(f"    Std weights:  {std_weights}")
    if n_components <= 5:
        print(f"    Mean of means (comp 0, first 5 dims): {mean_means[0][:5]}")
        print(f"    Std of means (comp 0, first 5 dims):  {std_means[0][:5]}")
    print(f"    Max std across all mean components: {np.max(std_means):.6f}")
    print()
    
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "param_std_weights": float(np.mean(std_weights)),
        "param_std_means": float(np.max(std_means)),
    }

if __name__ == "__main__":
    print("scikit-learn GaussianMixture figure profiling")
    print(f"Configurations: {len(SAMPLE_SIZES)} × {len(DIM_COMPONENTS)} × {len(COV_TYPES)} = {len(SAMPLE_SIZES)*len(DIM_COMPONENTS)*len(COV_TYPES)}")
    print(f"Runs per config: {NUM_RUNS}\n")

    results = []
    config_count = 0
    total_configs = len(SAMPLE_SIZES) * len(DIM_COMPONENTS) * len(COV_TYPES)

    for n_samples in SAMPLE_SIZES:
        for d, k in DIM_COMPONENTS:
            for cov_type in COV_TYPES:
                config_count += 1
                print(f"[{config_count:2d}/{total_configs}] N={n_samples:7d}, D={d:3d}, K={k:2d}, cov={cov_type:10s}...", 
                      end="", flush=True)

                # Generate data once for all runs (we'll regenerate inside measure_fit_multiple)
                X = np.random.randn(n_samples, d).astype(np.float32)
                
                stats = measure_fit_multiple(
                    X,
                    n_components=k,
                    cov_type=cov_type,
                    max_iter=20,
                    n_init=1,
                    init_params="kmeans",
                    num_runs=NUM_RUNS,
                )

                results.append({
                    "n_samples": n_samples,
                    "n_dims": d,
                    "n_components": k,
                    "cov_type": cov_type,
                    "mean_time_s": stats["mean"],
                    "std_time_s": stats["std"],
                    "min_time_s": stats["min"],
                    "max_time_s": stats["max"],
                    "param_std_weights": stats["param_std_weights"],
                    "param_std_means_max": stats["param_std_means"],
                })
                
                print(f" mean={stats['mean']:.6f} s ± {stats['std']:.6f} s")

    # Export to Excel
    df = pd.DataFrame(results)
    out_xlsx = OUTDIR / "figure_profiling_sklearn.xlsx"
    df.to_excel(out_xlsx, index=False)
    print(f"\nExported {len(results)} configurations to: {out_xlsx}")

    # Also export to JSON for easy access
    json_results = {str(r): results for r, results in enumerate(results)}
    json_path = OUTDIR / "figure_profiling_sklearn.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported to: {json_path}")
