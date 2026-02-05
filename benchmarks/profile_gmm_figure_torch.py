"""Profile TorchGaussianMixture for figure data (10 runs across configurations)."""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation._torch_gmm_em import TorchGaussianMixture  # noqa: E402

OUTDIR = Path("profiles")
OUTDIR.mkdir(exist_ok=True)

NUM_RUNS = 10

# Test configurations
SAMPLE_SIZES = [10000, 100000, 1000000]
DIM_COMPONENTS = [(100, 5), (20, 20), (5, 50)]
COV_TYPES = ["diag", "spherical", "tied", "full"]

def _wall_time_fit(gmm: TorchGaussianMixture, X: torch.Tensor) -> float:
    """Measure true wall time of gmm.fit(X) with CUDA synchronization."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gmm.fit(X)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0

def measure_baseline_multiple(n_samples=10000, n_dims=100, n_components=5, 
                              cov_type="full", num_runs=NUM_RUNS):
    """Measure baseline wall-time across multiple runs with different data."""
    
    times = []
    for run in range(num_runs):
        # Generate NEW random data for each run with different seed
        torch.manual_seed(42 + run)
        torch.cuda.manual_seed(42 + run)
        X_run = torch.randn(n_samples, n_dims, device="cuda", dtype=torch.float32)
        
        gmm = TorchGaussianMixture(
            n_components=n_components,
            covariance_type=cov_type,
            max_iter=20,
            n_init=1,
            init_params="kmeans",
            device="cuda",
            dtype=torch.float32,
        )
        
        # Warmup fit (not timed)
        gmm.fit(X_run)
        
        # Measure
        t_run = _wall_time_fit(gmm, X_run)
        times.append(t_run)
    
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    print("PyTorch GaussianMixture figure profiling")
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

                stats = measure_baseline_multiple(
                    n_samples=n_samples,
                    n_dims=d,
                    n_components=k,
                    cov_type=cov_type,
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
                })
                
                print(f" mean={stats['mean']:.6f} s ± {stats['std']:.6f} s")

    # Export to Excel
    df = pd.DataFrame(results)
    out_xlsx = OUTDIR / "figure_profiling_torch.xlsx"
    df.to_excel(out_xlsx, index=False)
    print(f"\nExported {len(results)} configurations to: {out_xlsx}")

    # Also export to JSON for easy access
    json_path = OUTDIR / "figure_profiling_torch.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported to: {json_path}")
