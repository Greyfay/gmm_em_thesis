"""Measure scikit-learn GaussianMixture wall time for comparison with torch."""

import sys
import os
import warnings
import json
import time
import numpy as np
from sklearn.mixture import GaussianMixture
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTDIR = Path("profiles")
OUTDIR.mkdir(exist_ok=True)


def measure_fit(n_samples=100000, n_features=200, n_components=5, cov_type="full"):
    """Measure wall-time of a single fit() call."""
    print(f"[{cov_type:9s}] Generating data...", flush=True)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        max_iter=1000,
        n_init=100,
        init_params="kmeans",
        random_state=42,   
    )

    # Warmup
    print(f"[{cov_type:9s}] Warmup fit (this will take a few minutes)...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm.fit(X)

    # Measure wall time
    print(f"[{cov_type:9s}] Measuring wall time...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        gmm.fit(X)
        t1 = time.perf_counter()
    
    wall_time_s = t1 - t0
    print(f"[{cov_type:9s}] wall={wall_time_s:.6f} s\n", flush=True)
    return wall_time_s


if __name__ == "__main__":
    print("scikit-learn GaussianMixture wall-time measurement\n")
    
    sklearn_times = {}
    for cov_type in ["diag", "spherical", "tied", "full"]:
        sklearn_times[cov_type] = measure_fit(cov_type=cov_type)
    
    # Export wall times for comparison
    times_path = OUTDIR / "sklearn_runtimes.json"
    with open(times_path, "w") as f:
        json.dump(sklearn_times, f, indent=2)
    print(f"\nExported sklearn runtimes to: {times_path}")