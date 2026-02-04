"""Measure scikit-learn GaussianMixture wall time (10 runs) for comparison with torch."""

import os
import json
import time
import warnings
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture

OUTDIR = Path("profiles")
OUTDIR.mkdir(exist_ok=True)

NUM_RUNS = 10

def measure_fit_multiple(X, n_components=5, cov_type="full",
                         max_iter=50, n_init=1, init_params="random", tol=1e-3,
                         warmup_iter=2, num_runs=NUM_RUNS):
    """Measure wall-time across multiple runs."""
    print(
        f"[{cov_type:9s}] setup: N={X.shape[0]}, D={X.shape[1]}, "
        f"K={n_components}, max_iter={max_iter}, n_init={n_init}, init={init_params}",
        flush=True,
    )

    gmm_template = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        max_iter=max_iter,
        n_init=n_init,
        init_params=init_params,
        tol=tol,
        random_state=42,
        verbose=0,
    )

    # Warmup once
    warm = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        max_iter=warmup_iter,
        n_init=1,
        init_params=init_params,
        tol=tol,
        random_state=42,
        verbose=0,
    )

    print(f"[{cov_type:9s}] warmup ({warmup_iter} iter)...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm.fit(X)

    # Measure multiple runs with different random data
    times = []
    for run in range(num_runs):
        print(f"[{cov_type:9s}] run {run+1}/{num_runs}...", flush=True, end="")
        
        # Generate NEW random data for each run with different seed
        np.random.seed(42 + run)
        X_run = np.random.randn(X.shape[0], X.shape[1]).astype(np.float32)
        
        # Create new GMM instance for each run (fresh random state each time)
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=cov_type,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            tol=tol,
            random_state=42 + run,  # Different seed each run
            verbose=0,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            gmm.fit(X_run)
            t1 = time.perf_counter()

        wall = t1 - t0
        times.append(wall)
        print(f" {wall:.6f} s", flush=True)

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"[{cov_type:9s}] mean={mean_time:.6f} s | std={std_time:.6f} s\n", flush=True)
    
    return {
        "mean": float(mean_time),
        "std": float(std_time),
        "runs": [float(t) for t in times],
    }

if __name__ == "__main__":
    print(f"scikit-learn GaussianMixture wall-time measurement ({NUM_RUNS} runs)\n")

    # Generate ONCE so each covariance type sees identical data
    n_samples = 100000
    n_features = 500
    n_components = 5

    print("[data] generating...", flush=True)
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    sklearn_stats = {}
    for cov_type in ["diag", "spherical", "tied", "full"]:
        sklearn_stats[cov_type] = measure_fit_multiple(
            X,
            n_components=n_components,
            cov_type=cov_type,
            max_iter=50,
            n_init=1,
            init_params="random",
            tol=1e-3,
            warmup_iter=2,
            num_runs=NUM_RUNS,
        )

    stats_path = OUTDIR / "sklearn_runtimes_stats.json"
    with open(stats_path, "w") as f:
        json.dump(sklearn_stats, f, indent=2)
    print(f"Exported sklearn runtime stats to: {stats_path}")
