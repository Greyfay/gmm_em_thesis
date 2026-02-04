"""Measure scikit-learn GaussianMixture wall time for comparison with torch."""

import os
import json
import time
import warnings
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture

OUTDIR = Path("profiles")
OUTDIR.mkdir(exist_ok=True)

def measure_fit(X, n_components=5, cov_type="full",
                max_iter=50, n_init=1, init_params="random", tol=1e-3,
                warmup_iter=2):
    """Measure wall-time of a single fit() call with sane settings."""
    print(
        f"[{cov_type:9s}] setup: N={X.shape[0]}, D={X.shape[1]}, "
        f"K={n_components}, max_iter={max_iter}, n_init={n_init}, init={init_params}",
        flush=True,
    )

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        max_iter=max_iter,
        n_init=n_init,
        init_params=init_params,
        tol=tol,
        random_state=42,
        verbose=2,              # show progress
        verbose_interval=10,
    )

    # Warmup: small number of iterations to warm BLAS/cache, not a full run
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

    print(f"[{cov_type:9s}] measuring...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        gmm.fit(X)
        t1 = time.perf_counter()

    wall = t1 - t0
    print(f"[{cov_type:9s}] wall={wall:.6f} s | converged={gmm.converged_} | n_iter={gmm.n_iter_}\n", flush=True)
    return wall

if __name__ == "__main__":
    print("scikit-learn GaussianMixture wall-time measurement\n")

    # Generate ONCE so each covariance type sees identical data
    n_samples = 100000
    n_features = 200
    n_components = 5

    print("[data] generating...", flush=True)
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    sklearn_times = {}
    for cov_type in ["diag", "spherical", "tied", "full"]:
        sklearn_times[cov_type] = measure_fit(
            X,
            n_components=n_components,
            cov_type=cov_type,
            max_iter=50,          # <- adjust
            n_init=1,             # <- IMPORTANT
            init_params="random", # <- avoids kmeans dominating runtime
            tol=1e-3,
            warmup_iter=2,
        )

    times_path = OUTDIR / "sklearn_runtimes.json"
    with open(times_path, "w") as f:
        json.dump(sklearn_times, f, indent=2)
    print(f"\nExported sklearn runtimes to: {times_path}")
