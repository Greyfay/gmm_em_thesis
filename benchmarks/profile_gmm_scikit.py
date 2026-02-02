"""Profile scikit-learn GaussianMixture to compare with torch implementation."""

import sys
import os
import warnings
import numpy as np
from sklearn.mixture import GaussianMixture
import time
import cProfile
import pstats
from io import StringIO
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTDIR = Path("profiles")
OUTDIR.mkdir(exist_ok=True)

def profile_fit(n_samples=10000, n_features=50, n_components=5, cov_type="full"):
    """Profile a single fit() call."""
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        max_iter=50,
        n_init=10,
        init_params="kmeans",
        random_state=42,   
    )

    # Warmup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm.fit(X)

    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm.fit(X)
    profiler.disable()

    # save .pstats for export
    tag = f"sklearn_{cov_type}_N{n_samples}_D{n_features}_K{n_components}"
    pstats_path = OUTDIR / f"{tag}.pstats"
    profiler.dump_stats(str(pstats_path))

    # Print summary
    print(f"\n{'='*70}")
    print(f"Profile: {cov_type:12s} | N={n_samples:5d} | D={n_features:2d} | K={n_components}")
    print(f"Saved:  {pstats_path}")
    print(f"{'='*70}")
    
    # Sort by cumulative time and print top 30 functions
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    print("scikit-learn GaussianMixture Benchmark")
    for cov_type in ["diag", "spherical", "tied", "full"]:
        profile_fit(cov_type=cov_type)