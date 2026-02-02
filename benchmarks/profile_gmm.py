"""Profile TorchGaussianMixture to identify bottlenecks."""

import sys
import os
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation._torch_gmm_em import TorchGaussianMixture


def profile_fit(n_samples=10000, n_features=50, n_components=5, cov_type="full"):
    """Profile a single fit() call."""
    X = torch.randn(n_samples, n_features, device="cuda", dtype=torch.float32)
    
    gmm = TorchGaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        max_iter=50,
        n_init=10,
        init_params="kmeans",
        device="cuda",
        dtype=torch.float32,
        random_state=42,
    )

    # Warmup
    gmm.fit(X)

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=tensorboard_trace_handler(f"./results/profiles/{cov_type}"),
    ) as prof:
        gmm.fit(X)

    # Print summary
    print(f"\n=== Profile: {cov_type}, N={n_samples}, D={n_features}, K={n_components} ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    for cov_type in ["diag", "spherical", "tied", "full"]:
        profile_fit(cov_type=cov_type)