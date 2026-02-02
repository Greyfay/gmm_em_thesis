"""Profile TorchGaussianMixture to identify bottlenecks."""

import sys
import os
from pathlib import Path
import torch
import pandas as pd
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation._torch_gmm_em import TorchGaussianMixture

OUTDIR = Path("profiles")
OUTDIR.mkdir(exist_ok=True)

TRACEDIR = Path("results/profiles")
TRACEDIR.mkdir(parents=True, exist_ok=True)

def prof_to_df(prof):
    events = prof.key_averages()
    rows = []

    for e in events:
        # CPU times are stable
        cpu_total = getattr(e, "cpu_time_total", None)
        cpu_avg   = getattr(e, "cpu_time", None)  # avg per call (us)
        self_cpu  = getattr(e, "self_cpu_time_total", None)

        # GPU/device times vary by torch version:
        # try CUDA-specific, else device-generic
        cuda_total = getattr(e, "cuda_time_total", None)
        cuda_avg   = getattr(e, "cuda_time", None)
        self_cuda  = getattr(e, "self_cuda_time_total", None)

        device_total = getattr(e, "device_time_total", None)
        device_avg   = getattr(e, "device_time", None)
        self_device  = getattr(e, "self_device_time_total", None)

        # Prefer CUDA fields if present; otherwise use device fields
        gpu_total = cuda_total if cuda_total is not None else device_total
        gpu_avg   = cuda_avg   if cuda_avg   is not None else device_avg
        self_gpu  = self_cuda  if self_cuda  is not None else self_device

        rows.append({
            "name": e.key,
            "calls": getattr(e, "count", None),

            "cpu_time_total_us": cpu_total,
            "cpu_time_avg_us": cpu_avg,
            "self_cpu_time_total_us": self_cpu,

            # keep both, but "gpu_*" is the unified one you’ll sort by
            "gpu_time_total_us": gpu_total,
            "gpu_time_avg_us": gpu_avg,
            "self_gpu_time_total_us": self_gpu,

            # Optional memory fields (also version-dependent)
            "cpu_memory_usage_bytes": getattr(e, "cpu_memory_usage", None),
            "cuda_memory_usage_bytes": getattr(e, "cuda_memory_usage", None),
            "input_shapes": str(getattr(e, "input_shapes", "")),
        })

    import pandas as pd
    df = pd.DataFrame(rows)

    # Sort by whatever we actually have
    sort_col = "gpu_time_total_us" if df["gpu_time_total_us"].notna().any() else "cpu_time_total_us"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df


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
    )

    # Warmup
    gmm.fit(X)

    trace_path = TRACEDIR / cov_type
    trace_path.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=tensorboard_trace_handler(str(trace_path)),
    ) as prof:
        gmm.fit(X)

    tag = f"torch_{cov_type}_N{n_samples}_D{n_features}_K{n_components}"
    out_csv = OUTDIR / f"{tag}.csv"
    out_xlsx = OUTDIR / f"{tag}.xlsx"

    df = prof_to_df(prof)
    df.to_csv(out_csv, index=False)
    df.to_excel(out_xlsx, index=False)

    print(f"\n=== Profile: {cov_type}, N={n_samples}, D={n_features}, K={n_components} ===")
    print("Saved:", out_csv)
    print("Saved:", out_xlsx)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    for cov_type in ["diag", "spherical", "tied", "full"]:
        profile_fit(cov_type=cov_type)
