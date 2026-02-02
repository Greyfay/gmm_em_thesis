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

def prof_to_df(prof) -> pd.DataFrame:
    # key_averages() returns an EventList of FunctionEventAvg
    events = prof.key_averages()
    rows = []
    for e in events:
        rows.append({
            "name": e.key,
            "calls": e.count,
            "cpu_time_total_us": e.cpu_time_total,
            "cpu_time_avg_us": e.cpu_time,           # average per call (us)
            "cuda_time_total_us": e.cuda_time_total,
            "cuda_time_avg_us": e.cuda_time,         # average per call (us)
            "self_cpu_time_total_us": e.self_cpu_time_total,
            "self_cuda_time_total_us": getattr(e, "self_cuda_time_total", None),
            "cpu_memory_usage_bytes": getattr(e, "cpu_memory_usage", None),
            "cuda_memory_usage_bytes": getattr(e, "cuda_memory_usage", None),
            "input_shapes": str(getattr(e, "input_shapes", "")),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("cuda_time_total_us", ascending=False).reset_index(drop=True)
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
