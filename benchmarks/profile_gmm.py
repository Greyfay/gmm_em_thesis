"""
Profile TorchGaussianMixture to identify bottlenecks.

Fixes applied:
- Correct wall-time measurement: torch.cuda.synchronize() before/after.
- Avoid expensive trace export by default (optional deep-trace mode).
- Avoid deprecated cuda_time fields: prefer device_time_* when available.
- Print tables sorted by a supported key (device_time_total preferred).
- Export stable CSV/XLSX with unified gpu_time_* columns.
- Optionally measure baseline (no profiler) wall time for comparison.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import torch
import pandas as pd
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation._torch_gmm_em import TorchGaussianMixture  # noqa: E402

OUTDIR = Path("profiles")
OUTDIR.mkdir(exist_ok=True)

TRACEDIR = Path("results/profiles")
TRACEDIR.mkdir(parents=True, exist_ok=True)

# Track baseline times for comparison export
BASELINE_TIMES = {}  # Will store all 10 measurements
BASELINE_STATS = {}  # Will store mean/std
SKLEARN_STATS = {}  # Will be loaded from sklearn_runtimes_stats.json


def load_sklearn_stats() -> dict:
    """Load sklearn runtime statistics from JSON, if available."""
    sklearn_file = OUTDIR / "sklearn_runtimes_stats.json"
    if sklearn_file.exists():
        with open(sklearn_file) as f:
            return json.load(f)
    return {}


def _pick_sort_key_for_table() -> str:
    """
    Choose a profiler table sort key that works across PyTorch versions.
    Prefer device_time_total (new), fall back to cuda_time_total (older).
    """
    # We can't introspect keys without a prof instance, so we return the preferred key
    # and later fall back if printing fails.
    return "device_time_total"


def prof_to_df(prof, key_avg=None) -> pd.DataFrame:
    """
    Convert torch.profiler output to a DataFrame with unified GPU timing columns.
    Pass pre-computed key_avg to avoid redundant aggregation on large traces.
    Prefer device_time_* (modern), fall back to cuda_time_* (legacy).
    """
    if key_avg is None:
        key_avg = prof.key_averages()
    
    events = key_avg
    rows = []

    for e in events:
        # CPU timing (stable across versions)
        cpu_total = getattr(e, "cpu_time_total", None)
        cpu_avg = getattr(e, "cpu_time", None)  # avg per call (us)
        self_cpu = getattr(e, "self_cpu_time_total", None)

        # Prefer device-time fields (newer PyTorch)
        device_total = getattr(e, "device_time_total", None)
        device_avg = getattr(e, "device_time", None)
        self_device = getattr(e, "self_device_time_total", None)

        # Legacy CUDA-time fields (older PyTorch)
        cuda_total = getattr(e, "cuda_time_total", None)
        cuda_avg = getattr(e, "cuda_time", None)  # may warn deprecated in some versions
        self_cuda = getattr(e, "self_cuda_time_total", None)

        # Use device_* if present, else fall back to cuda_*
        gpu_total = device_total if device_total is not None else cuda_total
        gpu_avg = device_avg if device_avg is not None else cuda_avg
        self_gpu = self_device if self_device is not None else self_cuda

        rows.append(
            {
                "name": e.key,
                "calls": getattr(e, "count", None),
                "cpu_time_total_us": cpu_total,
                "cpu_time_avg_us": cpu_avg,
                "self_cpu_time_total_us": self_cpu,
                "gpu_time_total_us": gpu_total,
                "gpu_time_avg_us": gpu_avg,
                "self_gpu_time_total_us": self_gpu,
                # Memory fields are version-dependent; keep if available
                "cpu_memory_usage_bytes": getattr(e, "cpu_memory_usage", None),
                "cuda_memory_usage_bytes": getattr(e, "cuda_memory_usage", None),
                "input_shapes": str(getattr(e, "input_shapes", "")),
            }
        )

    df = pd.DataFrame(rows)

    # Sort by whatever we actually have
    if "gpu_time_total_us" in df.columns and df["gpu_time_total_us"].notna().any():
        sort_col = "gpu_time_total_us"
    else:
        sort_col = "cpu_time_total_us"

    df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce").fillna(0)
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df


def _wall_time_fit(gmm: TorchGaussianMixture, X: torch.Tensor) -> float:
    """Measure true wall time of gmm.fit(X) with CUDA synchronization."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gmm.fit(X)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0


def profile_fit(
    X: Optional[torch.Tensor] = None,
    n_samples: int = 100000,
    n_features: int = 500,
    n_components: int = 5,
    cov_type: str = "full",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    enable_trace: bool = False,
    trace_dir: Optional[Path] = None,
    measure_baseline: bool = True,
    num_runs: int = 10,
    run_seed: int = 42,
) -> None:
    """
    Profile a single fit() call across multiple runs.

    - enable_trace=False by default to avoid huge slowdowns from trace export.
    - measure_baseline=True measures baseline wall time without profiler.
    - num_runs: number of baseline measurements to average (default 10).
    - run_seed: random seed for data generation (different per covariance type).
    - Profiler is run once after baselines are collected.
    """
    assert device in ("cuda", "cpu"), f"Unsupported device={device}"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if X is None:
        # Set seed for reproducible but different data per run
        torch.manual_seed(run_seed)
        if device == "cuda":
            torch.cuda.manual_seed(run_seed)
        X = torch.randn(n_samples, n_features, device=device, dtype=dtype)
    else:
        n_samples, n_features = X.shape

    gmm = TorchGaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        max_iter=50,
        n_init=1,
        init_params="random",
        device=device,
        dtype=dtype,
    )

    # Warmup (do not time / do not profile)
    gmm.fit(X)

    # Baseline measurements (no profiler)
    baseline_times = []
    if measure_baseline:
        print(f"[{cov_type:9s}] Collecting {num_runs} baseline measurements...", flush=True)
        for run in range(num_runs):
            # Generate NEW random data for each run with different seed
            torch.manual_seed(run_seed + run)
            if device == "cuda":
                torch.cuda.manual_seed(run_seed + run)
            X_run = torch.randn(n_samples, n_features, device=device, dtype=dtype)
            
            t_run = _wall_time_fit(gmm, X_run)
            baseline_times.append(t_run)
            print(f"  run {run+1}/{num_runs}: {t_run:.6f} s", flush=True)
        
        mean_baseline = np.mean(baseline_times)
        std_baseline = np.std(baseline_times)
        print(f"[baseline] cov={cov_type:9s} mean={mean_baseline:.6f} s | std={std_baseline:.6f} s")
        
        # Store for later export
        BASELINE_TIMES[cov_type] = baseline_times
        BASELINE_STATS[cov_type] = {"mean": float(mean_baseline), "std": float(std_baseline)}

    # Profiler config
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    on_trace_ready = None
    if enable_trace:
        # Deep-trace mode should be used only for small runs.
        tdir = trace_dir if trace_dir is not None else (TRACEDIR / cov_type)
        tdir.mkdir(parents=True, exist_ok=True)
        on_trace_ready = tensorboard_trace_handler(str(tdir))

    # Measure wall time of the profiled region (with sync)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        on_trace_ready=on_trace_ready,
    ) as prof:
        gmm.fit(X)

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    profiled_wall_s = t1 - t0

    print(f"[profiled ] cov={cov_type:9s} wall={profiled_wall_s:.6f} s (with profiler)")

    # Export tables
    tag = f"torch_{cov_type}_N{n_samples}_D{n_features}_K{n_components}"
    out_csv = OUTDIR / f"{tag}.csv"
    out_xlsx = OUTDIR / f"{tag}.xlsx"

    print("\n=== Profile: {}, N={}, D={}, K={} ===".format(cov_type, n_samples, n_features, n_components))
    print("Processing profiler events (aggregating and sorting)...")
    
    # Compute key_averages once and reuse
    key_avg = prof.key_averages()
    
    # Get sklearn runtime for this cov_type if available
    sklearn_time = SKLEARN_TIMES.get(cov_type, None)
    
    df = prof_to_df(prof, key_avg)
    
    # Add sklearn runtime as a comparison column
    if sklearn_time is not None:
        df["sklearn_time_s"] = sklearn_time
    
    df.to_csv(out_csv, index=False)
    df.to_excel(out_xlsx, index=False)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_xlsx}")

    # Print top ops using the already-computed key_avg (no re-computation)
    sort_key = _pick_sort_key_for_table()
    try:
        print(key_avg.table(sort_by=sort_key, row_limit=20))
    except Exception:
        # Older versions may not have device_time_total; fall back
        print(key_avg.table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    # Load sklearn runtime statistics
    SKLEARN_STATS = load_sklearn_stats()
    if SKLEARN_STATS:
        print("Loaded sklearn runtime statistics:")
        for cov_type, stats in SKLEARN_STATS.items():
            print(f"  {cov_type:12s}: mean={stats['mean']:.6f} s | std={stats['std']:.6f} s")
        print()
    else:
        print("Warning: sklearn_runtimes_stats.json not found. Run profile_gmm_scikit.py first.\n")
    
    # Profile each covariance type with different random data
    for run_id, cov_type in enumerate(["diag", "spherical", "tied", "full"]):
        print(f"\n{'='*60}")
        print(f"Profiling {cov_type}")
        print(f"{'='*60}")
        
        profile_fit(
            X=None,  # Generate new random data
            cov_type=cov_type,
            enable_trace=False,
            measure_baseline=True,
            num_runs=10,
            run_seed=42 + run_id,  # Different seed per covariance type
        )

    # Export baseline statistics
    baseline_stats_path = OUTDIR / "torch_baselines_stats.json"
    with open(baseline_stats_path, "w") as f:
        json.dump(BASELINE_STATS, f, indent=2)
    print(f"\nExported torch baseline statistics to: {baseline_stats_path}")

    # If you ever want a deep trace for one small run, do something like:
    # profile_fit(cov_type="full", n_samples=2000, n_features=20, n_components=5,
    #             enable_trace=True, measure_baseline=False)
