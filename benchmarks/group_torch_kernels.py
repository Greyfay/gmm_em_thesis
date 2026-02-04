from pathlib import Path
import re
import json
import pandas as pd

IN_XLSX  = Path("profiles/profile_tables.xlsx")
OUT_XLSX = Path("profiles/profile_tables_grouped.xlsx")
BASELINE_FILE = Path("profiles/torch_baselines.json")

# Ordered patterns: first match wins.
# You can extend this list as you see new kernel names.
PATTERNS = [
    # --- Big buckets you care about ---
    ("matmul_gemm", [
        r"\bgemm\b", r"cublas", r"cutlass", r"mma", r"mm\b", r"bmm\b",
        r"addmm", r"matmul", r"aten::mm", r"aten::bmm", r"aten::matmul",
        r"sgemm", r"volta_.*gemm",
    ], "MatMul / GEMM (linear algebra)"),

    ("tensor_create_init", [
        r"aten::zeros\b", r"aten::ones\b", r"aten::empty\b", r"aten::empty_strided\b",
        r"aten::full\b", r"aten::ones_like\b", r"aten::arange\b",
        r"aten::empty_like\b", r"aten::eye\b",
    ], "Tensor creation / init"),

    ("distance", [
        r"aten::cdist\b", r"aten::_euclidean_dist\b",
    ], "Distance computation"),

    ("controlflow_sync", [
        r"aten::is_nonzero\b",
    ], "CPU–GPU sync / control-flow"),

    ("triangular_ops", [
        r"aten::tril_\b", r"triu_tril_kernel",
    ], "Triangular ops (tril/triu)"),

    ("linalg_cholesky_info", [
        r"xxtrf4_set_info_ker",
    ], "Linear algebra: Cholesky (info/status)"),

    ("linalg_cholesky", [
        r"aten::linalg_cholesky\b", r"aten::linalg_cholesky_ex\b",
    ], "Linear algebra: Cholesky"),

    ("linalg_triangular_solve", [
        r"aten::linalg_solve_triangular\b",
    ], "Linear algebra: triangular solve"),

    ("linalg_checks", [
        r"aten::_linalg_check_errors\b",
    ], "Linear algebra: checks"),

    ("concat_stack", [
        r"aten::stack\b", r"aten::cat\b",
    ], "Tensor concatenation / stacking"),

    ("diag_extract", [
        r"aten::diagonal\b",
    ], "View/Indexing (diagonal)"),

    ("reduce_minmax_arg", [
        r"aten::argmin\b", r"aten::argmax\b",
        r"aten::max\b", r"aten::min\b",
    ], "Reduction (min/max/argmin/argmax)"),

    ("memory_allocator", [
        r"^\[memory\]$",
    ], "Memory / allocator overhead"),

    ("mask_index", [
        r"aten::nonzero\b",
    ], "Masking / indexing"),

    ("mask_index_compaction", [
        r"cub::DeviceCompactInitKernel", r"DeviceCompactInitKernel",
    ], "Masking / indexing (CUB compaction)"),

    ("bool_reduce", [
        r"aten::any\b",
    ], "Reduction (boolean)"),

    ("compare_mask", [
        r"aten::eq\b", r"aten::lt\b", r"aten::le\b", r"aten::gt\b", r"aten::ge\b",
        r"aten::bitwise_and\b", r"aten::bitwise_not\b",
    ], "Comparisons / Masking"),

    ("debug_assert", [
        r"_assert_async_cuda_kernel", r"aten::_assert_async\b",
    ], "Debug / assert overhead"),

    ("cuda_runtime", [
        r"cudaPeekAtLastError", r"cudaoccupancy", r"cudapeekatenderror",
        r"cudadevicegetattribute",
    ], "CUDA runtime / launch overhead"),

    ("elementwise_log", [
        r"aten::log_\b",
    ], "Elementwise math (log)"),

    ("elementwise_exp", [
        r"aten::exp_\b",
    ], "Elementwise math (exp)"),

    ("elementwise_reciprocal", [
        r"aten::reciprocal\b",
    ], "Elementwise math (reciprocal)"),

    ("cpu_gpu_sync", [
        r"aten::item\b", r"aten::_local_scalar_dense\b",
    ], "CPU–GPU synchronization"),

    ("fill_reset", [
        r"aten::fill_\b", r"aten::zero_\b",
    ], "Fill / Reset (in-place)"),

    ("rng", [
        r"aten::exponential_\b", r"philox", r"random", r"curand",
        r"rand\b", r"dropout",
    ], "Random number generation"),

    ("view_stride_index", [
        r"aten::expand\b", r"aten::t\b", r"aten::mt\b", r"aten::numpy_t\b",
        r"aten::slice\b",
    ], "View/Stride/Indexing"),

    ("autograd_meta", [
        r"aten::detach_\b", r"\bdetach_\b", r"aten::lift_fresh\b",
        r"aten::result_type\b", r"aten::resize_\b", r"aten::set_\b",
    ], "Autograd / metadata / bookkeeping"),

    ("profiler_overhead", [
        r"activity buffer request", r"buffer flush",
    ], "Profiler overhead"),

    ("type_numeric_util", [
        r"aten::real\b",
    ], "Type / numeric utility"),

    ("reduce", [
        r"reduce_kernel", r"reduce", r"cub::DeviceReduce", r"block_reduce",
        r"aten::sum", r"aten::mean", r"aten::amax", r"aten::amin", r"aten::prod",
    ], "Reduction (sum/mean/max/…)"),

    ("softmax_logsumexp", [
        r"logsumexp", r"softmax", r"log_softmax",
    ], "Softmax / LogSumExp"),

    ("elementwise", [
        r"elementwise", r"TensorIterator", r"pointwise", r"vectorized_elementwise",
        r"aten::add", r"aten::sub", r"aten::mul", r"aten::div",
        r"aten::add_", r"aten::mul_", r"aten::div_",
        r"aten::sqrt", r"aten::rsqrt", r"aten::pow",
        r"aten::clamp", r"aten::maximum", r"aten::minimum",
        r"aten::where", r"aten::abs",
    ], "Elementwise ops (add/mul/div/…)"),

    ("broadcast_index", [
        r"\bindex\b", r"gather", r"scatter", r"\btake\b", r"select",
        r"masked", r"advanced_index", r"index_select",
    ], "Indexing / Gather / Scatter"),

    ("transpose_view_reshape", [
        r"transpose", r"permute", r"\bview\b", r"reshape", r"contiguous",
        r"as_strided", r"flatten", r"squeeze", r"unsqueeze",
    ], "View/Reshape/Transpose/Contiguous"),

    ("norm_stats", [
        r"\bvar\b", r"variance", r"\bstd\b", r"\bnorm\b", r"layer_norm", r"batch_norm",
    ], "Norm / Statistics"),

    ("memory_copy", [
        r"memcpy", r"memset", r"\bcopy\b", r"dtoh", r"htod", r"cudaMemcpy",
        r"aten::copy_", r"aten::to", r"pin_memory",
    ], "Memory copy / Transfers"),

    ("alloc_free", [
        r"malloc", r"\bfree\b", r"alloc", r"cudaMalloc", r"cudaFree",
        r"caching_allocator", r"empty_cache",
    ], "Allocation / Free"),

    ("sync_launch", [
        r"cudaLaunchKernel", r"cudaDeviceSynchronize", r"synchronize",
        r"cudaGetLastError", r"cudaStream", r"cudaEvent",
    ], "Kernel launch / Sync / Runtime"),

    ("cuda_unspecified_kernel", [
        r"cuda_kernel", r"kernel",
    ], "CUDA kernel (unspecified/fused)"),
]

def classify(name: str):
    if not isinstance(name, str) or not name:
        return ("unknown", "Unknown / Other")

    lower = name.lower()
    for group, pats, label in PATTERNS:
        for p in pats:
            if re.search(p.lower(), lower):
                return (group, label)
    return ("unknown", "Unknown / Other")

def shorten(name: str, maxlen: int = 110) -> str:
    if not isinstance(name, str):
        return ""
    # Keep the left-most part and cut templates a bit
    s = re.sub(r"<.*>", "<…>", name)  # collapse template args
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= maxlen:
        return s
    return s[:maxlen - 1] + "…"

def is_torch_sheet(df: pd.DataFrame) -> bool:
    cols = set(df.columns.str.lower())
    return ("name" in cols) and (("gpu_time_total_us" in cols) or ("device_time_total_us" in cols) or ("cuda_time_total_us" in cols))

def pick_time_column(df: pd.DataFrame) -> str:
    # prefer unified field if you already created it, else fall back
    for c in ["gpu_time_total_us", "device_time_total_us", "cuda_time_total_us", "cpu_time_total_us"]:
        if c in df.columns:
            return c
    raise KeyError("No time column found")

def load_baselines() -> dict:
    """Load torch baseline times from JSON, if available."""
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return {}

def main():
    if not IN_XLSX.exists():
        raise SystemExit(f"Missing {IN_XLSX}")

    xls = pd.ExcelFile(IN_XLSX)
    sheet_names = xls.sheet_names

    # Load baseline times and totals from summaries
    torch_baselines = load_baselines()
    sklearn_totals = {}
    torch_gpu_totals = {}

    # Try to read summary sheets from IN_XLSX
    try:
        sklearn_sum_df = pd.read_excel(IN_XLSX, sheet_name="sklearn_totals")
        sklearn_totals = dict(zip(sklearn_sum_df["covariance_type"], sklearn_sum_df["total_cumtime_s"]))
    except Exception:
        pass

    try:
        torch_sum_df = pd.read_excel(IN_XLSX, sheet_name="torch_totals")
        torch_gpu_totals = dict(zip(torch_sum_df["covariance_type"], torch_sum_df["total_gpu_time_s"]))
    except Exception:
        pass

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        index_rows = []

        for sh in sheet_names:
            df = pd.read_excel(IN_XLSX, sheet_name=sh)

            # Keep original sheets as-is if not torch-format
            if not is_torch_sheet(df):
                df.to_excel(w, sheet_name=sh[:31], index=False)
                index_rows.append({"sheet": sh[:31], "type": "passthrough"})
                continue

            time_col = pick_time_column(df)

            # Add readable columns
            df["op_group"], df["op_label"] = zip(*df["name"].map(classify))
            df["name_short"] = df["name"].map(shorten)

            # Reorder columns to keep things readable in Excel
            preferred = ["op_label", "op_group", "name_short", "name", time_col, "calls"]
            cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
            df = df[cols]

            # Write detailed sheet
            out_sheet = sh[:28] + "_raw"  # reserve room for suffix
            df.to_excel(w, sheet_name=out_sheet[:31], index=False)
            index_rows.append({"sheet": out_sheet[:31], "type": "torch_raw", "source": sh})

            # Build summary (grouped)
            # Convert to numeric safely (Excel import can sometimes turn to object)
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0)

            summary = (
                df.groupby(["op_label", "op_group"], as_index=False)[time_col]
                  .sum()
                  .sort_values(time_col, ascending=False)
                  .reset_index(drop=True)
            )
            total = summary[time_col].sum()
            summary["percent"] = (summary[time_col] / total * 100.0) if total > 0 else 0.0

            # Make time human-friendly (microseconds → milliseconds)
            if time_col.endswith("_us"):
                summary["time_ms"] = summary[time_col] / 1000.0
            else:
                summary["time_ms"] = summary[time_col]

            # Keep summary columns neat
            summary = summary[["op_label", "op_group", "time_ms", "percent"]]

            sum_sheet = sh[:28] + "_sum"
            summary.to_excel(w, sheet_name=sum_sheet[:31], index=False)
            index_rows.append({"sheet": sum_sheet[:31], "type": "torch_summary", "source": sh})

        # Create comparison sheet with sklearn vs pytorch baseline times + variance
        comparison_rows = []
        
        # Load baseline statistics JSON files
        torch_baseline_stats_file = Path("profiles/torch_baselines_stats.json")
        sklearn_runtime_stats_file = Path("profiles/sklearn_runtimes_stats.json")
        
        torch_stats_dict = {}
        sklearn_stats_dict = {}
        
        if torch_baseline_stats_file.exists():
            with open(torch_baseline_stats_file) as f:
                torch_stats_dict = json.load(f)
        
        if sklearn_runtime_stats_file.exists():
            with open(sklearn_runtime_stats_file) as f:
                sklearn_stats_dict = json.load(f)
        
        for cov_type in sorted(set(torch_stats_dict.keys()) | set(sklearn_stats_dict.keys())):
            sklearn_data = sklearn_stats_dict.get(cov_type, {})
            torch_data = torch_stats_dict.get(cov_type, {})
            
            sklearn_mean = sklearn_data.get("mean")
            sklearn_std = sklearn_data.get("std")
            torch_mean = torch_data.get("mean")
            torch_std = torch_data.get("std")
            
            speedup = None
            if sklearn_mean is not None and torch_mean is not None and torch_mean > 0:
                speedup = sklearn_mean / torch_mean

            comparison_rows.append({
                "covariance_type": cov_type,
                "sklearn_time_mean_s": sklearn_mean,
                "sklearn_time_std_s": sklearn_std,
                "torch_time_mean_s": torch_mean,
                "torch_time_std_s": torch_std,
                "speedup_ratio": speedup,
            })

        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            comparison_df.to_excel(w, sheet_name="COMPARISON", index=False)
            index_rows.append({"sheet": "COMPARISON", "type": "comparison"})

        pd.DataFrame(index_rows).to_excel(w, sheet_name="INDEX", index=False)

    print(f"Wrote: {OUT_XLSX}")

if __name__ == "__main__":
    main()
