from pathlib import Path
import re
import pandas as pd

IN_XLSX  = Path("profiles/profile_tables.xlsx")
OUT_XLSX = Path("profiles/profile_tables_grouped.xlsx")

# Ordered patterns: first match wins.
# You can extend this list as you see new kernel names.
PATTERNS = [
    # --- Big buckets you care about ---
    ("matmul_gemm", [
        r"\bgemm\b", r"cublas", r"cutlass", r"mma", r"mm\b", r"bmm\b",
        r"addmm", r"matmul", r"aten::mm", r"aten::bmm", r"aten::matmul",
    ], "MatMul / GEMM (linear algebra)"),

    ("reduce", [
        r"reduce_kernel", r"reduce", r"cub::DeviceReduce", r"block_reduce",
        r"aten::sum", r"aten::mean", r"aten::amax", r"aten::amin", r"aten::prod",
    ], "Reduction (sum/mean/max/…)"),

    ("softmax_logsumexp", [
        r"logsumexp", r"softmax", r"log_softmax",
    ], "Softmax / LogSumExp"),

    ("exp_log", [
        r"\bexp\b", r"\blog\b", r"log1p", r"expm1",
    ], "Exp / Log"),

    ("elementwise", [
        r"elementwise", r"TensorIterator", r"pointwise", r"vectorized_elementwise",
        r"aten::add", r"aten::sub", r"aten::mul", r"aten::div",
        r"aten::add_", r"aten::mul_", r"aten::div_",
        r"aten::sqrt", r"aten::rsqrt", r"aten::pow",
        r"aten::clamp", r"aten::maximum", r"aten::minimum",
        r"aten::where", r"aten::abs",
    ], "Elementwise ops (add/mul/div/…)"),

    ("broadcast_index", [
        r"index", r"gather", r"scatter", r"take", r"select",
        r"masked", r"advanced_index", r"index_select",
    ], "Indexing / Gather / Scatter"),

    ("transpose_view_reshape", [
        r"transpose", r"permute", r"view", r"reshape", r"contiguous",
        r"as_strided", r"flatten", r"squeeze", r"unsqueeze",
    ], "View/Reshape/Transpose/Contiguous"),

    ("norm_stats", [
        r"var", r"variance", r"std", r"norm", r"layer_norm", r"batch_norm",
    ], "Norm / Statistics"),

    ("rng", [
        r"philox", r"random", r"curand", r"rand", r"dropout",
    ], "Random number generation"),

    ("memory_copy", [
        r"memcpy", r"memset", r"copy", r"dtoh", r"htod", r"cudaMemcpy",
        r"aten::copy_", r"aten::to", r"pin_memory",
    ], "Memory copy / Transfers"),

    ("alloc_free", [
        r"malloc", r"free", r"alloc", r"cudaMalloc", r"cudaFree",
        r"caching_allocator", r"empty_cache",
    ], "Allocation / Free"),

    ("sync_launch", [
        r"cudaLaunchKernel", r"cudaDeviceSynchronize", r"synchronize",
        r"cudaGetLastError", r"cudaStream", r"cudaEvent",
    ], "Kernel launch / Sync / Runtime"),
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

def main():
    if not IN_XLSX.exists():
        raise SystemExit(f"Missing {IN_XLSX}")

    xls = pd.ExcelFile(IN_XLSX)
    sheet_names = xls.sheet_names

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

        pd.DataFrame(index_rows).to_excel(w, sheet_name="INDEX", index=False)

    print(f"Wrote: {OUT_XLSX}")

if __name__ == "__main__":
    main()
