from pathlib import Path
import re
import pstats
import pandas as pd

PROF_DIR = Path("profiles")

def safe_sheet_name(name: str) -> str:
    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)
    return name[:31]

def pstats_to_df(pstats_file: Path) -> pd.DataFrame:
    st = pstats.Stats(str(pstats_file))
    rows = []
    for func, (cc, nc, tt, ct, callers) in st.stats.items():
        filename, line, fn = func
        rows.append({
            "file": filename,
            "line": line,
            "function": fn,
            "ncalls": nc,
            "primitive_calls": cc,
            "tottime_s": tt,
            "cumtime_s": ct,
        })
    df = pd.DataFrame(rows).sort_values("cumtime_s", ascending=False).reset_index(drop=True)
    return df

def get_sklearn_total_time(df: pd.DataFrame) -> float:
    """Extract total cumulative time from sklearn pstats DataFrame."""
    if "cumtime_s" in df.columns:
        return float(df["cumtime_s"].sum())
    return 0.0

def main():
    if not PROF_DIR.exists():
        raise SystemExit("No ./profiles directory found")

    pstats_files = sorted(PROF_DIR.glob("*.pstats"))
    torch_csvs   = sorted([p for p in PROF_DIR.glob("torch_*.csv") if p.name != "torch_summary.csv"])

    if not pstats_files and not torch_csvs:
        raise SystemExit("No .pstats or torch_*.csv files found in ./profiles")

    out_xlsx = PROF_DIR / "profile_tables.xlsx"

    sklearn_totals = {}
    torch_totals = {}

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        index_rows = []

        # scikit / cProfile runs
        for psfile in pstats_files:
            df = pstats_to_df(psfile)
            sheet = safe_sheet_name(psfile.stem)
            df.to_excel(w, sheet_name=sheet, index=False)
            index_rows.append({"sheet": sheet, "source": "pstats", "file": psfile.name})
            # Extract cov_type from filename (e.g., sklearn_diag_N10000_D50_K5 -> diag)
            cov_type = psfile.stem.split("_")[1]  # sklearn_{cov_type}_...
            sklearn_totals[cov_type] = get_sklearn_total_time(df)

        # torch runs (already CSV tables)
        for csvfile in torch_csvs:
            df = pd.read_csv(csvfile)
            sheet = safe_sheet_name(csvfile.stem)
            df.to_excel(w, sheet_name=sheet, index=False)
            index_rows.append({"sheet": sheet, "source": "torch_csv", "file": csvfile.name})
            # Extract cov_type from filename (e.g., torch_diag_N10000_D50_K5 -> diag)
            cov_type = csvfile.stem.split("_")[1]  # torch_{cov_type}_...
            if "gpu_time_total_us" in df.columns:
                torch_totals[cov_type] = df["gpu_time_total_us"].sum() / 1_000_000  # us -> seconds
            elif "cpu_time_total_us" in df.columns:
                torch_totals[cov_type] = df["cpu_time_total_us"].sum() / 1_000_000

        # Add sklearn summary sheet
        if sklearn_totals:
            sklearn_summary = pd.DataFrame([
                {"covariance_type": cov, "total_cumtime_s": t}
                for cov, t in sorted(sklearn_totals.items())
            ])
            sklearn_summary.to_excel(w, sheet_name="sklearn_totals", index=False)
            index_rows.append({"sheet": "sklearn_totals", "source": "summary", "file": "sklearn_totals"})

        # Add torch summary sheet
        if torch_totals:
            torch_summary = pd.DataFrame([
                {"covariance_type": cov, "total_gpu_time_s": t}
                for cov, t in sorted(torch_totals.items())
            ])
            torch_summary.to_excel(w, sheet_name="torch_totals", index=False)
            index_rows.append({"sheet": "torch_totals", "source": "summary", "file": "torch_totals"})

        pd.DataFrame(index_rows).to_excel(w, sheet_name="INDEX", index=False)

    print("Wrote:", out_xlsx)

if __name__ == "__main__":
    main()
