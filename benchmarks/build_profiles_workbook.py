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

def main():
    if not PROF_DIR.exists():
        raise SystemExit("No ./profiles directory found")

    pstats_files = sorted(PROF_DIR.glob("*.pstats"))
    torch_csvs   = sorted([p for p in PROF_DIR.glob("torch_*.csv") if p.name != "torch_summary.csv"])

    if not pstats_files and not torch_csvs:
        raise SystemExit("No .pstats or torch_*.csv files found in ./profiles")

    out_xlsx = PROF_DIR / "profile_tables.xlsx"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        index_rows = []

        # scikit / cProfile runs
        for psfile in pstats_files:
            df = pstats_to_df(psfile)
            sheet = safe_sheet_name(psfile.stem)
            df.to_excel(w, sheet_name=sheet, index=False)
            index_rows.append({"sheet": sheet, "source": "pstats", "file": psfile.name})

        # torch runs (already CSV tables)
        for csvfile in torch_csvs:
            df = pd.read_csv(csvfile)
            sheet = safe_sheet_name(csvfile.stem)
            df.to_excel(w, sheet_name=sheet, index=False)
            index_rows.append({"sheet": sheet, "source": "torch_csv", "file": csvfile.name})

        pd.DataFrame(index_rows).to_excel(w, sheet_name="INDEX", index=False)

    print("Wrote:", out_xlsx)

if __name__ == "__main__":
    main()
