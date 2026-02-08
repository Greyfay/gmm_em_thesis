#!/usr/bin/env python3
"""Benchmark comparing PyTorch (unoptimized) vs scikit-learn GaussianMixture.

This script compares the runtime of the loop-based PyTorch implementation
against scikit-learn's reference GaussianMixture to show that both produce
equivalent results but with different performance characteristics.
"""

import sys
import os
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture

try:
    from openpyxl.styles import Font, PatternFill, Alignment
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# Add parent directory to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Import unoptimized PyTorch implementation
from implementation import _torch_gmm_em_old as torch_impl


def timer(func: Callable, *args, n_runs: int = 5, warmup: int = 1, **kwargs) -> Tuple[float, float]:
    """Time a function with warmup runs.
    
    Returns:
        (mean_time, std_time) in milliseconds
    """
    # Warmup runs
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)


def generate_test_data(
    N: int = 1000,
    D: int = 50,
    K: int = 5,
    seed: int = None,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Generate synthetic test data."""
    if seed is None:
        seed = np.random.randint(1, 1001)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate data
    X_np = np.random.randn(N, D).astype(np.float64)
    X_torch = torch.from_numpy(X_np).to(torch.float64)
    
    return X_np, X_torch


def benchmark_fit():
    """Benchmark GMM fit for various covariance types and problem sizes."""
    print("\n" + "="*100)
    print("BENCHMARK: TorchGaussianMixture (unoptimized) vs scikit-learn GaussianMixture")
    print("="*100)
    
    results = []
    
    for cov_type in ["diag", "tied", "full"]:
        print(f"\n--- Covariance type: {cov_type} ---")
        
        for N, D, K in [(500, 20, 5), (1000, 50, 5), (2000, 100, 5)]:
            X_np, X_torch = generate_test_data(N, D, K)
            
            # sklearn fit
            def fit_sklearn():
                model = GaussianMixture(
                    n_components=K,
                    covariance_type=cov_type,
                    max_iter=300,
                    n_init=5,
                    init_params="kmeans",
                    random_state=42,
                    verbose=0,
                    tol=1e-4,
                )
                model.fit(X_np)
                return model
            
            # PyTorch fit
            def fit_torch():
                model = torch_impl.TorchGaussianMixture(
                    n_components=K,
                    covariance_type=cov_type,
                    max_iter=300,
                    n_init=5,
                    init_params="kmeans",
                    dtype=torch.float64,
                    tol=1e-4,
                )
                model.fit(X_torch)
                return model
            
            sklearn_time, sklearn_std = timer(fit_sklearn, n_runs=3, warmup=1)
            torch_time, torch_std = timer(fit_torch, n_runs=3, warmup=1)
            
            # sklearn is faster (lower is better)
            speedup = sklearn_time / torch_time
            
            print(f"N={N}, D={D}, K={K}:")
            print(f"  scikit-learn: {sklearn_time:.3f} ± {sklearn_std:.3f} ms")
            print(f"  PyTorch:      {torch_time:.3f} ± {torch_std:.3f} ms")
            print(f"  Ratio (sklearn/pytorch): {speedup:.2f}x")
            
            results.append({
                "Covariance Type": cov_type,
                "N": N,
                "D": D,
                "K": K,
                "scikit-learn Time (ms)": sklearn_time,
                "scikit-learn Std (ms)": sklearn_std,
                "PyTorch Time (ms)": torch_time,
                "PyTorch Std (ms)": torch_std,
                "Ratio (sklearn/pytorch)": speedup,
            })
    
    return results


def benchmark_individual_functions():
    """Benchmark individual functions where applicable."""
    print("\n" + "="*100)
    print("BENCHMARK: Individual function comparisons")
    print("="*100)
    
    results = []
    
    # Test data
    N, D, K = 1000, 50, 5
    random_seed = np.random.randint(1, 1001)
    
    print(f"\nUsing N={N}, D={D}, K={K}, seed={random_seed}")
    
    # Benchmark E-step (log probability computation)
    print(f"\n--- E-step: Log probability computation ---")
    
    for cov_type in ["diag", "tied", "full"]:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        X_np = np.random.randn(N, D).astype(np.float64)
        X_torch = torch.from_numpy(X_np).to(torch.float64)
        
        means_np = np.random.randn(K, D).astype(np.float64)
        means_torch = torch.from_numpy(means_np).to(torch.float64)
        
        weights_np = np.ones(K) / K
        weights_torch = torch.from_numpy(weights_np).to(torch.float64)
        
        # Generate covariances
        if cov_type == "diag":
            cov_torch = torch.ones((K, D), dtype=torch.float64) + 0.5
        elif cov_type == "tied":
            cov_torch = torch.eye(D, dtype=torch.float64) + 0.1
        else:  # full
            cov_torch = torch.stack([
                torch.eye(D, dtype=torch.float64) + 0.1 * torch.randn(D, D)
                for _ in range(K)
            ])
            cov_torch = torch.bmm(cov_torch, cov_torch.transpose(-1, -2))
        
        # PyTorch E-step
        def torch_estep():
            lower, log_resp = torch_impl._expectation_step_precchol(
                X_torch, means_torch, 
                torch_impl._compute_precisions_cholesky(cov_torch, cov_type),
                weights_torch, cov_type
            )
            return lower, log_resp
        
        torch_time, torch_std = timer(torch_estep, n_runs=5, warmup=2)
        
        print(f"{cov_type:9s}: {torch_time:.3f} ± {torch_std:.3f} ms")
        
        results.append({
            "Function": "E-step",
            "Covariance Type": cov_type,
            "N": N,
            "D": D,
            "K": K,
            "PyTorch Time (ms)": torch_time,
            "PyTorch Std (ms)": torch_std,
        })
    
    return results


def main():
    """Run all benchmarks."""
    print("="*100)
    print("PYTORCH (UNOPTIMIZED) vs SCIKIT-LEARN COMPARISON")
    print("="*100)
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"scikit-learn version: (imported)")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Run benchmarks
    fit_results = benchmark_fit()
    func_results = benchmark_individual_functions()
    
    print("\n" + "="*100)
    print("BENCHMARKS COMPLETE")
    print("="*100)
    
    # Create DataFrames
    df_fit = pd.DataFrame(fit_results)
    
    # Save to Excel with enhanced formatting
    output_file = os.path.join(os.path.dirname(__file__), "pytorch_vs_sklearn.xlsx")
    
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Write fit results
        df_fit.to_excel(writer, sheet_name="Full Fit Comparison", index=False)
        
        # Write function results if available
        if func_results:
            df_func = pd.DataFrame(func_results)
            df_func.to_excel(writer, sheet_name="Function Comparison", index=False)
        
        # Create summary statistics sheet
        summary_data = {
            "Metric": [
                "Total Benchmarks",
                "Average Ratio (sklearn/pytorch)",
                "Max Ratio (sklearn faster)",
                "Min Ratio (pytorch faster)",
                "Total scikit-learn Time (ms)",
                "Total PyTorch Time (ms)",
            ],
            "Value": [
                len(df_fit),
                f"{df_fit['Ratio (sklearn/pytorch)'].mean():.2f}x",
                f"{df_fit['Ratio (sklearn/pytorch)'].max():.2f}x",
                f"{df_fit['Ratio (sklearn/pytorch)'].min():.2f}x",
                f"{df_fit['scikit-learn Time (ms)'].sum():.2f}",
                f"{df_fit['PyTorch Time (ms)'].sum():.2f}",
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        
        # Create breakdown by covariance type
        breakdown_data = []
        for cov_type in df_fit["Covariance Type"].unique():
            subset = df_fit[df_fit["Covariance Type"] == cov_type]
            breakdown_data.append({
                "Covariance Type": cov_type,
                "Num Benchmarks": len(subset),
                "Avg sklearn Time (ms)": subset["scikit-learn Time (ms)"].mean(),
                "Avg PyTorch Time (ms)": subset["PyTorch Time (ms)"].mean(),
                "Avg Ratio (sklearn/pytorch)": subset["Ratio (sklearn/pytorch)"].mean(),
            })
        df_breakdown = pd.DataFrame(breakdown_data)
        df_breakdown.to_excel(writer, sheet_name="By Covariance Type", index=False)
        
        # Apply formatting if openpyxl available
        if OPENPYXL_AVAILABLE:
            _format_excel_sheets(writer, df_fit, df_func if func_results else None)
    
    print(f"\n✓ Results exported to: {output_file}")
    print(f"\nFull Fit Summary:")
    print(f"  Total benchmarks: {len(df_fit)}")
    print(f"  Average ratio (sklearn/pytorch): {df_fit['Ratio (sklearn/pytorch)'].mean():.2f}x")
    print(f"  Max ratio: {df_fit['Ratio (sklearn/pytorch)'].max():.2f}x")
    print(f"  Min ratio: {df_fit['Ratio (sklearn/pytorch)'].min():.2f}x")
    
    # Print interpretation
    ratio_mean = df_fit['Ratio (sklearn/pytorch)'].mean()
    if ratio_mean > 1:
        print(f"\n→ scikit-learn is faster by {ratio_mean:.2f}x on average")
    else:
        print(f"\n→ PyTorch is faster by {1/ratio_mean:.2f}x on average")
    
    # Breakdown by covariance type
    print(f"\nBreakdown by Covariance Type:")
    for cov_type in sorted(df_fit["Covariance Type"].unique()):
        subset = df_fit[df_fit["Covariance Type"] == cov_type]
        avg_ratio = subset["Ratio (sklearn/pytorch)"].mean()
        print(f"  {cov_type:6s}: {avg_ratio:.2f}x")


def _format_excel_sheets(writer, df_fit, df_func):
    """Apply formatting to Excel sheets."""
    # Format Full Fit Comparison sheet
    ws = writer.sheets["Full Fit Comparison"]
    
    # Set column widths
    col_widths = {
        "A": 18, "B": 16, "C": 8, "D": 8, "E": 8,
        "F": 22, "G": 18, "H": 18, "I": 18, "J": 25
    }
    for col, width in col_widths.items():
        ws.column_dimensions[col].width = width
    
    # Format header row
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
    
    # Format data cells
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center")
            # Format numeric columns
            if cell.column in [3, 4, 5]:  # N, D, K columns
                cell.number_format = "0"
            elif cell.column >= 6:  # Time columns
                cell.number_format = "0.000"
    
    # Color code the ratio column
    for row_num, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row, 
                                                min_col=10, max_col=10), start=2):
        try:
            ratio = float(row[0].value)
            if ratio > 1.1:
                row[0].fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
            elif ratio < 0.9:
                row[0].fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        except (ValueError, TypeError):
            pass
    
    # Format Summary sheet if it exists
    if "Summary" in writer.sheets:
        ws_summary = writer.sheets["Summary"]
        for col in ["A", "B"]:
            ws_summary.column_dimensions[col].width = 35
        
        for cell in ws_summary[1]:
            cell.fill = header_fill
            cell.font = header_font
    
    # Format By Covariance Type sheet
    if "By Covariance Type" in writer.sheets:
        ws_cov = writer.sheets["By Covariance Type"]
        for col in ["A", "B", "C", "D", "E"]:
            ws_cov.column_dimensions[col].width = 20
        
        for cell in ws_cov[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")


if __name__ == "__main__":
    main()
