#!/usr/bin/env python3
"""Generate a text summary report from parallelization analysis."""

import sys
import pandas as pd


def generate_report(excel_path: str):
    """Generate a markdown summary report."""
    
    try:
        sheets = {}
        with pd.ExcelFile(excel_path) as xls:
            for sheet_name in xls.sheet_names:
                sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error loading {excel_path}: {e}")
        sys.exit(1)
    
    report = []
    report.append("# Parallelization Analysis Summary Report\n")
    report.append(f"**Analysis File:** `{excel_path}`\n")
    
    # Amdahl's Law Summary
    if 'Amdahl_Summary' in sheets:
        df = sheets['Amdahl_Summary']
        report.append("\n## 1. Amdahl's Law Analysis\n")
        report.append("### Overall Results\n")
        
        avg_speedup = df['actual_speedup'].mean()
        max_speedup = df['actual_speedup'].max()
        min_speedup = df['actual_speedup'].min()
        avg_efficiency = df['speedup_efficiency_pct'].mean()
        
        report.append(f"- **Average Speedup:** {avg_speedup:.2f}x")
        report.append(f"- **Speedup Range:** {min_speedup:.2f}x to {max_speedup:.2f}x")
        report.append(f"- **Average Efficiency:** {avg_efficiency:.1f}%")
        
        old_serial = df['old_serial_fraction'].mean() * 100
        new_serial = df['new_serial_fraction'].mean() * 100
        serial_reduction = ((old_serial - new_serial) / old_serial) * 100
        
        report.append(f"- **Old Serial Fraction:** {old_serial:.1f}%")
        report.append(f"- **New Serial Fraction:** {new_serial:.1f}%")
        report.append(f"- **Serial Fraction Reduction:** {serial_reduction:.1f}%\n")
        
        # Best and worst cases
        best_idx = df['actual_speedup'].idxmax()
        worst_idx = df['actual_speedup'].idxmin()
        
        report.append("### Best Configuration\n")
        best = df.iloc[best_idx]
        report.append(f"- **Config:** N={best['N']}, D={best['D']}, K={best['K']}, {best['cov_type']}")
        report.append(f"- **Speedup:** {best['actual_speedup']:.2f}x")
        report.append(f"- **Efficiency:** {best['speedup_efficiency_pct']:.1f}%\n")
        
        report.append("### Worst Configuration\n")
        worst = df.iloc[worst_idx]
        report.append(f"- **Config:** N={worst['N']}, D={worst['D']}, K={worst['K']}, {worst['cov_type']}")
        report.append(f"- **Speedup:** {worst['actual_speedup']:.2f}x")
        report.append(f"- **Efficiency:** {worst['speedup_efficiency_pct']:.1f}%\n")
        
        # Detailed table
        report.append("### Detailed Results by Configuration\n")
        report.append("| N | D | K | Cov Type | Speedup | Efficiency | Old Serial % | New Serial % |")
        report.append("|---|---|---|----------|---------|------------|--------------|--------------|")
        
        for _, row in df.iterrows():
            report.append(f"| {row['N']} | {row['D']} | {row['K']} | {row['cov_type']} | "
                         f"{row['actual_speedup']:.2f}x | {row['speedup_efficiency_pct']:.1f}% | "
                         f"{row['old_serial_fraction']*100:.1f}% | {row['new_serial_fraction']*100:.1f}% |")
        report.append("")
    
    # Component breakdown
    if 'Amdahl_Components' in sheets:
        df = sheets['Amdahl_Components']
        report.append("\n## 2. Component-Level Analysis\n")
        
        # Average time by component
        old_df = df[df['impl'] == 'old'].groupby('component')['time_ms'].mean().sort_values(ascending=False)
        new_df = df[df['impl'] == 'new'].groupby('component')['time_ms'].mean().sort_values(ascending=False)
        
        report.append("### Average Execution Time by Component\n")
        report.append("| Component | Old (ms) | New (ms) | Speedup |")
        report.append("|-----------|----------|----------|---------|")
        
        for component in old_df.index:
            old_time = old_df[component]
            new_time = new_df[component] if component in new_df.index else 0
            speedup = old_time / new_time if new_time > 0 else float('inf')
            report.append(f"| {component} | {old_time:.2f} | {new_time:.2f} | {speedup:.2f}x |")
        report.append("")
        
        # Bottlenecks
        report.append("### Identified Bottlenecks (Old Implementation)\n")
        report.append("Components consuming most time:\n")
        for i, (component, time) in enumerate(old_df.head(3).items(), 1):
            fraction = time / old_df.sum() * 100
            report.append(f"{i}. **{component}**: {time:.2f}ms ({fraction:.1f}% of total)")
        report.append("")
    
    # GPU Analysis
    if 'GPU_Analysis' in sheets and not sheets['GPU_Analysis'].empty:
        df = sheets['GPU_Analysis']
        report.append("\n## 3. GPU Occupancy & Memory Bandwidth\n")
        
        avg_old_gpu = df['old_gpu_util_pct'].mean()
        avg_new_gpu = df['new_gpu_util_pct'].mean()
        avg_old_bw = df['old_bandwidth_util_pct'].mean()
        avg_new_bw = df['new_bandwidth_util_pct'].mean()
        
        report.append(f"- **Old GPU Utilization:** {avg_old_gpu:.1f}%")
        report.append(f"- **New GPU Utilization:** {avg_new_gpu:.1f}%")
        report.append(f"- **Improvement:** {avg_new_gpu - avg_old_gpu:+.1f}%\n")
        
        report.append(f"- **Old Bandwidth Utilization:** {avg_old_bw:.1f}%")
        report.append(f"- **New Bandwidth Utilization:** {avg_new_bw:.1f}%")
        report.append(f"- **Improvement:** {avg_new_bw - avg_old_bw:+.1f}%\n")
        
        # Classification
        if avg_new_bw > 50:
            classification = "**Memory-Bound**"
            recommendation = "Focus on memory access optimization (RQ3)"
        elif avg_new_gpu > 70:
            classification = "**Compute-Bound**"
            recommendation = "Focus on algorithmic efficiency"
        else:
            classification = "**Overhead-Bound**"
            recommendation = "Reduce kernel launch overhead and improve occupancy"
        
        report.append(f"**Operation Classification:** {classification}")
        report.append(f"**Recommendation:** {recommendation}\n")
    else:
        report.append("\n## 3. GPU Analysis\n")
        report.append("⚠️ GPU analysis not available. Run with `--device cuda` on a CUDA-enabled system.\n")
    
    # Kernel Fusion
    if 'Kernel_Fusion' in sheets and not sheets['Kernel_Fusion'].empty:
        df = sheets['Kernel_Fusion']
        report.append("\n## 4. Kernel Fusion Impact\n")
        
        avg_kernel_reduction = df['kernel_reduction_pct'].mean()
        avg_traffic_reduction = df['traffic_reduction_pct'].mean()
        avg_old_overhead = df['old_launch_overhead_pct'].mean()
        avg_new_overhead = df['new_launch_overhead_pct'].mean()
        
        report.append(f"- **Average Kernel Reduction:** {avg_kernel_reduction:.1f}%")
        report.append(f"- **Average Traffic Reduction:** {avg_traffic_reduction:.1f}%")
        report.append(f"- **Old Launch Overhead:** {avg_old_overhead:.1f}% of total time")
        report.append(f"- **New Launch Overhead:** {avg_new_overhead:.1f}% of total time")
        report.append(f"- **Overhead Reduction:** {avg_old_overhead - avg_new_overhead:.1f}%\n")
        
        total_old_traffic = df['old_estimated_traffic_mb'].sum()
        total_new_traffic = df['new_estimated_traffic_mb'].sum()
        total_saved = total_old_traffic - total_new_traffic
        
        report.append(f"- **Total Memory Traffic (Old):** {total_old_traffic:.1f} MB")
        report.append(f"- **Total Memory Traffic (New):** {total_new_traffic:.1f} MB")
        report.append(f"- **Total Traffic Saved:** {total_saved:.1f} MB\n")
    else:
        report.append("\n## 4. Kernel Fusion Analysis\n")
        report.append("⚠️ Kernel fusion analysis not available. Run with `--device cuda` on a CUDA-enabled system.\n")
    
    # Conclusions
    report.append("\n## 5. Key Takeaways\n")
    
    if 'Amdahl_Summary' in sheets:
        df = sheets['Amdahl_Summary']
        avg_speedup = df['actual_speedup'].mean()
        avg_efficiency = df['speedup_efficiency_pct'].mean()
        
        if avg_speedup > 3:
            report.append(f"✓ **Excellent speedup achieved:** {avg_speedup:.2f}x average across configurations")
        elif avg_speedup > 2:
            report.append(f"✓ **Good speedup achieved:** {avg_speedup:.2f}x average across configurations")
        else:
            report.append(f"⚠ **Moderate speedup:** {avg_speedup:.2f}x average - room for improvement")
        
        if avg_efficiency > 70:
            report.append(f"✓ **High efficiency:** {avg_efficiency:.1f}% - close to theoretical limits")
        elif avg_efficiency > 50:
            report.append(f"→ **Moderate efficiency:** {avg_efficiency:.1f}% - some optimization opportunities remain")
        else:
            report.append(f"⚠ **Low efficiency:** {avg_efficiency:.1f}% - significant gap to theoretical limits")
    
    # Bridge to RQ3
    report.append("\n## 6. Connection to RQ3 (Memory Optimizations)\n")
    report.append("This analysis motivates memory-focused optimizations:\n")
    
    if 'GPU_Analysis' in sheets and not sheets['GPU_Analysis'].empty:
        df = sheets['GPU_Analysis']
        avg_bw = df['new_bandwidth_util_pct'].mean()
        if avg_bw > 40:
            report.append("- Memory bandwidth utilization indicates **memory-bound operations**")
    
    if 'Kernel_Fusion' in sheets and not sheets['Kernel_Fusion'].empty:
        report.append("- Significant memory traffic remains despite fusion")
    
    report.append("- Serial bottlenecks suggest memory access patterns as optimization target")
    report.append("- **Next step:** Systematic memory access optimization (coalescing, caching, layout)\n")
    
    # Output report
    report_text = "\n".join(report)
    return report_text


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate summary report from analysis")
    parser.add_argument("--input", type=str, default="results/parallelization_analysis.xlsx",
                       help="Input Excel file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output markdown file (default: print to stdout)")
    
    args = parser.parse_args()
    
    report = generate_report(args.input)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
