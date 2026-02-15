#!/usr/bin/env python3
"""Visualize parallelization analysis results.

Creates publication-quality figures from parallelization_analysis.xlsx
"""

import sys
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_analysis(excel_path: str) -> dict:
    """Load all sheets from the analysis Excel file."""
    sheets = {}
    try:
        with pd.ExcelFile(excel_path) as xls:
            for sheet_name in xls.sheet_names:
                sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error loading {excel_path}: {e}")
        sys.exit(1)
    return sheets


def plot_amdahl_summary(df: pd.DataFrame, output_dir: Path):
    """Create visualizations for Amdahl's Law analysis."""
    if df.empty:
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Amdahl's Law Analysis: Parallelization Benefits", fontsize=16, fontweight='bold')
    
    # Prepare labels for x-axis
    df['config'] = df.apply(lambda row: f"N={row['N']}\nD={row['D']}\nK={row['K']}\n{row['cov_type']}", axis=1)
    
    # 1. Actual Speedup
    ax = axes[0, 0]
    x_pos = range(len(df))
    ax.bar(x_pos, df['actual_speedup'], color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='red', linestyle='--', label='No speedup', linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['config'], fontsize=9)
    ax.set_ylabel('Speedup Factor', fontweight='bold')
    ax.set_title('Actual Speedup Achieved (Old vs New)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Speedup vs Theoretical Limit
    ax = axes[0, 1]
    x_pos = range(len(df))
    width = 0.35
    ax.bar([x - width/2 for x in x_pos], df['actual_speedup'], width, 
           label='Actual', color='#3498db', alpha=0.7, edgecolor='black')
    ax.bar([x + width/2 for x in x_pos], df['theoretical_speedup_p_cores'], width,
           label='Theoretical Limit', color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['config'], fontsize=9)
    ax.set_ylabel('Speedup Factor', fontweight='bold')
    ax.set_title('Actual vs Theoretical Speedup', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Serial Fraction Comparison
    ax = axes[1, 0]
    x_pos = range(len(df))
    width = 0.35
    ax.bar([x - width/2 for x in x_pos], df['old_serial_fraction'] * 100, width,
           label='Old (Loop-based)', color='#e67e22', alpha=0.7, edgecolor='black')
    ax.bar([x + width/2 for x in x_pos], df['new_serial_fraction'] * 100, width,
           label='New (Vectorized)', color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['config'], fontsize=9)
    ax.set_ylabel('Serial Fraction (%)', fontweight='bold')
    ax.set_title('Serial Fraction: Lower is Better', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Speedup Efficiency
    ax = axes[1, 1]
    colors = ['#27ae60' if eff >= 70 else '#f39c12' if eff >= 50 else '#e74c3c' 
              for eff in df['speedup_efficiency_pct']]
    bars = ax.bar(x_pos, df['speedup_efficiency_pct'], color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Good (≥70%)')
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Moderate (≥50%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['config'], fontsize=9)
    ax.set_ylabel('Efficiency (%)', fontweight='bold')
    ax.set_title('Speedup Efficiency (Actual/Theoretical × 100)', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "amdahl_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_component_breakdown(df: pd.DataFrame, output_dir: Path):
    """Create component-level breakdown visualizations."""
    if df.empty:
        return
    
    # Focus on one representative configuration
    configs = df[['N', 'D', 'K', 'cov_type']].drop_duplicates()
    
    for idx, (_, config_row) in enumerate(configs.iterrows()):
        if idx >= 2:  # Limit to 2 configurations for clarity
            break
            
        N, D, K, cov_type = config_row['N'], config_row['D'], config_row['K'], config_row['cov_type']
        
        # Filter data for this configuration
        config_data = df[(df['N'] == N) & (df['D'] == D) & 
                        (df['K'] == K) & (df['cov_type'] == cov_type)]
        
        if config_data.empty:
            continue
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Component Breakdown: N={N}, D={D}, K={K}, {cov_type}", 
                    fontsize=14, fontweight='bold')
        
        # Separate old and new
        old_data = config_data[config_data['impl'] == 'old'].sort_values('time_ms', ascending=False)
        new_data = config_data[config_data['impl'] == 'new'].sort_values('time_ms', ascending=False)
        
        # 1. Time comparison
        ax = axes[0]
        components = old_data['component'].values
        x_pos = range(len(components))
        width = 0.35
        
        ax.barh([x - width/2 for x in x_pos], old_data['time_ms'], width,
               label='Old (Loop-based)', color='#e67e22', alpha=0.7, edgecolor='black')
        ax.barh([x + width/2 for x in x_pos], new_data['time_ms'], width,
               label='New (Vectorized)', color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_yticks(x_pos)
        ax.set_yticklabels(components, fontsize=10)
        ax.set_xlabel('Time (ms)', fontweight='bold')
        ax.set_title('Execution Time by Component', fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # 2. Fraction pie charts
        ax = axes[1]
        ax.axis('off')
        
        # Old implementation pie
        ax_old = fig.add_subplot(122, position=[0.55, 0.3, 0.18, 0.4])
        ax_old.pie(old_data['fraction'], labels=old_data['component'],
                  autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
        ax_old.set_title('Old Impl.\nTime Distribution', fontweight='bold')
        
        # New implementation pie
        ax_new = fig.add_subplot(122, position=[0.77, 0.3, 0.18, 0.4])
        ax_new.pie(new_data['fraction'], labels=new_data['component'],
                  autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
        ax_new.set_title('New Impl.\nTime Distribution', fontweight='bold')
        
        plt.tight_layout()
        output_path = output_dir / f"components_N{N}_D{D}_K{K}_{cov_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()


def plot_gpu_analysis(df: pd.DataFrame, output_dir: Path):
    """Create GPU utilization visualizations."""
    if df.empty:
        print("⚠ GPU analysis data not available (run with --device cuda)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GPU Utilization & Memory Bandwidth Analysis", fontsize=16, fontweight='bold')
    
    # Group by operation
    operations = df['operation'].unique()
    
    # 1. GPU Utilization
    ax = axes[0, 0]
    for op in operations:
        op_data = df[df['operation'] == op]
        x_pos = range(len(op_data))
        ax.plot(x_pos, op_data['old_gpu_util_pct'], 'o-', label=f'{op} (old)', linewidth=2)
        ax.plot(x_pos, op_data['new_gpu_util_pct'], 's--', label=f'{op} (new)', linewidth=2)
    ax.set_ylabel('GPU Utilization (%)', fontweight='bold')
    ax.set_xlabel('Test Configuration', fontweight='bold')
    ax.set_title('GPU Utilization Percentage', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    # 2. Memory Bandwidth
    ax = axes[0, 1]
    width = 0.35
    for i, op in enumerate(operations):
        op_data = df[df['operation'] == op].iloc[0]  # Take first config as example
        x_pos = [i]
        ax.bar([x - width/2 for x in x_pos], [op_data['old_bandwidth_gbs']], width,
              label=f'Old' if i == 0 else '', color='#e67e22', alpha=0.7, edgecolor='black')
        ax.bar([x + width/2 for x in x_pos], [op_data['new_bandwidth_gbs']], width,
              label=f'New' if i == 0 else '', color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(operations)))
    ax.set_xticklabels(operations, fontsize=10)
    ax.set_ylabel('Bandwidth (GB/s)', fontweight='bold')
    ax.set_title('Achieved Memory Bandwidth', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Bandwidth Utilization %
    ax = axes[1, 0]
    for op in operations:
        op_data = df[df['operation'] == op]
        x_pos = range(len(op_data))
        ax.plot(x_pos, op_data['old_bandwidth_util_pct'], 'o-', label=f'{op} (old)', linewidth=2)
        ax.plot(x_pos, op_data['new_bandwidth_util_pct'], 's--', label=f'{op} (new)', linewidth=2)
    ax.set_ylabel('Bandwidth Utilization (%)', fontweight='bold')
    ax.set_xlabel('Test Configuration', fontweight='bold')
    ax.set_title('Memory Bandwidth Utilization %', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    # 4. CUDA vs CPU time
    ax = axes[1, 1]
    # Aggregate across all operations for first config
    first_config = df.groupby(['N', 'D', 'K']).first().reset_index().iloc[0]
    categories = ['Old Impl.', 'New Impl.']
    cuda_times = [first_config['old_cuda_time_ms'], first_config['new_cuda_time_ms']]
    cpu_times = [first_config['old_cpu_time_ms'], first_config['new_cpu_time_ms']]
    
    x_pos = range(len(categories))
    width = 0.35
    ax.bar([x - width/2 for x in x_pos], cpu_times, width, label='CPU Time', 
          color='#95a5a6', alpha=0.7, edgecolor='black')
    ax.bar([x + width/2 for x in x_pos], cuda_times, width, label='CUDA Time',
          color='#76d7c4', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('CPU vs CUDA Execution Time', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "gpu_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_kernel_fusion(df: pd.DataFrame, output_dir: Path):
    """Create kernel fusion analysis visualizations."""
    if df.empty:
        print("⚠ Kernel fusion analysis data not available (run with --device cuda)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Kernel Fusion Impact Analysis", fontsize=16, fontweight='bold')
    
    # Prepare labels
    df['config'] = df.apply(lambda row: f"N={row['N']}\nD={row['D']}\nK={row['K']}\n{row['cov_type']}", axis=1)
    x_pos = range(len(df))
    
    # 1. Kernel Count Reduction
    ax = axes[0, 0]
    width = 0.35
    ax.bar([x - width/2 for x in x_pos], df['old_kernel_count'], width,
          label='Old (Loop-based)', color='#e67e22', alpha=0.7, edgecolor='black')
    ax.bar([x + width/2 for x in x_pos], df['new_kernel_count'], width,
          label='New (Vectorized)', color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['config'], fontsize=9)
    ax.set_ylabel('Kernel Count', fontweight='bold')
    ax.set_title('Number of Kernel Launches', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Kernel Reduction %
    ax = axes[0, 1]
    colors = ['#27ae60' if r >= 30 else '#f39c12' if r >= 15 else '#95a5a6' 
              for r in df['kernel_reduction_pct']]
    bars = ax.bar(x_pos, df['kernel_reduction_pct'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['config'], fontsize=9)
    ax.set_ylabel('Reduction (%)', fontweight='bold')
    ax.set_title('Kernel Launch Reduction Percentage', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Launch Overhead
    ax = axes[1, 0]
    width = 0.35
    ax.bar([x - width/2 for x in x_pos], df['old_launch_overhead_pct'], width,
          label='Old', color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.bar([x + width/2 for x in x_pos], df['new_launch_overhead_pct'], width,
          label='New', color='#27ae60', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['config'], fontsize=9)
    ax.set_ylabel('Overhead (%)', fontweight='bold')
    ax.set_title('Kernel Launch Overhead as % of Total Time', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Memory Traffic Reduction
    ax = axes[1, 1]
    width = 0.35
    ax.bar([x - width/2 for x in x_pos], df['old_estimated_traffic_mb'], width,
          label='Old', color='#e67e22', alpha=0.7, edgecolor='black')
    ax.bar([x + width/2 for x in x_pos], df['new_estimated_traffic_mb'], width,
          label='New', color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['config'], fontsize=9)
    ax.set_ylabel('Traffic (MB)', fontweight='bold')
    ax.set_title('Estimated Memory Traffic', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "kernel_fusion.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_summary_table(sheets: dict, output_dir: Path):
    """Generate a summary table as an image."""
    if 'Amdahl_Summary' not in sheets:
        return
    
    df = sheets['Amdahl_Summary']
    
    # Select key columns
    summary_cols = ['N', 'D', 'K', 'cov_type', 'actual_speedup', 
                   'theoretical_speedup_p_cores', 'speedup_efficiency_pct',
                   'old_serial_fraction', 'new_serial_fraction']
    
    if all(col in df.columns for col in summary_cols):
        summary_df = df[summary_cols].copy()
        summary_df.columns = ['N', 'D', 'K', 'Cov', 'Actual\nSpeedup', 
                             'Theoretical\nSpeedup', 'Efficiency\n(%)',
                             'Old Serial\nFraction', 'New Serial\nFraction']
        
        # Round for display
        for col in ['Actual\nSpeedup', 'Theoretical\nSpeedup']:
            summary_df[col] = summary_df[col].round(2)
        for col in ['Efficiency\n(%)', 'Old Serial\nFraction', 'New Serial\nFraction']:
            summary_df[col] = summary_df[col].round(3)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, len(summary_df) * 0.6 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                        cellLoc='center', loc='center', 
                        colWidths=[0.08, 0.08, 0.08, 0.08, 0.12, 0.12, 0.12, 0.14, 0.14])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        for i in range(1, len(summary_df) + 1):
            for j in range(len(summary_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        plt.title('Parallelization Analysis Summary', fontsize=16, fontweight='bold', pad=20)
        
        output_path = output_dir / "summary_table.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize parallelization analysis results")
    parser.add_argument("--input", type=str, default="results/parallelization_analysis.xlsx",
                       help="Input Excel file from analyze_parallelization.py")
    parser.add_argument("--output-dir", type=str, default="results/figures",
                       help="Output directory for figures")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading analysis from: {args.input}")
    sheets = load_analysis(args.input)
    
    print(f"\nGenerating figures in: {output_dir}")
    print("="*60)
    
    # Generate all plots
    if 'Amdahl_Summary' in sheets:
        plot_amdahl_summary(sheets['Amdahl_Summary'], output_dir)
    
    if 'Amdahl_Components' in sheets:
        plot_component_breakdown(sheets['Amdahl_Components'], output_dir)
    
    if 'GPU_Analysis' in sheets and not sheets['GPU_Analysis'].empty:
        plot_gpu_analysis(sheets['GPU_Analysis'], output_dir)
    
    if 'Kernel_Fusion' in sheets and not sheets['Kernel_Fusion'].empty:
        plot_kernel_fusion(sheets['Kernel_Fusion'], output_dir)
    
    # Generate summary table
    generate_summary_table(sheets, output_dir)
    
    print("="*60)
    print(f"✓ All figures generated successfully!")
    print(f"\nView figures in: {output_dir}/")


if __name__ == "__main__":
    main()
