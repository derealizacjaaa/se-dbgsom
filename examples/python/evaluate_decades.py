"""
Evaluate World Panel Vesanto Clustering across Decades (1990s, 2000s, 2010s).

Reads: examples/output/world_panel_vesanto/vesanto_evaluation/cluster_evaluation_summary.csv
Writes: examples/output/world_panel_vesanto/decade_evaluation_report.md
        examples/output/world_panel_vesanto/plots/decade_evaluation_plots.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def main():
    # Paths
    base_dir = Path(__file__).parent.parent / "output" / "world_panel_vesanto"
    input_csv = base_dir / "vesanto_evaluation" / "cluster_evaluation_summary.csv"
    output_report = base_dir / "decade_evaluation_report.md"
    plot_dir = base_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    if not input_csv.exists():
        print(f"Error: Could not find {input_csv}")
        return

    df = pd.read_csv(input_csv)
    
    # Filter for decades of interest (excluding 2020s if incomplete or not requested, 
    # but user asked for "3 different decades", likely 1990, 2000, 2010)
    target_decades = [1990, 2000, 2010]
    df_filtered = df[df['decade'].isin(target_decades)].copy()
    
    # --- Analysis ---
    grouped = df_filtered.groupby('decade').agg({
        'db_score': ['mean', 'std', 'min', 'max'],
        'sil_score': ['mean', 'std', 'min', 'max'],
        'n_active_neurons': ['mean'],
        'n_countries': ['first'] # constant per decade usually
    })
    
    # --- Report Generation ---
    with open(output_report, "w") as f:
        f.write("# World Panel Vesanto Clustering Evaluation (3 Decades)\n\n")
        f.write("## Overview\n")
        f.write("Evaluation of clustering stability and quality across the 1990s, 2000s, and 2010s.\n\n")
        
        f.write("## Metrics Summary\n")
        f.write("| Decade | Mean DB Score (Lower is better) | Mean Silhouette (Higher is better) | Avg Active Neurons |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        for decade in target_decades:
            stats = grouped.loc[decade]
            f.write(f"| {decade}s | {stats['db_score']['mean']:.3f} ± {stats['db_score']['std']:.3f} | ")
            f.write(f"{stats['sil_score']['mean']:.3f} ± {stats['sil_score']['std']:.3f} | ")
            f.write(f"{stats['n_active_neurons']['mean']:.1f} |\n")
            
        f.write("\n## Detailed Observations\n")
        best_db = grouped['db_score']['mean'].idxmin()
        best_sil = grouped['sil_score']['mean'].idxmax()
        
        f.write(f"- **Best Davies-Bouldin Score**: {best_db}s ({grouped.loc[best_db]['db_score']['mean']:.3f})\n")
        f.write(f"- **Best Silhouette Score**: {best_sil}s ({grouped.loc[best_sil]['sil_score']['mean']:.3f})\n")
        
    print(f"Report generated: {output_report}")

    # --- Visualization ---
    plt.figure(figsize=(12, 5))
    
    # DB Score Plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_filtered, x='decade', y='db_score')
    plt.title('Davies-Bouldin Score Distribution')
    plt.ylabel('DB Score (Lower is better)')
    plt.xlabel('Decade')
    
    # Silhouette Score Plot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_filtered, x='decade', y='sil_score')
    plt.title('Silhouette Score Distribution')
    plt.ylabel('Silhouette Score (Higher is better)')
    plt.xlabel('Decade')
    
    plt.tight_layout()
    plot_path = plot_dir / "decade_metrics_boxplot.png"
    plt.savefig(plot_path)
    print(f"Plot generated: {plot_path}")
    
    # Line plot of metrics over time
    plt.figure(figsize=(12, 6))
    plt.plot(df_filtered['snapshot_year'], df_filtered['db_score'], marker='o', label='DB Score')
    plt.plot(df_filtered['snapshot_year'], df_filtered['sil_score'], marker='s', label='Silhouette Score')
    
    # Add decade shading
    colors = ['r', 'g', 'b']
    for i, decade in enumerate(target_decades):
        subset = df_filtered[df_filtered['decade'] == decade]
        if not subset.empty:
            plt.axvspan(subset['snapshot_year'].min() - 1, subset['snapshot_year'].max() + 1, 
                        color=colors[i], alpha=0.1, label=f"{decade}s")

    plt.title('Clustering Metrics Over Time')
    plt.xlabel('Year')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path_time = plot_dir / "metrics_over_time.png"
    plt.savefig(plot_path_time)
    print(f"Time plot generated: {plot_path_time}")

if __name__ == "__main__":
    main()
