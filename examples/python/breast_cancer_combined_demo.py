"""
Breast Cancer Dataset Demo - Combined Hexagonal DBGSOM + Vesanto Clustering

Demonstrates:
- Bayesian hyperparameter optimization for DBGSOM training
- Hexagonal grid visualization (expanded U-matrix, component planes, hit map, labels)
- Vesanto-Alhoniemi hierarchical clustering with Ward's method (Auto k vs True k=2)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from dbgsom import (
    SEDBGSOM,
    assign_cluster_labels,
    compute_all_metrics,
)
from dbgsom.clustering import SOMVesantoClustering
from dbgsom.visualization import DBGSOMVisualizer


def compute_cluster_purities(som, df, y, neuron_labels, target_names):
    """Compute overall purity, ARI, and per-cluster purity breakdown."""
    sample_labels = assign_cluster_labels(som, df, neuron_labels)
    bmus = som.predict(df)
    metrics = compute_all_metrics(som, neuron_labels, df, y, bmus)

    per_cluster = {}
    for c in sorted(set(neuron_labels.values())):
        mask = sample_labels == c
        size = int(mask.sum())
        if size > 0:
            counts = [np.sum((y == i) & mask) for i in range(len(target_names))]
            dominant_idx = np.argmax(counts)
            purity = counts[dominant_idx] / size
            per_cluster[c] = (target_names[dominant_idx], purity, size)

    return metrics['purity'], metrics['ari'], per_cluster


def plot_dendrogram_with_purity(
    ax, linkage_matrix, k, purity, ari, per_cluster, title
):
    """Plot a truncated dendrogram with purity text box and branch annotations."""
    from scipy.cluster.hierarchy import dendrogram

    cut_height = linkage_matrix[-k + 1, 2] if k > 1 else 0

    dendro = dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90,
        leaf_font_size=9,
        show_contracted=True,
        color_threshold=cut_height,
    )
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Ward Distance")

    if k > 1:
        ax.axhline(y=cut_height, color='r', linestyle='--', linewidth=1.2,
                    label=f'Cut at k={k}')
        ax.legend(loc='upper left', fontsize=8)

    lines = [f"Purity: {purity:.3f} | ARI: {ari:.3f}"]
    for c_id, (dominant, c_purity, c_size) in sorted(per_cluster.items()):
        lines.append(f"C{c_id}: {dominant} ({c_purity:.0%}, n={c_size})")
    text = "\n".join(lines)

    ax.text(
        0.98, 0.97, text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.85),
    )

    if k > 1 and per_cluster:
        x_min, x_max = ax.get_xlim()
        sorted_clusters = sorted(per_cluster.items())
        n_clusters = len(sorted_clusters)
        for idx, (c_id, (dominant, c_purity, c_size)) in enumerate(sorted_clusters):
            x_pos = x_min + (x_max - x_min) * (idx + 0.5) / n_clusters
            ax.annotate(
                f"C{c_id}: {c_purity:.0%}",
                xy=(x_pos, cut_height),
                xytext=(x_pos, cut_height * 1.12),
                fontsize=8, fontweight='bold', color='darkred',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-', color='darkred', lw=0.8),
            )


def main():
    print("=" * 70)
    print("  Combined DBGSOM Demo: Breast Cancer Dataset")
    print("=" * 70)

    # 1. Load Data
    print("\n1. Loading Breast Cancer dataset...")
    data_path = Path(__file__).parent.parent / "data" / "breast_cancer.csv"
    df_raw = pd.read_csv(data_path)
    
    target_col = 'target'
    feature_names = [c for c in df_raw.columns if c != target_col]
    
    X = df_raw[feature_names].values
    y = df_raw[target_col].values
    
    # Infer target names (0=Malignant, 1=Benign)
    unique_classes = sorted(np.unique(y))
    # Standard Wisconsin Diagnostic encoding: 0=Malignant, 1=Benign
    if list(unique_classes) == [0, 1]:
        target_names = ["Malignant", "Benign"]
    else:
        target_names = [f"Class {c}" for c in unique_classes]

    print(f"   Samples: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {len(target_names)} - {target_names}")

    df = pd.DataFrame(X, columns=feature_names)

    # 2. Train DBGSOM
    print("\n2. Training DBGSOM with Bayesian hyperparameter optimization...")

    # Using somewhat larger ranges/neurons for larger dataset
    som = SEDBGSOM(
        bayesian=True,
        bayesian_trials=30, # Reduced to 30 for speed on larger dataset
        bayesian_te_constraint=0.35,
        bayesian_ranges={
            'lambda_': (0.5, 3.0),
            'max_neurons': (100, 200),
            'n_iter': (150, 300),
            'init_size': (2, 5),
        },
        random_state=42,
    )
    som.fit(df)

    opt = som.optimization_result_
    print(f"   Final neurons: {som.n_neurons_}")
    print(f"   Best lambda:   {som.lambda_:.3f}")
    print(f"   QE:            {opt['best_qe']:.4f}")
    print(f"   TE:            {opt['best_te']:.4f}")

    # 3. Vesanto Clustering
    print("\n3. Vesanto-Alhoniemi clustering...")
    print("-" * 70)

    clustering_auto = SOMVesantoClustering(
        method='ward', auto_k_method='davies_bouldin'
    )
    labels_auto = clustering_auto.fit(som, df)
    k_auto = clustering_auto.n_clusters_

    true_k = len(target_names)
    clustering_true = SOMVesantoClustering(method='ward')
    labels_true = clustering_true.fit(som, df, n_clusters=true_k)

    bmus = som.predict(df)
    metrics_auto = compute_all_metrics(som, labels_auto, df, y, bmus)
    metrics_true = compute_all_metrics(som, labels_true, df, y, bmus)

    print(f"\n   {'Method':<30} {'k':>3}  {'NMI':>6}  {'ARI':>6}  {'Purity':>7}")
    print("   " + "-" * 56)
    print(f"   {'Vesanto (auto k, DB)':<30} {k_auto:>3}  {metrics_auto['nmi']:>6.3f}  {metrics_auto['ari']:>6.3f}  {metrics_auto['purity']:>7.3f}")
    print(f"   {'Vesanto (true k='+str(true_k)+')':<30} {true_k:>3}  {metrics_true['nmi']:>6.3f}  {metrics_true['ari']:>6.3f}  {metrics_true['purity']:>7.3f}")
    print("   " + "-" * 56)

    # 4. Visualizations
    print("\n4. Creating SOM visualizations...")

    output_dir = Path(__file__).parent.parent / "output"
    viz = DBGSOMVisualizer(output_dir, "breast_cancer_combined")

    if viz.raw_data_path.exists() and not any(viz.raw_data_path.iterdir()):
        viz.raw_data_path.rmdir()

    viz.plot_expanded_umatrix(som, "umatrix_expanded.png")
    print(f"   Saved: umatrix_expanded.png")

    viz.plot_hit_counts(som, df, "hit_counts.png")
    print(f"   Saved: hit_counts.png")

    viz.plot_labels_hex(som, df, y, target_names, "labels_hex.png")
    print(f"   Saved: labels_hex.png")

    # Component planes - limit to first 6 if too many, or just plot all. 
    # Breast cancer has 30 features, creating 30 plots might be overwhelming but visualizer handles it.
    # The visualizer usually creates a grid. 30 is manageable.
    try:
        viz.plot_component_planes(som, feature_names, "component_planes")
        print(f"   Saved: component_planes_all.png")
    except AttributeError:
        print("   Component planes: skipped (not available)")

    # 5. K Evaluation
    print("\n5. Creating K evaluation plots...")

    try:
        import matplotlib.pyplot as plt
        plots_dir = viz.plots_path
        db_scores, sil_scores = clustering_auto.get_k_evaluation_profile()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ks_db = sorted(db_scores.keys())
        axes[0].plot(ks_db, [db_scores[k] for k in ks_db], 'b-o')
        axes[0].axvline(x=clustering_auto.optimal_k_, color='r', linestyle='--')
        axes[0].set_title('Davies-Bouldin')

        ks_sil = sorted(sil_scores.keys())
        axes[1].plot(ks_sil, [sil_scores[k] for k in ks_sil], 'g-o')
        best_sil_k = max(sil_scores.keys(), key=sil_scores.get)
        axes[1].axvline(x=best_sil_k, color='r', linestyle='--')
        axes[1].set_title('Silhouette')

        plt.tight_layout()
        plt.savefig(plots_dir / "vesanto_k_evaluation.png", dpi=150)
        plt.close()
        print(f"   Saved: vesanto_k_evaluation.png")
    except ImportError:
        pass

    # 6. Dendrograms
    print("\n6. Creating dendrograms...")
    purity_auto, ari_auto, per_cluster_auto = compute_cluster_purities(
        som, df, y, labels_auto, target_names
    )
    purity_true, ari_true, per_cluster_true = compute_cluster_purities(
        som, df, y, labels_true, target_names
    )

    linkage_auto = clustering_auto.get_dendrogram_data()
    linkage_true = clustering_true.get_dendrogram_data()

    if linkage_auto is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_dendrogram_with_purity(
            ax, linkage_auto, k_auto, purity_auto, ari_auto, per_cluster_auto,
            f"Vesanto Dendrogram (Auto k={k_auto})"
        )
        plt.savefig(plots_dir / "vesanto_dendrogram_auto_k.png", dpi=150)
        plt.close()

    if linkage_true is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_dendrogram_with_purity(
            ax, linkage_true, true_k, purity_true, ari_true, per_cluster_true,
            f"Vesanto Dendrogram (True k={true_k})"
        )
        plt.savefig(plots_dir / "vesanto_dendrogram_true_k.png", dpi=150)
        plt.close()

    print("\n" + "=" * 70)
    print("  Demo complete! Check examples/output/breast_cancer_combined/ for plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()
