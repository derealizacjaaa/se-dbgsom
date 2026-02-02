"""
Circles Dataset Demo - Combined Hexagonal DBGSOM + Vesanto Clustering

Demonstrates:
- DBGSOM training with StatisticalError growth threshold
- Hexagonal grid visualization (expanded U-matrix, component planes, hit map, labels)
- Vesanto-Alhoniemi hierarchical clustering with Ward's method
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


def plot_dendrogram_with_purity(ax, linkage_matrix, k, purity, ari, per_cluster, title):
    from scipy.cluster.hierarchy import dendrogram
    cut_height = linkage_matrix[-k + 1, 2] if k > 1 else 0
    dendrogram(linkage_matrix, ax=ax, truncate_mode='lastp', p=12,
               leaf_rotation=90, leaf_font_size=9, show_contracted=True,
               color_threshold=cut_height)
    ax.set_title(title, fontweight='bold', fontsize=11)
    
    if k > 1:
        ax.axhline(y=cut_height, color='r', linestyle='--', label=f'Cut k={k}')
        ax.legend(loc='upper left', fontsize=8)

    lines = [f"Purity: {purity:.3f} | ARI: {ari:.3f}"]
    for c_id, (dom, cp, csz) in sorted(per_cluster.items()):
        lines.append(f"C{c_id}: {dom} ({cp:.0%}, n={csz})")
    text = "\n".join(lines)
    ax.text(0.98, 0.97, text, transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.85))


def main():
    print("=" * 70)
    print("  Combined DBGSOM Demo: Circles Dataset")
    print("=" * 70)

    # 1. Load Data
    print("\n1. Loading Circles dataset...")
    data_path = Path(__file__).parent.parent / "data" / "circles.csv"
    df_raw = pd.read_csv(data_path)
    
    features = ['x1', 'x2']
    target = 'target'
    
    X = df_raw[features].values
    y = df_raw[target].values
    
    target_names = [f"Class {c}" for c in sorted(np.unique(y))]
    print(f"   Samples: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")

    df = pd.DataFrame(X, columns=features)

    # 2. Train
    print("\n2. Training DBGSOM...")
    som = SEDBGSOM(
        lambda_=1.5, max_neurons=130, n_iter=200, init_size=(3, 3),
        random_state=42,
    )
    som.fit(df)
    print(f"   Final neurons: {som.n_neurons_}")

    # 3. Clustering
    print("\n3. Vesanto Clustering...")
    clustering_auto = SOMVesantoClustering(method='ward', auto_k_method='davies_bouldin')
    labels_auto = clustering_auto.fit(som, df)
    k_auto = clustering_auto.n_clusters_

    true_k = len(target_names)
    clustering_true = SOMVesantoClustering(method='ward')
    labels_true = clustering_true.fit(som, df, n_clusters=true_k)
    
    bmus = som.predict(df)
    m_auto = compute_all_metrics(som, labels_auto, df, y, bmus)
    m_true = compute_all_metrics(som, labels_true, df, y, bmus)
    
    print(f"\n   {'Method':<20} {'k':>3} {'NMI':>6} {'ARI':>6} {'Purity':>6}")
    print(f"   Auto k={k_auto:<13} {k_auto:>3} {m_auto['nmi']:>6.3f} {m_auto['ari']:>6.3f} {m_auto['purity']:>6.3f}")
    print(f"   True k={true_k:<13} {true_k:>3} {m_true['nmi']:>6.3f} {m_true['ari']:>6.3f} {m_true['purity']:>6.3f}")

    # 4. Visualizations
    print("\n4. Visualizations...")
    viz = DBGSOMVisualizer(Path(__file__).parent.parent / "output", "circles_combined")
    if viz.raw_data_path.exists() and not any(viz.raw_data_path.iterdir()):
        viz.raw_data_path.rmdir()

    viz.plot_expanded_umatrix(som, "umatrix_expanded.png")
    viz.plot_hit_counts(som, df, "hit_counts.png")
    viz.plot_labels_hex(som, df, y, target_names, "labels_hex.png")
    try:
        viz.plot_component_planes(som, features, "component_planes")
    except (AttributeError, NotImplementedError):
        pass

    print(f"   Saved plots to {viz.plots_path}")

    # 5. K Eval
    try:
        import matplotlib.pyplot as plt
        db, sil = clustering_auto.get_k_evaluation_profile()
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(sorted(db), [db[k] for k in sorted(db)], 'b-o'); ax[0].set_title('Davies-Bouldin')
        ax[1].plot(sorted(sil), [sil[k] for k in sorted(sil)], 'g-o'); ax[1].set_title('Silhouette')
        plt.tight_layout(); plt.savefig(viz.plots_path / "k_eval.png"); plt.close()
    except Exception: pass

    # 6. Dendrograms
    try:
        import matplotlib.pyplot as plt
        l_auto = clustering_auto.get_dendrogram_data()
        l_true = clustering_true.get_dendrogram_data()
        p_auto, a_auto, pc_auto = compute_cluster_purities(som, df, y, labels_auto, target_names)
        p_true, a_true, pc_true = compute_cluster_purities(som, df, y, labels_true, target_names)

        if l_auto is not None:
            fig, ax = plt.subplots(figsize=(10,6))
            plot_dendrogram_with_purity(ax, l_auto, k_auto, p_auto, a_auto, pc_auto, "Auto k")
            plt.savefig(viz.plots_path / "dendro_auto.png"); plt.close()

        if l_true is not None:
            fig, ax = plt.subplots(figsize=(10,6))
            plot_dendrogram_with_purity(ax, l_true, true_k, p_true, a_true, pc_true, "True k")
            plt.savefig(viz.plots_path / "dendro_true.png"); plt.close()
    except ImportError:
        pass

if __name__ == "__main__":
    main()
