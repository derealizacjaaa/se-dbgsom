
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dbgsom.visualization import DBGSOMVisualizer

def main():
    # 1. Setup paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "circles.csv"
    output_dir = base_dir / "output"
    
    # 2. Load Data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 3. Initialize Visualizer to apply styles
    # Use "circles_combined" to match the requested output folder
    viz = DBGSOMVisualizer(output_dir, "circles_combined")
    
    # 4. Create Plot
    print("Creating plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique targets for coloring
    targets = df['target'].unique()
    targets.sort()
    
    # Use the color cycle from the style
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for i, t in enumerate(targets):
        mask = df['target'] == t
        color = colors[i % len(colors)]
        label = f"Class {t}"
        
        ax.scatter(
            df.loc[mask, 'x1'], 
            df.loc[mask, 'x2'], 
            c=color, 
            label=label,
            s=100, 
            alpha=0.8, 
            edgecolors='white', 
            linewidths=1.5
        )
        
    ax.set_title("Circles Dataset (Raw Data)", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(title="Target", loc='upper right', frameon=False)
    
    # 5. Save to the requested subdirectory
    output_file = viz.plots_path / "circles_data.png"
    print(f"Saving plot to {output_file}...")
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    # --- K-Means Section ---
    print("Running K-Means clustering...")
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import contingency_matrix
    import numpy as np

    X = df[['x1', 'x2']].values
    y_true = df['target'].values
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    # Calculate Purity
    # Purity = sum(max(contingency_matrix)) / n_samples
    cm = contingency_matrix(y_true, y_pred)
    purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    print(f"K-Means Purity: {purity:.4f}")
    
    # Create K-Means Plot
    print("Creating K-Means plot...")
    fig_km, ax_km = plt.subplots(figsize=(10, 8))
    
    # Get unique predicted clusters
    clusters = np.unique(y_pred)
    clusters.sort()
    
    for i, c in enumerate(clusters):
        mask = y_pred == c
        color = colors[i % len(colors)]
        label = f"Cluster {c}"
        
        ax_km.scatter(
            X[mask, 0], 
            X[mask, 1], 
            c=color, 
            label=label,
            s=100, 
            alpha=0.8, 
            edgecolors='white', 
            linewidths=1.5
        )
        
    ax_km.set_title("Circles Dataset (K-Means Clustering)", fontsize=16, fontweight='bold', pad=20)
    ax_km.set_xlabel("x1")
    ax_km.set_ylabel("x2")
    ax_km.legend(title="Cluster", loc='upper right', frameon=False)
    
    # Annotate Purity
    ax_km.text(
        0.02, 0.98, 
        f"Purity: {purity:.4f}", 
        transform=ax_km.transAxes, 
        fontsize=12, 
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    output_file_km = viz.plots_path / "circles_kmeans.png"
    print(f"Saving K-Means plot to {output_file_km}...")
    plt.tight_layout()
    plt.savefig(output_file_km, bbox_inches='tight')
    plt.close()

    # --- Optimal K Selection Section ---
    print("Running Optimal K Selection (Elbow & Silhouette)...")
    from sklearn.metrics import silhouette_score
    
    inertias = []
    sil_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))
        
    # Create K-Eval Plot
    print("Creating K-Eval plot...")
    fig_eval, ax_eval = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Inertia (Elbow Method)
    ax_eval[0].plot(k_range, inertias, 'bo-')
    ax_eval[0].set_title('Elbow Method (Inertia)', fontsize=14, fontweight='bold')
    ax_eval[0].set_xlabel('Number of clusters (k)')
    ax_eval[0].set_ylabel('Inertia')
    ax_eval[0].grid(True, alpha=0.3)
    
    # Plot Silhouette Score
    ax_eval[1].plot(k_range, sil_scores, 'go-')
    ax_eval[1].set_title('Silhouette Score', fontsize=14, fontweight='bold')
    ax_eval[1].set_xlabel('Number of clusters (k)')
    ax_eval[1].set_ylabel('Silhouette Score')
    ax_eval[1].grid(True, alpha=0.3)
    
    output_file_eval = viz.plots_path / "circles_kmeans_k_eval.png"
    print(f"Saving K-Eval plot to {output_file_eval}...")
    plt.tight_layout()
    plt.savefig(output_file_eval, bbox_inches='tight')
    plt.close()

    # --- Optimal K Clustering Section ---
    print("Running Optimal K Clustering...")
    # Find k with highest Silhouette Score
    best_k_idx = np.argmax(sil_scores)
    best_k = k_range[best_k_idx]
    print(f"Optimal k determined as: {best_k}")
    
    # Run K-Means with optimal k
    km_opt = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    y_pred_opt = km_opt.fit_predict(X)
    
    # Calculate Purity for optimal k
    cm_opt = contingency_matrix(y_true, y_pred_opt)
    purity_opt = np.sum(np.amax(cm_opt, axis=0)) / np.sum(cm_opt)
    print(f"Optimal K-Means Purity (k={best_k}): {purity_opt:.4f}")
    
    # Create Optimal K Plot
    print(f"Creating Optimal K (k={best_k}) plot...")
    fig_opt, ax_opt = plt.subplots(figsize=(10, 8))
    
    clusters_opt = np.unique(y_pred_opt)
    clusters_opt.sort()
    
    for i, c in enumerate(clusters_opt):
        mask = y_pred_opt == c
        color = colors[i % len(colors)]
        label = f"Cluster {c}"
        
        ax_opt.scatter(
            X[mask, 0], 
            X[mask, 1], 
            c=color, 
            label=label,
            s=100, 
            alpha=0.8, 
            edgecolors='white', 
            linewidths=1.5
        )
        
    ax_opt.set_title(f"K-Means Optimal Clustering (k={best_k})", fontsize=16, fontweight='bold', pad=20)
    ax_opt.set_xlabel("x1")
    ax_opt.set_ylabel("x2")
    ax_opt.legend(title="Cluster", loc='upper right', frameon=False)
    
    # Annotate Purity and k
    stats_text = f"k={best_k}\nPurity: {purity_opt:.4f}"
    ax_opt.text(
        0.02, 0.98, 
        stats_text, 
        transform=ax_opt.transAxes, 
        fontsize=12, 
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    output_file_opt = viz.plots_path / "circles_kmeans_optimal.png"
    print(f"Saving Optimal K plot to {output_file_opt}...")
    plt.tight_layout()
    plt.savefig(output_file_opt, bbox_inches='tight')
    plt.close()

    # --- K=15 Clustering Section ---
    k_fixed = 15
    print(f"Running K={k_fixed} Clustering...")
    
    # Run K-Means with k=15
    km_15 = KMeans(n_clusters=k_fixed, random_state=42, n_init=10)
    y_pred_15 = km_15.fit_predict(X)
    
    # Calculate Purity for k=15
    cm_15 = contingency_matrix(y_true, y_pred_15)
    purity_15 = np.sum(np.amax(cm_15, axis=0)) / np.sum(cm_15)
    print(f"K-Means Purity (k={k_fixed}): {purity_15:.4f}")
    
    # Create K=15 Plot
    print(f"Creating K={k_fixed} plot...")
    fig_15, ax_15 = plt.subplots(figsize=(10, 8))
    
    clusters_15 = np.unique(y_pred_15)
    clusters_15.sort()
    
    # Use a larger color map if needed, but cycle logic handles it
    for i, c in enumerate(clusters_15):
        mask = y_pred_15 == c
        color = colors[i % len(colors)]
        label = f"Cluster {c}"
        
        ax_15.scatter(
            X[mask, 0], 
            X[mask, 1], 
            c=color, 
            label=label,
            s=100, 
            alpha=0.8, 
            edgecolors='white', 
            linewidths=1.5
        )
        
    ax_15.set_title(f"K-Means Clustering (k={k_fixed})", fontsize=16, fontweight='bold', pad=20)
    ax_15.set_xlabel("x1")
    ax_15.set_ylabel("x2")
    # Legend might be too big for 15 clusters, but we'll include it.
    ax_15.legend(title="Cluster", loc='upper right', frameon=False, ncol=2) 
    
    # Annotate Purity and k
    stats_text = f"k={k_fixed}\nPurity: {purity_15:.4f}"
    ax_15.text(
        0.02, 0.98, 
        stats_text, 
        transform=ax_15.transAxes, 
        fontsize=12, 
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    output_file_15 = viz.plots_path / f"circles_kmeans_{k_fixed}.png"
    print(f"Saving K={k_fixed} plot to {output_file_15}...")
    plt.tight_layout()
    plt.savefig(output_file_15, bbox_inches='tight')
    plt.close()

    print("Done!")

if __name__ == "__main__":
    main()
