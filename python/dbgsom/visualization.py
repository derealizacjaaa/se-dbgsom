"""
Visualization module for DBGSOM.

Creates plots and exports raw data for analysis.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

# Try to import matplotlib (optional dependency)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _ensure_matplotlib():
    """Ensure matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


class DBGSOMVisualizer:
    """
    Visualization and data export for DBGSOM models.

    Creates standardized output folders with plots and raw data.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory (e.g., 'examples/output').
    dataset_name : str
        Name of the dataset (used for folder naming).

    Examples
    --------
    >>> from dbgsom import DBGSOMWrapper, DBGSOMVisualizer
    >>>
    >>> som = DBGSOMWrapper(sf=0.6)
    >>> som.fit(df)
    >>>
    >>> viz = DBGSOMVisualizer('examples/output', 'iris')
    >>> viz.save_all(som, df)
    """

    def __init__(self, output_dir: Union[str, Path], dataset_name: str, style: str = 'modern'):
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.base_path = self.output_dir / dataset_name
        self.plots_path = self.base_path / "plots"
        self.raw_data_path = self.base_path / "raw_data"
        self.style = style

        # Create directories
        self.plots_path.mkdir(parents=True, exist_ok=True)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
        # Configure global style
        self._configure_style()

    def _configure_style(self):
        """Configure matplotlib style settings."""
        if not HAS_MATPLOTLIB:
            return
            
        plt.style.use('default')
        # Modern, clean look - matching 'Clean Laboratory' site theme
        font_family = 'monospace'
        
        # Site variable mappings:
        # --foreground: 222 47% 11% -> #0f172a (Slate-900)
        # --primary: 221 83% 53% -> #2563eb (Blue-600)
        # --muted-foreground: 215.4 16.3% 46.9% -> #64748b (Slate-500)
        # --border: 214.3 31.8% 91.4% -> #e2e8f0 (Slate-200)
        
        c_text = '#0f172a'
        c_subtext = '#64748b'
        c_grid = '#e2e8f0'
        c_primary = '#2563eb'

        params = {
            'font.family': font_family,
            'font.monospace': ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 14,  # Increased from 12
            'axes.titleweight': 'bold',
            'axes.titlepad': 15,   # Added padding
            'axes.labelcolor': c_text,
            'axes.titlecolor': c_text,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'xtick.color': c_subtext,
            'ytick.color': c_subtext,
            'text.color': c_text,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': c_grid,
            'axes.grid': False,
            'grid.alpha': 0.5,
            'grid.color': c_grid,
            'legend.frameon': False,
            'legend.fontsize': 9,
            'legend.labelcolor': c_text,
            # Set default color cycle to strict palette + complementary scientific colors
            'axes.prop_cycle': plt.cycler(color=[
                c_primary,  # Blue-600
                '#0ea5e9',  # Sky-500
                '#10b981',  # Emerald-500
                '#f59e0b',  # Amber-500
                '#ef4444',  # Red-500
                '#8b5cf6',  # Violet-500
                '#ec4899',  # Pink-500
                '#6366f1',  # Indigo-500
            ])
        }
        plt.rcParams.update(params)

    def save_all(
        self,
        som: Any,
        X: Union[pd.DataFrame, np.ndarray],
        labels: Optional[np.ndarray] = None,
        label_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Save all plots and raw data for a trained SOM.
        
        Parameters
        ----------
        som : DBGSOMWrapper
            Trained SOM model.
        X : DataFrame or ndarray
            Input data used for training.
        labels : ndarray, optional
            Class labels for coloring (if available).
        label_names : list of str, optional
            Names for each label value (e.g., ['Setosa', 'Versicolor', 'Virginica']).
        feature_names : list of str, optional
            Names of features for component planes.
            
        Returns
        -------
        result : dict
            Summary with cluster assignments.
        """
        _ensure_matplotlib()

        # Extract feature names
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                feature_names = [f"Feature {i+1}" for i in range(X.shape[1] if X.ndim > 1 else 1)]

        # Get clusters using U* matrix (U-matrix + density)
        clusters = som.cluster(X)
        n_clusters = len(set(clusters.values()))

        # Save plots
        self.plot_umatrix(som)
        self.plot_hit_counts(som, X)
        self.plot_component_planes(som, feature_names)

        # Use class distribution plot (pure/mixed) when labels available, else standard plots
        if labels is not None:
            class_stats = self.plot_class_distribution(
                som, X, labels, label_names, filename="clusters.png"
            )
            logger.info(f"Neuron purity: {class_stats['pure_neurons']} pure, {class_stats['mixed_neurons']} mixed, {class_stats['empty_neurons']} empty")
        else:
            self.plot_grid(som)
            self.plot_clusters(som, X)

        # Save raw data
        self.save_umatrix_data(som)
        self.save_neuron_weights(som, feature_names)
        self.save_clusters(som, X)

        # Save enhanced sample mappings with cluster info
        self._save_sample_mappings_with_clusters(
            som, X, clusters, labels, label_names
        )

        logger.info(f"Output saved to: {self.base_path}")
        logger.info(f"Clustering: {n_clusters} clusters (U* matrix)")

        # Calculate sample distribution across clusters
        bmus = som.predict(X)
        cluster_samples = {}
        cluster_labels = {} if labels is not None else None

        for i, bmu in enumerate(bmus):
            c = clusters[bmu]
            cluster_samples[c] = cluster_samples.get(c, 0) + 1
            if labels is not None:
                if c not in cluster_labels:
                    cluster_labels[c] = {}
                lbl = labels[i]
                cluster_labels[c][lbl] = cluster_labels[c].get(lbl, 0) + 1

        # Log cluster distribution
        logger.info("Cluster distribution:")
        for c in sorted(cluster_samples.keys()):
            count = cluster_samples[c]
            pct = count / len(bmus) * 100
            if cluster_labels and label_names:
                # Show label breakdown
                lbl_str = ", ".join(
                    f"{label_names[lbl]}={cnt}"
                    for lbl, cnt in sorted(cluster_labels[c].items())
                )
                logger.info(f"  Cluster {c}: {count} samples ({pct:.1f}%) - {lbl_str}")
            else:
                logger.info(f"  Cluster {c}: {count} samples ({pct:.1f}%)")

        # Calculate cluster purity if labels provided
        if labels is not None:
            total_correct = sum(
                max(lbl_counts.values()) for lbl_counts in cluster_labels.values()
            )
            purity = total_correct / len(labels) * 100
            logger.info(f"Cluster purity: {purity:.1f}%")

        return {
            'n_clusters': n_clusters,
            'clusters': clusters,
            'cluster_samples': cluster_samples
        }

    def _save_sample_mappings_with_clusters(
        self,
        som: Any,
        X: Union[pd.DataFrame, np.ndarray],
        clusters: Dict[Tuple[int, int], int],
        labels: Optional[np.ndarray] = None,
        label_names: Optional[List[str]] = None
    ) -> None:
        """Save BMU assignments with cluster info for each sample."""
        bmus, distances = som.predict_with_distances(X)

        rows = []
        for i, (bmu, dist) in enumerate(zip(bmus, distances)):
            row = {
                'sample_id': i,
                'bmu_x': bmu[0],
                'bmu_y': bmu[1],
                'distance': dist,
                'cluster': clusters[bmu]
            }
            if labels is not None:
                row['label'] = labels[i]
                if label_names is not None:
                    row['label_name'] = label_names[labels[i]]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.raw_data_path / "sample_mappings.csv", index=False)

    # =========================================================================
    # Plot Methods
    # =========================================================================

    def plot_grid(self, som: Any, filename: str = "grid.png") -> None:
        """Plot the SOM grid topology."""
        _ensure_matplotlib()

        positions, _ = som.get_neuron_weights()
        bounds = som.get_grid_bounds()
        (x_min, x_max), (y_min, y_max) = bounds

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot neurons as points
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax.scatter(xs, ys, s=200, c='steelblue', edgecolors='white', linewidths=1.5, zorder=3, alpha=0.9)

        # Plot grid connections
        pos_set = set(positions)
        for pos in positions:
            x, y = pos
            # Check right neighbor
            if (x + 1, y) in pos_set:
                ax.plot([x, x + 1], [y, y], '-', color='#bbbbbb', linewidth=1, zorder=1)
            # Check bottom neighbor
            if (x, y + 1) in pos_set:
                ax.plot([x, x], [y, y + 1], '-', color='#bbbbbb', linewidth=1, zorder=1)

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'SOM Grid ({som.n_neurons_} neurons)', pad=20)

        plt.tight_layout()
        plt.savefig(self.plots_path / filename, bbox_inches='tight')
        plt.close()

    def _hex_to_pixel(self, x: int, y: int, size: float = 1.0) -> Tuple[float, float]:
        """
        Convert odd-r offset hex coordinates to pixel coordinates.

        Parameters
        ----------
        x, y : int
            Hex grid coordinates (odd-r offset system)
        size : float
            Hex cell size

        Returns
        -------
        px, py : float
            Pixel coordinates for the hex center
        """
        px = size * 3/2 * x
        py = size * np.sqrt(3) * (y + 0.5 * (x & 1))
        return px, py

    def plot_umatrix(self, som: Any, filename: str = "umatrix.png") -> None:
        """Plot the U-matrix using hexagonal cells."""
        _ensure_matplotlib()
        from matplotlib.patches import RegularPolygon
        from matplotlib.collections import PatchCollection

        umat = som.get_umatrix()

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect('equal')

        # Collect hex patches
        patches = []
        colors = []

        hex_size = 1.0
        for (x, y), val in umat.items():
            px, py = self._hex_to_pixel(x, y, hex_size)
            hex_patch = RegularPolygon(
                (px, py), numVertices=6, radius=hex_size / np.sqrt(3),
                orientation=np.radians(30)  # Flat-top hexagons
            )
            patches.append(hex_patch)
            colors.append(val)

        collection = PatchCollection(
            patches, cmap='viridis', edgecolors='white', linewidths=0.5
        )
        collection.set_array(np.array(colors))
        ax.add_collection(collection)

        ax.autoscale_view()
        cbar = plt.colorbar(collection, ax=ax, label='U-matrix value', pad=0.02)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('U-matrix value', fontsize=11, fontweight='bold')
        
        ax.set_title('U-Matrix (Hexagonal)', pad=20, fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.plots_path / filename, bbox_inches='tight')
        plt.close()

    def plot_expanded_umatrix(
        self, som: Any, filename: str = "expanded_umatrix.png", log: bool = False
    ) -> None:
        """
        Plot expanded U-matrix with intermediate hexagons showing edge distances.

        The expanded U-matrix shows both neuron cells (larger hexagons with mean
        neighbor distance) and edge cells (smaller hexagons between adjacent
        neurons showing their specific pairwise distance).

        Parameters
        ----------
        som : DBGSOMWrapper or SEDBGSOM
            Trained SOM model with get_expanded_umatrix method.
        filename : str
            Output filename.
        log : bool
            If True, use log-transformed distances.
        """
        _ensure_matplotlib()
        from matplotlib.patches import RegularPolygon
        from matplotlib.collections import PatchCollection

        neuron_umat, edge_umat = som.get_expanded_umatrix(log=log)

        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_aspect('equal')

        hex_size = 1.0
        small_hex_size = hex_size * 0.5  # Intermediate hexes are smaller

        patches = []
        colors = []

        # Draw neuron hexagons (larger)
        spacing = hex_size * 1.5
        for (x, y), val in neuron_umat.items():
            px, py = self._hex_to_pixel(x, y, spacing)
            hex_patch = RegularPolygon(
                (px, py), numVertices=6, radius=hex_size / np.sqrt(3),
                orientation=np.radians(30)
            )
            patches.append(hex_patch)
            colors.append(val)

        # Draw edge hexagons (smaller, between neurons)
        for ((x1, y1), (x2, y2)), val in edge_umat.items():
            px1, py1 = self._hex_to_pixel(x1, y1, spacing)
            px2, py2 = self._hex_to_pixel(x2, y2, spacing)
            mid_px, mid_py = (px1 + px2) / 2, (py1 + py2) / 2

            hex_patch = RegularPolygon(
                (mid_px, mid_py), numVertices=6,
                radius=small_hex_size / np.sqrt(3),
                orientation=np.radians(30)
            )
            patches.append(hex_patch)
            colors.append(val)

        collection = PatchCollection(
            patches, cmap='magma_r', edgecolors='white', linewidths=0.2
        )
        collection.set_array(np.array(colors))
        ax.add_collection(collection)

        ax.autoscale_view()
        label = 'Distance (log)' if log else 'Distance'
        cbar = plt.colorbar(collection, ax=ax, label=label, pad=0.02)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(label, fontsize=11, fontweight='bold')
        
        ax.set_title('Expanded U-Matrix (Hexagonal)', pad=20, fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.plots_path / filename, bbox_inches='tight')
        plt.close()

    def plot_labels_hex(
        self,
        som: Any,
        X: Union[pd.DataFrame, np.ndarray],
        labels: np.ndarray,
        label_names: Optional[List[str]] = None,
        filename: str = "labels_hex.png"
    ) -> None:
        """
        Plot hexagonal SOM with neurons colored by dominant true label.

        Parameters
        ----------
        som : DBGSOMWrapper or SEDBGSOM
            Trained SOM model.
        X : DataFrame or ndarray
            Input data used for training.
        labels : ndarray
            True class labels for each sample (integers starting from 0).
        label_names : list of str, optional
            Names for each class (e.g., ['Setosa', 'Versicolor', 'Virginica']).
        filename : str
            Output filename.
        """
        _ensure_matplotlib()
        from matplotlib.patches import RegularPolygon
        from matplotlib.collections import PatchCollection
        from matplotlib.patheffects import withStroke
        from collections import Counter

        # Get BMU for each sample
        bmus = som.predict(X)

        # Count labels per neuron
        neuron_labels = {}
        for bmu, label in zip(bmus, labels):
            if bmu not in neuron_labels:
                neuron_labels[bmu] = []
            neuron_labels[bmu].append(label)

        # Find dominant label for each neuron
        neuron_dominant = {}
        neuron_purity = {}
        for pos, label_list in neuron_labels.items():
            counter = Counter(label_list)
            dominant_label, count = counter.most_common(1)[0]
            neuron_dominant[pos] = dominant_label
            neuron_purity[pos] = count / len(label_list)

        # Get all neuron positions
        umat = som.get_umatrix()
        all_positions = list(umat.keys())

        # Assign -1 to neurons with no samples
        for pos in all_positions:
            if pos not in neuron_dominant:
                neuron_dominant[pos] = -1
                neuron_purity[pos] = 0.0

        # Determine number of classes
        unique_labels = sorted(set(labels))
        n_classes = len(unique_labels)

        # Create colormap
        if n_classes <= 10:
            cmap = plt.cm.get_cmap('Set2', n_classes + 1)
        else:
            cmap = plt.cm.get_cmap('tab20', n_classes + 1)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect('equal')

        hex_size = 1.0
        patches = []
        colors = []

        for pos in all_positions:
            px, py = self._hex_to_pixel(pos[0], pos[1], hex_size)
            hex_patch = RegularPolygon(
                (px, py), numVertices=6, radius=hex_size / np.sqrt(3),
                orientation=np.radians(30)
            )
            patches.append(hex_patch)

            label = neuron_dominant[pos]
            if label == -1:
                colors.append(n_classes)  # Use last color for empty
            else:
                colors.append(label)

        # Create collection with discrete colors
        collection = PatchCollection(patches, edgecolors='white', linewidths=0.5)
        collection.set_array(np.array(colors))
        collection.set_cmap(cmap)
        collection.set_clim(-0.5, n_classes + 0.5)
        ax.add_collection(collection)

        # Add purity text inside neurons with samples
        for pos in all_positions:
            if neuron_dominant[pos] != -1:
                px, py = self._hex_to_pixel(pos[0], pos[1], hex_size)
                purity = neuron_purity[pos]
                # Only show purity if not 100%
                if purity < 1.0:
                    ax.text(px, py, f'{purity:.0%}', ha='center', va='center',
                            fontsize=7, fontweight='bold', color='white',
                            path_effects=[withStroke(linewidth=2, foreground='#333333')])

        ax.autoscale_view()

        # Create legend
        if label_names is None:
            label_names = [f'Class {i}' for i in unique_labels]

        legend_patches = []
        for i, name in enumerate(label_names):
            color = cmap(i)
            patch = mpatches.Patch(color=color, label=name)
            legend_patches.append(patch)

        # Add empty neurons to legend - REMOVED per user request
        # empty_patch = mpatches.Patch(color=cmap(n_classes), label='No samples')
        # legend_patches.append(empty_patch)

        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1), 
                 fontsize=14, title="Classes", title_fontsize=16, frameon=False)

        ax.set_title('SOM with True Labels (Hexagonal)', pad=20, fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.plots_path / filename, bbox_inches='tight')
        plt.close()

    def plot_hit_counts(
        self,
        som: Any,
        X: Union[pd.DataFrame, np.ndarray],
        filename: str = "hit_counts.png"
    ) -> None:
        """Plot hit counts (samples per neuron)."""
        _ensure_matplotlib()

        hits = som.get_hit_counts(X)
        bounds = som.get_grid_bounds()
        (x_min, x_max), (y_min, y_max) = bounds

        fig, ax = plt.subplots(figsize=(10, 8))

        positions = list(hits.keys())
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        sizes = [hits[p] * 30 + 100 for p in positions]  # Scale for visibility
        colors = [hits[p] for p in positions]

        scatter = ax.scatter(
            xs, ys, s=sizes, c=colors, cmap='Oranges',
            edgecolors='white', linewidths=1.5, alpha=0.9
        )

        # Add count labels
        for pos in positions:
            if hits[pos] > 0:
                ax.annotate(
                    str(hits[pos]), pos, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='#333333'
                )

        cbar = plt.colorbar(scatter, ax=ax, label='Hit count', pad=0.02)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Hit count', fontsize=11, fontweight='bold')

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Hit Counts (Samples per Neuron)', pad=20, fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_path / filename, bbox_inches='tight')
        plt.close()

    def plot_component_planes(
        self,
        som: Any,
        feature_names: List[str],
        filename_prefix: str = "component_plane"
    ) -> None:
        """Plot component planes for each feature using hexagonal grid."""
        _ensure_matplotlib()
        from matplotlib.patches import RegularPolygon
        from matplotlib.collections import PatchCollection

        positions, weights = som.get_neuron_weights()
        bounds = som.get_grid_bounds()
        (x_min, x_max), (y_min, y_max) = bounds
        
        n_features = weights.shape[1]
        
        # Grid settings
        hex_size = 1.0

        # Create a combined figure with all component planes
        n_cols = min(2, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
        if n_features == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Global min/max for common color scale (optional, generally per-feature is better)
        # using per-feature scaling here as features have different ranges

        for i in range(n_features):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]
            
            # Extract feature values for all neurons
            feature_values = weights[:, i]
            
            patches = []
            colors = []
            
            for j, pos in enumerate(positions):
                px, py = self._hex_to_pixel(pos[0], pos[1], hex_size)
                hex_patch = RegularPolygon(
                    (px, py), numVertices=6, radius=hex_size / np.sqrt(3),
                    orientation=np.radians(30)
                )
                patches.append(hex_patch)
                colors.append(feature_values[j])

            collection = PatchCollection(
                patches, cmap='viridis', edgecolors='white', linewidths=0.2
            )
            collection.set_array(np.array(colors))
            ax.add_collection(collection)
            
            ax.autoscale_view()
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Title with larger font
            fname = feature_names[i] if i < len(feature_names) else f"Feature {i+1}"
            ax.set_title(fname, fontsize=14, fontweight='bold', pad=10)
            
            # Individual colorbar
            cbar = plt.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=10)

        # Hide unused subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row, col].set_visible(False)

        plt.suptitle('Component Planes', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
        plt.savefig(self.plots_path / f"{filename_prefix}_all.png", bbox_inches='tight')
        plt.close()

    def plot_clusters(
        self,
        som: Any,
        X: Union[pd.DataFrame, np.ndarray],
        filename: str = "clusters.png"
    ) -> None:
        """Plot neuron clusters from U* matrix watershed."""
        _ensure_matplotlib()
        from matplotlib.patheffects import withStroke

        clusters = som.cluster(X)
        bounds = som.get_grid_bounds()
        (x_min, x_max), (y_min, y_max) = bounds

        fig, ax = plt.subplots(figsize=(10, 8))

        positions = list(clusters.keys())
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        colors = [clusters[p] for p in positions]

        n_clusters = len(set(colors))
        ax.scatter(
            xs, ys, s=300, c=colors, cmap='tab10',
            edgecolors='white', linewidths=1.5, alpha=0.9
        )

        # Add cluster labels
        for pos in positions:
            ax.annotate(
                str(clusters[pos]), pos, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white',
                path_effects=[withStroke(linewidth=2, foreground='#333333')]
            )

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Neuron Clusters ({n_clusters} clusters)', pad=20, fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_path / filename, bbox_inches='tight')
        plt.close()

    def plot_class_distribution(
        self,
        som: Any,
        X: Union[pd.DataFrame, np.ndarray],
        labels: np.ndarray,
        label_names: Optional[List[str]] = None,
        filename: str = "clusters.png"
    ) -> Dict[str, Any]:
        """
        Plot neurons colored by class with pure/mixed indicators and U-matrix overlay.

        This visualization replaces both grid.png and clusters.png when labels are provided.
        Shows:
        - Left panel: Pure neurons only (single class)
        - Middle panel: Mixed neurons only (multiple classes)
        - Right panel: All neurons with U-matrix background

        Parameters
        ----------
        som : DBGSOMWrapper
            Trained SOM model.
        X : DataFrame or ndarray
            Input data.
        labels : ndarray
            Class labels for each sample.
        label_names : list of str, optional
            Names for each class (e.g., ['Setosa', 'Versicolor', 'Virginica']).
        filename : str
            Output filename.

        Returns
        -------
        stats : dict
            Statistics about pure/mixed neurons.
        """
        _ensure_matplotlib()
        from collections import defaultdict
        from scipy.interpolate import griddata
        from matplotlib.patheffects import withStroke

        positions, weights = som.get_neuron_weights()
        umat = som.get_umatrix()
        bmus = som.predict(X)
        bounds = som.get_grid_bounds()
        (x_min, x_max), (y_min, y_max) = bounds

        # Count samples per class for each neuron
        neuron_class_counts = defaultdict(lambda: defaultdict(int))
        for i, bmu in enumerate(bmus):
            pos = tuple(bmu)
            neuron_class_counts[pos][labels[i]] += 1

        # Get unique classes
        unique_classes = sorted(set(labels))
        n_classes = len(unique_classes)

        # Default class names
        if label_names is None:
            label_names = [f"Class {c}" for c in unique_classes]

        # Colors for classes (use tab10 colormap)
        cmap = plt.cm.tab10
        class_colors = {c: cmap(i / max(n_classes - 1, 1)) for i, c in enumerate(unique_classes)}

        # Classify neurons as pure or mixed
        pure_neurons = []  # (pos, class_id, count)
        mixed_neurons = []  # (pos, dominant_class, counts_dict, purity)
        empty_neurons = []

        for pos in positions:
            counts = neuron_class_counts[pos]
            if len(counts) == 0:
                empty_neurons.append(pos)
                continue

            total = sum(counts.values())
            dominant_class = max(counts.keys(), key=lambda c: counts[c])
            dominant_count = counts[dominant_class]

            if len(counts) == 1:
                pure_neurons.append((pos, dominant_class, total))
            else:
                purity = dominant_count / total
                mixed_neurons.append((pos, dominant_class, dict(counts), purity))

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Get grid bounds with padding
        x_pad = (x_max - x_min) * 0.1 + 1
        y_pad = (y_max - y_min) * 0.1 + 1

        # Plot 1: Pure neurons only
        ax1 = axes[0]
        ax1.set_title(f"Pure Neurons ({len(pure_neurons)})\n(single class only)", fontsize=11, fontweight='bold')

        # Draw all neurons as light gray background
        for pos in positions:
            ax1.scatter(pos[0], pos[1], s=80, c='#f0f0f0', marker='s')

        # Draw pure neurons
        for pos, class_id, count in pure_neurons:
            ax1.scatter(pos[0], pos[1], s=40 + count * 8, c=[class_colors[class_id]],
                       edgecolors='white', linewidths=0.5, marker='o', alpha=0.9)

        ax1.set_xlim(x_min - x_pad, x_max + x_pad)
        ax1.set_ylim(y_min - y_pad, y_max + y_pad)
        ax1.set_aspect('equal')
        ax1.axis('off')

        # Legend for classes
        for c in unique_classes:
            name = label_names[c] if c < len(label_names) else f"Class {c}"
            ax1.scatter([], [], c=[class_colors[c]], label=name, s=100)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=9, frameon=False, ncol=min(3, n_classes))

        # Plot 2: Mixed neurons only
        ax2 = axes[1]
        ax2.set_title(f"Mixed Neurons ({len(mixed_neurons)})\n(multiple classes)", fontsize=11, fontweight='bold')

        # Draw all neurons as light gray background
        for pos in positions:
            ax2.scatter(pos[0], pos[1], s=80, c='#f0f0f0', marker='s')

        # Draw mixed neurons with thick border (more mixed = thicker)
        for pos, dominant_class, counts, purity in mixed_neurons:
            total = sum(counts.values())
            size = 40 + total * 8
            # Make edge color indicate it's mixed (e.g., darker version of dominant color or black)
            edge_width = 2.5 * (1 - purity) + 0.5
            ax2.scatter(pos[0], pos[1], s=size, c=[class_colors[dominant_class]],
                       edgecolors='#333333', linewidths=edge_width, marker='o', alpha=0.9)

            # Add small text showing composition
            if total < 100:  # Only if not too crowded
                ax2.text(pos[0], pos[1], f"{purity:.0%}", fontsize=6, 
                        ha='center', va='center', color='white', fontweight='bold',
                        path_effects=[withStroke(linewidth=1.5, foreground='#333333')])

        ax2.set_xlim(x_min - x_pad, x_max + x_pad)
        ax2.set_ylim(y_min - y_pad, y_max + y_pad)
        ax2.set_aspect('equal')
        ax2.axis('off')

        # Plot 3: All neurons with U-matrix background
        ax3 = axes[2]
        ax3.set_title(f"All Neurons ({som.n_neurons_})\n(on U-matrix)", fontsize=11, fontweight='bold')

        # Draw U-matrix background (simplified)
        # Interpolate U-matrix to grid
        xi = np.linspace(x_min, x_max, 100)
        yi = np.linspace(y_min, y_max, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Prepare data for interpolation
        u_points = np.array(list(umat.keys()))
        u_values = np.array(list(umat.values()))
        
        # Griddata interpolation
        Zi = griddata(u_points, u_values, (Xi, Yi), method='cubic', fill_value=np.mean(u_values))
        
        # Plot contour background
        ax3.contourf(Xi, Yi, Zi, levels=20, cmap='Greys', alpha=0.3)

        # Draw all neurons on top
        for pos, class_id, count in pure_neurons:
             ax3.scatter(pos[0], pos[1], s=30 + count * 6, c=[class_colors[class_id]],
                        edgecolors='white', linewidths=0.5, alpha=0.9)
                        
        for pos, dominant_class, counts, purity in mixed_neurons:
            total = sum(counts.values())
            size = 30 + total * 6
            edge_width = 2 * (1 - purity) + 0.5
            ax3.scatter(pos[0], pos[1], s=size, c=[class_colors[dominant_class]],
                       edgecolors='#333333', linewidths=edge_width, alpha=0.9)

        ax3.set_xlim(x_min - x_pad, x_max + x_pad)
        ax3.set_ylim(y_min - y_pad, y_max + y_pad)
        ax3.set_aspect('equal')
        ax3.axis('off')

        plt.tight_layout()
        plt.savefig(self.plots_path / filename, bbox_inches='tight', dpi=150)
        plt.close()

        # Return statistics
        stats = {
            'pure_neurons': len(pure_neurons),
            'mixed_neurons': len(mixed_neurons),
            'empty_neurons': len(empty_neurons),
            'total_neurons': len(positions),
            'pure_by_class': {},
            'mixed_details': []
        }

        for c in unique_classes:
            name = label_names[c] if c < len(label_names) else f"Class {c}"
            count = sum(1 for _, cls, _ in pure_neurons if cls == c)
            samples = sum(cnt for _, cls, cnt in pure_neurons if cls == c)
            stats['pure_by_class'][name] = {'neurons': count, 'samples': samples}

        for pos, dominant_class, counts, purity in mixed_neurons:
            stats['mixed_details'].append({
                'position': pos,
                'counts': counts,
                'purity': purity
            })

        return stats

    # =========================================================================
    # Raw Data Export Methods
    # =========================================================================

    def save_umatrix_data(self, som: Any, filename: str = "umatrix.csv") -> None:
        """Save U-matrix values to CSV."""
        umat = som.get_umatrix()

        rows = []
        for (x, y), val in umat.items():
            rows.append({'x': x, 'y': y, 'umatrix_value': val})

        df = pd.DataFrame(rows)
        df.to_csv(self.raw_data_path / filename, index=False)

    def save_sample_mappings(
        self,
        som: Any,
        X: Union[pd.DataFrame, np.ndarray],
        labels: Optional[np.ndarray] = None,
        filename: str = "sample_mappings.csv"
    ) -> None:
        """Save BMU assignments for each sample."""
        bmus, distances = som.predict_with_distances(X)

        rows = []
        for i, (bmu, dist) in enumerate(zip(bmus, distances)):
            row = {
                'sample_idx': i,
                'bmu_x': bmu[0],
                'bmu_y': bmu[1],
                'distance': dist
            }
            if labels is not None:
                row['label'] = labels[i]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.raw_data_path / filename, index=False)

    def save_neuron_weights(
        self,
        som: Any,
        feature_names: List[str],
        filename: str = "neuron_weights.csv"
    ) -> None:
        """Save neuron weight vectors."""
        positions, weights = som.get_neuron_weights()

        rows = []
        for i, (pos, weight) in enumerate(zip(positions, weights)):
            row = {'x': pos[0], 'y': pos[1]}
            for j, w in enumerate(weight):
                fname = feature_names[j] if j < len(feature_names) else f"feature_{j}"
                row[fname] = w
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.raw_data_path / filename, index=False)

    def save_clusters(
        self,
        som: Any,
        X: Union[pd.DataFrame, np.ndarray],
        filename: str = "clusters.csv"
    ) -> None:
        """Save cluster assignments for neurons."""
        clusters = som.cluster(X)

        rows = []
        for (x, y), cluster_id in clusters.items():
            rows.append({'x': x, 'y': y, 'cluster': cluster_id})

        df = pd.DataFrame(rows)
        df.to_csv(self.raw_data_path / filename, index=False)


class PanelDataVisualizer(DBGSOMVisualizer):
    """
    Visualization for panel/time-series data with sliding windows.

    Extends DBGSOMVisualizer with trajectory tracking and per-window outputs.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory.
    dataset_name : str
        Name of the dataset.

    Examples
    --------
    >>> viz = PanelDataVisualizer('examples/output', 'countries_panel')
    >>> viz.save_all_windows(som, windows_data, entity_ids)
    >>> viz.plot_trajectory(som, entity_trajectories, entity_name='Germany')
    """

    def __init__(self, output_dir: Union[str, Path], dataset_name: str):
        super().__init__(output_dir, dataset_name)
        self.trajectories_path = self.base_path / "trajectories"
        self.windows_path = self.base_path / "windows"
        self.trajectories_path.mkdir(parents=True, exist_ok=True)
        self.windows_path.mkdir(parents=True, exist_ok=True)

    def save_window(
        self,
        som: Any,
        X_window: Union[pd.DataFrame, np.ndarray],
        window_idx: int,
        labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """Save plots and data for a single window."""
        _ensure_matplotlib()

        window_name = f"window_{window_idx:03d}"
        window_plots = self.windows_path / window_name / "plots"
        window_data = self.windows_path / window_name / "raw_data"
        window_plots.mkdir(parents=True, exist_ok=True)
        window_data.mkdir(parents=True, exist_ok=True)

        # Temporarily change paths
        orig_plots = self.plots_path
        orig_data = self.raw_data_path
        self.plots_path = window_plots
        self.raw_data_path = window_data

        # Save window outputs
        if feature_names is None:
            if isinstance(X_window, pd.DataFrame):
                feature_names = list(X_window.columns)
            else:
                n_features = X_window.shape[1] if X_window.ndim > 1 else som.n_features_in_
                feature_names = [f"Feature {i+1}" for i in range(n_features)]

        self.plot_grid(som)
        self.plot_umatrix(som)
        self.plot_hit_counts(som, X_window)
        self.save_umatrix_data(som)
        self.save_sample_mappings(som, X_window, labels)

        # Restore paths
        self.plots_path = orig_plots
        self.raw_data_path = orig_data

    def save_trajectory(
        self,
        som: Any,
        entity_windows: List[np.ndarray],
        entity_id: str,
        time_labels: Optional[List[str]] = None
    ) -> None:
        """
        Save trajectory data for a single entity across time windows.

        Parameters
        ----------
        som : DBGSOMWrapper
            Trained SOM model.
        entity_windows : list of ndarray
            List of feature vectors for each time window.
        entity_id : str
            Identifier for the entity.
        time_labels : list of str, optional
            Labels for each time point.
        """
        bmus = []
        distances = []

        for window_data in entity_windows:
            # Reshape if needed
            if window_data.ndim == 1:
                window_data = window_data.reshape(1, -1)
            bmu_list, dist_list = som.predict_with_distances(window_data)
            bmus.append(bmu_list[0])
            distances.append(dist_list[0])

        # Save trajectory data
        rows = []
        for i, (bmu, dist) in enumerate(zip(bmus, distances)):
            row = {
                'entity_id': entity_id,
                'time_idx': i,
                'bmu_x': bmu[0],
                'bmu_y': bmu[1],
                'distance': dist
            }
            if time_labels and i < len(time_labels):
                row['time_label'] = time_labels[i]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.trajectories_path / f"trajectory_{entity_id}.csv", index=False)

        return bmus

    def plot_trajectory(
        self,
        som: Any,
        entity_windows: List[np.ndarray],
        entity_id: str,
        time_labels: Optional[List[str]] = None,
        filename: Optional[str] = None
    ) -> None:
        """
        Plot trajectory of an entity on the SOM grid.

        Parameters
        ----------
        som : DBGSOMWrapper
            Trained SOM model.
        entity_windows : list of ndarray
            List of feature vectors for each time window.
        entity_id : str
            Identifier for the entity.
        time_labels : list of str, optional
            Labels for each time point.
        filename : str, optional
            Output filename. Defaults to 'trajectory_{entity_id}.png'.
        """
        _ensure_matplotlib()

        # Get BMUs for trajectory
        bmus = self.save_trajectory(som, entity_windows, entity_id, time_labels)

        bounds = som.get_grid_bounds()
        (x_min, x_max), (y_min, y_max) = bounds

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot all neurons as background
        positions, _ = som.get_neuron_weights()
        all_xs = [p[0] for p in positions]
        all_ys = [p[1] for p in positions]
        ax.scatter(all_xs, all_ys, s=100, c='lightgray', edgecolors='gray', alpha=0.5, zorder=1)

        # Plot trajectory path
        traj_xs = [b[0] for b in bmus]
        traj_ys = [b[1] for b in bmus]

        # Draw path with arrows
        for i in range(len(bmus) - 1):
            ax.annotate(
                '', xy=(traj_xs[i+1], traj_ys[i+1]), xytext=(traj_xs[i], traj_ys[i]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                zorder=2
            )

        # Plot trajectory points with time coloring
        colors = np.arange(len(bmus))
        scatter = ax.scatter(
            traj_xs, traj_ys, s=200, c=colors, cmap='viridis',
            edgecolors='black', linewidths=2, zorder=3
        )

        # Add time labels
        for i, (x, y) in enumerate(zip(traj_xs, traj_ys)):
            label = time_labels[i] if time_labels and i < len(time_labels) else str(i)
            ax.annotate(
                label, (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold'
            )

        plt.colorbar(scatter, ax=ax, label='Time step')

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Trajectory: {entity_id}')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filename is None:
            filename = f"trajectory_{entity_id}.png"
        plt.savefig(self.trajectories_path / filename, dpi=150)
        plt.close()

    def plot_all_trajectories(
        self,
        som: Any,
        entity_data: Dict[str, List[np.ndarray]],
        time_labels: Optional[List[str]] = None,
        filename: str = "all_trajectories.png"
    ) -> None:
        """
        Plot all entity trajectories on a single SOM grid.

        Parameters
        ----------
        som : DBGSOMWrapper
            Trained SOM model.
        entity_data : dict
            Mapping of entity_id -> list of feature vectors per time window.
        time_labels : list of str, optional
            Labels for time points.
        filename : str
            Output filename.
        """
        _ensure_matplotlib()

        bounds = som.get_grid_bounds()
        (x_min, x_max), (y_min, y_max) = bounds

        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot all neurons as background
        positions, _ = som.get_neuron_weights()
        all_xs = [p[0] for p in positions]
        all_ys = [p[1] for p in positions]
        ax.scatter(all_xs, all_ys, s=100, c='lightgray', edgecolors='gray', alpha=0.3, zorder=1)

        # Color map for entities
        colors = plt.cm.tab20(np.linspace(0, 1, len(entity_data)))

        legend_handles = []
        for idx, (entity_id, windows) in enumerate(entity_data.items()):
            # Get BMUs
            bmus = []
            for window_data in windows:
                if window_data.ndim == 1:
                    window_data = window_data.reshape(1, -1)
                bmu_list, _ = som.predict_with_distances(window_data)
                bmus.append(bmu_list[0])

            traj_xs = [b[0] for b in bmus]
            traj_ys = [b[1] for b in bmus]

            # Draw path
            ax.plot(traj_xs, traj_ys, '-', color=colors[idx], linewidth=1.5, alpha=0.7, zorder=2)

            # Mark start and end
            ax.scatter([traj_xs[0]], [traj_ys[0]], s=100, c=[colors[idx]], marker='o', edgecolors='black', zorder=3)
            ax.scatter([traj_xs[-1]], [traj_ys[-1]], s=100, c=[colors[idx]], marker='s', edgecolors='black', zorder=3)

            legend_handles.append(mpatches.Patch(color=colors[idx], label=entity_id))

        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('All Entity Trajectories (o=start, □=end)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.trajectories_path / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def save_all_trajectories_data(
        self,
        som: Any,
        entity_data: Dict[str, List[np.ndarray]],
        time_labels: Optional[List[str]] = None,
        filename: str = "all_trajectories.csv"
    ) -> None:
        """Save all trajectory data to a single CSV."""
        all_rows = []

        for entity_id, windows in entity_data.items():
            for i, window_data in enumerate(windows):
                if window_data.ndim == 1:
                    window_data = window_data.reshape(1, -1)
                bmu_list, dist_list = som.predict_with_distances(window_data)
                bmu = bmu_list[0]
                dist = dist_list[0]

                row = {
                    'entity_id': entity_id,
                    'time_idx': i,
                    'bmu_x': bmu[0],
                    'bmu_y': bmu[1],
                    'distance': dist
                }
                if time_labels and i < len(time_labels):
                    row['time_label'] = time_labels[i]
                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        df.to_csv(self.trajectories_path / filename, index=False)
