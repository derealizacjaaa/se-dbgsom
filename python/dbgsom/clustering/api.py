"""
High-level clustering API for SE-DBGSOM.

Provides simple functions to cluster trained SOM neurons:
- cluster_acute(): Best for non-convex shapes, complex boundaries
- cluster_leiden(): Best for convex/blob-like clusters
- cluster_vesanto(): Best for convex clusters with dendrogram visualization

Usage:
    from dbgsom import SEDBGSOM, cluster_acute, cluster_leiden, cluster_vesanto

    som = SEDBGSOM(lambda_=1.5, max_neurons=100, n_iter=200)
    som.fit(X)

    # For complex boundaries (moons, circles, irregular shapes)
    labels = cluster_acute(som, X)

    # For convex clusters (blobs, gaussians)
    labels = cluster_leiden(som, X, n_clusters=3)

    # For convex clusters with dendrogram visualization
    labels = cluster_vesanto(som, X)  # auto k via Davies-Bouldin
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, List

from .acute import SOMAcuteClustering
from .leiden import SOMLeidenClustering, HAS_LEIDEN
from .vesanto import SOMVesantoClustering


def cluster_acute(
    som,
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    n_clusters: Optional[int] = None,
    auto_method: str = 'combined',
    min_clusters: int = 2,
    max_clusters: int = 15
) -> Dict[Tuple[int, int], int]:
    """
    Cluster SOM neurons using ACUTE algorithm.

    ACUTE (Agglomerative Clustering Using Topological Relations) works by
    growing "balls" around neurons until they touch, using topological
    predicates to merge clusters. Excellent for non-convex shapes.

    Best for:
    - Two moons, circles, complex shapes
    - Data with non-linear boundaries
    - When you want automatic k detection

    Parameters
    ----------
    som : SEDBGSOM or DBGSOMWrapper
        Trained SOM model.
    X : DataFrame or ndarray, optional
        Input data (used for API consistency, not required).
    n_clusters : int, optional
        Target number of clusters. If None, automatically detected.
    auto_method : str, default='combined'
        Method for auto k selection: 'silhouette', 'stability', 'combined'.
    min_clusters : int, default=2
        Minimum clusters to consider for auto detection.
    max_clusters : int, default=15
        Maximum clusters to consider for auto detection.

    Returns
    -------
    labels : dict
        Mapping of neuron position (x, y) to cluster label (1-indexed).

    Examples
    --------
    >>> from dbgsom import SEDBGSOM, cluster_acute
    >>> som = SEDBGSOM(lambda_=1.5, max_neurons=100, n_iter=200).fit(X)
    >>>
    >>> # Auto-detect number of clusters
    >>> labels = cluster_acute(som, X)
    >>>
    >>> # Force specific number of clusters
    >>> labels = cluster_acute(som, X, n_clusters=3)
    """
    config = SOMAcuteClustering(
        auto_method=auto_method,
        min_clusters=min_clusters,
        max_clusters=max_clusters
    )
    return config.fit(som, X, n_clusters=n_clusters)


def cluster_leiden(
    som,
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    n_clusters: Optional[int] = None,
    resolution: float = 1.0,
    n_neighbors: int = 10,
    use_topology: bool = True
) -> Dict[Tuple[int, int], int]:
    """
    Cluster SOM neurons using Leiden community detection.

    Builds a k-NN graph from neuron weights and applies modularity
    optimization to find communities. Works well for convex clusters.

    Best for:
    - Blob-like, Gaussian clusters
    - Well-separated convex groups
    - When you know the target number of clusters

    Parameters
    ----------
    som : SEDBGSOM or DBGSOMWrapper
        Trained SOM model.
    X : DataFrame or ndarray, optional
        Input data (used for API consistency, not required).
    n_clusters : int, optional
        Target number of clusters. If None, uses resolution parameter.
    resolution : float, default=1.0
        Resolution parameter for modularity optimization.
        Higher = more clusters. Only used if n_clusters is None.
    n_neighbors : int, default=10
        Number of neighbors for k-NN graph construction.
    use_topology : bool, default=True
        Whether to include SOM grid topology edges.

    Returns
    -------
    labels : dict
        Mapping of neuron position (x, y) to cluster label (1-indexed).

    Examples
    --------
    >>> from dbgsom import SEDBGSOM, cluster_leiden
    >>> som = SEDBGSOM(lambda_=1.5, max_neurons=100, n_iter=200).fit(X)
    >>>
    >>> # Auto with default resolution
    >>> labels = cluster_leiden(som, X)
    >>>
    >>> # Target specific number of clusters
    >>> labels = cluster_leiden(som, X, n_clusters=5)
    >>>
    >>> # Adjust resolution for more/fewer clusters
    >>> labels = cluster_leiden(som, X, resolution=0.5)  # fewer clusters
    """
    if not HAS_LEIDEN:
        raise ImportError(
            "Leiden clustering requires igraph and leidenalg. "
            "Install with: pip install igraph leidenalg"
        )

    config = SOMLeidenClustering(
        n_neighbors=n_neighbors,
        resolution=resolution,
        use_topology=use_topology,
        random_state=42
    )
    return config.fit(som, X, n_clusters=n_clusters)


def cluster_vesanto(
    som,
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    n_clusters: Optional[int] = None,
    method: str = 'ward',
    auto_k_method: str = 'davies_bouldin',
    min_clusters: int = 2,
    max_clusters: int = 15
) -> Dict[Tuple[int, int], int]:
    """
    Cluster SOM neurons using Vesanto-Alhoniemi method.

    Two-level approach: clusters trained SOM prototypes using Ward's
    hierarchical clustering or K-means. Automatic k detection uses
    Davies-Bouldin index (minimize).

    Best for:
    - Convex, well-separated clusters
    - When dendrogram visualization is desired
    - When Davies-Bouldin index is preferred over silhouette

    Reference: Vesanto & Alhoniemi, "Clustering of the Self-Organizing Map",
    IEEE Trans. Neural Networks, 2000.

    Parameters
    ----------
    som : SEDBGSOM or DBGSOMWrapper
        Trained SOM model.
    X : DataFrame or ndarray, optional
        Input data (not used, for API consistency).
    n_clusters : int, optional
        Target number of clusters. If None, auto-detect using auto_k_method.
    method : str, default='ward'
        Clustering method:
        - 'ward': Agglomerative hierarchical clustering (recommended)
        - 'kmeans': K-means partitive clustering
    auto_k_method : str, default='davies_bouldin'
        Method for automatic k selection:
        - 'davies_bouldin': Minimize Davies-Bouldin index (recommended)
        - 'silhouette': Maximize silhouette score
    min_clusters : int, default=2
        Minimum k for auto detection.
    max_clusters : int, default=15
        Maximum k for auto detection.

    Returns
    -------
    Dict[Tuple[int, int], int]
        Mapping of neuron position (x, y) to cluster label (1-indexed).

    Examples
    --------
    >>> from dbgsom import SEDBGSOM, cluster_vesanto
    >>> som = SEDBGSOM(lambda_=1.5, max_neurons=100, n_iter=200).fit(X)
    >>>
    >>> # Auto-detect number of clusters
    >>> labels = cluster_vesanto(som, X)
    >>>
    >>> # Specify exact number of clusters
    >>> labels = cluster_vesanto(som, X, n_clusters=3)
    >>>
    >>> # Use K-means instead of Ward's method
    >>> labels = cluster_vesanto(som, X, method='kmeans', n_clusters=3)
    """
    config = SOMVesantoClustering(
        method=method,
        auto_k_method=auto_k_method,
        min_clusters=min_clusters,
        max_clusters=max_clusters
    )
    return config.fit(som, X, n_clusters=n_clusters)


def assign_cluster_labels(
    som,
    X: Union[pd.DataFrame, np.ndarray],
    neuron_labels: Dict[Tuple[int, int], int]
) -> np.ndarray:
    """
    Assign cluster labels to data samples based on their BMUs.

    Parameters
    ----------
    som : SEDBGSOM or DBGSOMWrapper
        Trained SOM model.
    X : DataFrame or ndarray
        Data samples.
    neuron_labels : dict
        Mapping of neuron position to cluster label.

    Returns
    -------
    sample_labels : ndarray of shape (n_samples,)
        Cluster label for each sample.

    Examples
    --------
    >>> neuron_labels = cluster_acute(som, X)
    >>> sample_labels = assign_cluster_labels(som, X, neuron_labels)
    >>> print(f"Cluster sizes: {np.bincount(sample_labels)[1:]}")
    """
    bmus = som.predict(X)
    return np.array([neuron_labels.get(bmu, 0) for bmu in bmus])


def get_cluster_info(
    som,
    X: Union[pd.DataFrame, np.ndarray],
    neuron_labels: Dict[Tuple[int, int], int]
) -> Dict[int, Dict]:
    """
    Get information about each cluster.

    Parameters
    ----------
    som : SEDBGSOM or DBGSOMWrapper
        Trained SOM model.
    X : DataFrame or ndarray
        Data samples.
    neuron_labels : dict
        Mapping of neuron position to cluster label.

    Returns
    -------
    info : dict
        For each cluster label, contains:
        - 'n_neurons': Number of neurons in cluster
        - 'n_samples': Number of samples mapped to cluster
        - 'positions': List of neuron positions in cluster
    """
    sample_labels = assign_cluster_labels(som, X, neuron_labels)

    info = {}
    for label in sorted(set(neuron_labels.values())):
        positions = [pos for pos, l in neuron_labels.items() if l == label]
        n_samples = np.sum(sample_labels == label)

        info[label] = {
            'n_neurons': len(positions),
            'n_samples': int(n_samples),
            'positions': positions
        }

    return info
