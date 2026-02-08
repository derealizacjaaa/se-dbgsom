"""
High-level clustering API for SE-DBGSOM.

Provides simple functions to cluster trained SOM neurons:
- cluster_vesanto(): Best for convex clusters with dendrogram visualization

Usage:
    from dbgsom import SEDBGSOM, cluster_vesanto

    som = SEDBGSOM(lambda_=1.5, max_neurons=100, n_iter=200)
    som.fit(X)

    # For convex clusters with dendrogram visualization
    labels = cluster_vesanto(som, X)  # auto k via Davies-Bouldin
"""

import numpy as np
from typing import TYPE_CHECKING, Dict, Tuple, Optional, Union

if TYPE_CHECKING:
    import pandas as pd

from .vesanto import SOMVesantoClustering


def cluster_vesanto(
    som,
    X: Optional[Union["pd.DataFrame", np.ndarray]] = None,
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
    X: Union["pd.DataFrame", np.ndarray],
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
    >>> neuron_labels = cluster_vesanto(som, X)
    >>> sample_labels = assign_cluster_labels(som, X, neuron_labels)
    >>> print(f"Cluster sizes: {np.bincount(sample_labels)[1:]}")
    """
    bmus = som.predict(X)
    return np.array([neuron_labels.get(bmu, 0) for bmu in bmus])


def get_cluster_info(
    som,
    X: Union["pd.DataFrame", np.ndarray],
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
