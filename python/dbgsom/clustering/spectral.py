"""
Spectral Clustering for SOM neurons.

Applies spectral clustering to the neuron weight vectors of a trained SOM.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class SOMSpectralClustering:
    """
    Spectral clustering on SOM neuron weight vectors.

    Uses sklearn's SpectralClustering with configurable affinity and
    automatic gamma estimation.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters. If None, estimated from neuron count.
    affinity : str, default='rbf'
        Affinity type: 'rbf', 'nearest_neighbors', or 'precomputed'.
    gamma : float, optional
        RBF kernel parameter. If None, auto-estimated from data.
    n_neighbors : int, default=10
        Number of neighbors for 'nearest_neighbors' affinity.
    assign_labels : str, default='kmeans'
        Strategy for label assignment: 'kmeans' or 'discretize'.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    labels_ : dict
        After fit, maps neuron positions to cluster labels.
    n_clusters_ : int
        Actual number of clusters used.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        affinity: str = 'rbf',
        gamma: Optional[float] = None,
        n_neighbors: int = 10,
        assign_labels: str = 'kmeans',
        random_state: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.assign_labels = assign_labels
        self.random_state = random_state

        self.labels_: Optional[Dict[Tuple[int, int], int]] = None
        self.n_clusters_: Optional[int] = None

    def fit(
        self,
        som,
        X=None,
        n_clusters: Optional[int] = None
    ) -> Dict[Tuple[int, int], int]:
        """
        Cluster SOM neurons using spectral clustering.

        Parameters
        ----------
        som : DBGSOMWrapper
            Trained SOM model.
        X : array-like, optional
            Input data (not used, included for API consistency).
        n_clusters : int, optional
            Override number of clusters.

        Returns
        -------
        dict
            Mapping from neuron position (x, y) to cluster label (1-indexed).
        """
        from sklearn.cluster import SpectralClustering
        from scipy.spatial.distance import pdist

        positions, weights = som.get_neuron_weights()
        n_neurons = len(positions)

        # Determine gamma for RBF kernel
        gamma = self.gamma
        if gamma is None and self.affinity == 'rbf':
            # Use median pairwise distance heuristic
            dists = pdist(weights)
            if len(dists) > 0:
                median_dist = np.median(dists)
                gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0
            else:
                gamma = 1.0

        # Determine number of clusters
        k = n_clusters or self.n_clusters
        if k is None:
            # Estimate: sqrt(n/2) is a common heuristic
            k = max(2, int(np.sqrt(n_neurons / 2)))
        k = min(k, n_neurons - 1)  # Can't have more clusters than neurons
        self.n_clusters_ = k

        # Apply spectral clustering
        sc = SpectralClustering(
            n_clusters=k,
            affinity=self.affinity,
            gamma=gamma,
            n_neighbors=min(self.n_neighbors, n_neurons - 1),
            assign_labels=self.assign_labels,
            random_state=self.random_state or 42
        )
        labels = sc.fit_predict(weights)

        # Return dict mapping position -> cluster (1-indexed)
        self.labels_ = {pos: int(labels[i]) + 1 for i, pos in enumerate(positions)}
        return self.labels_
