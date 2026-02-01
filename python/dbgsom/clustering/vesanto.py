"""
Vesanto-Alhoniemi: Two-Level Clustering of the Self-Organizing Map.

Based on the paper: "Clustering of the Self-Organizing Map"
by Juha Vesanto and Esa Alhoniemi (IEEE Trans. Neural Networks, 2000).

The two-level approach:
1. SOM training produces prototypes (neurons with weight vectors)
2. Cluster the prototypes using Ward's hierarchical clustering or K-means
3. Determine optimal k using Davies-Bouldin index (minimize)

Benefits:
- Noise reduction: prototypes are local averages, less sensitive to outliers
- Computational efficiency: cluster N neurons instead of M>>N samples
- Dendrogram visualization: interpretable cluster hierarchy
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


class SOMVesantoClustering:
    """
    Vesanto-Alhoniemi two-level clustering on SOM prototypes.

    Clusters trained SOM neurons using Ward's agglomerative hierarchical
    clustering or K-means. Based on the seminal paper by Vesanto & Alhoniemi
    (IEEE Trans. Neural Networks, 2000).

    Best for:
    - Convex, well-separated clusters
    - When dendrogram visualization is desired
    - When Davies-Bouldin index is preferred over silhouette

    Parameters
    ----------
    method : str, default='ward'
        Clustering method:
        - 'ward': Agglomerative hierarchical clustering (recommended)
        - 'kmeans': K-means partitive clustering
    auto_k_method : str, default='davies_bouldin'
        Method for automatic k selection:
        - 'davies_bouldin': Minimize Davies-Bouldin index (recommended)
        - 'silhouette': Maximize silhouette score
    min_clusters : int, default=2
        Minimum number of clusters to consider for auto k detection.
    max_clusters : int, default=15
        Maximum number of clusters to consider for auto k detection.
    random_state : int, optional
        Random seed for K-means (only used if method='kmeans').

    Attributes
    ----------
    labels_ : Dict[Tuple[int, int], int]
        After fit, maps neuron position (x, y) to cluster label (1-indexed).
    n_clusters_ : int
        Number of clusters found.
    optimal_k_ : int
        Automatically determined optimal k (if n_clusters not specified).
    linkage_matrix_ : ndarray or None
        Hierarchical clustering linkage matrix (for 'ward' method only).
        Can be used with scipy.cluster.hierarchy.dendrogram() for visualization.
    db_scores_ : Dict[int, float]
        Davies-Bouldin index for each k evaluated (lower is better).
    silhouette_scores_ : Dict[int, float]
        Silhouette scores for each k evaluated (higher is better).

    Examples
    --------
    >>> from dbgsom import SEDBGSOM
    >>> from dbgsom.clustering import SOMVesantoClustering
    >>> som = SEDBGSOM(bayesian=True).fit(X)
    >>>
    >>> # Auto-detect optimal k
    >>> clustering = SOMVesantoClustering()
    >>> labels = clustering.fit(som, X)
    >>>
    >>> # Visualize dendrogram
    >>> from scipy.cluster.hierarchy import dendrogram
    >>> dendrogram(clustering.linkage_matrix_)
    """

    def __init__(
        self,
        method: str = 'ward',
        auto_k_method: str = 'davies_bouldin',
        min_clusters: int = 2,
        max_clusters: int = 15,
        random_state: Optional[int] = None
    ):
        if method not in ('ward', 'kmeans'):
            raise ValueError(f"method must be 'ward' or 'kmeans', got '{method}'")
        if auto_k_method not in ('davies_bouldin', 'silhouette'):
            raise ValueError(
                f"auto_k_method must be 'davies_bouldin' or 'silhouette', "
                f"got '{auto_k_method}'"
            )
        if min_clusters < 2:
            raise ValueError(f"min_clusters must be >= 2, got {min_clusters}")
        if max_clusters < min_clusters:
            raise ValueError(
                f"max_clusters ({max_clusters}) must be >= min_clusters ({min_clusters})"
            )

        self.method = method
        self.auto_k_method = auto_k_method
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state

        self.labels_: Optional[Dict[Tuple[int, int], int]] = None
        self.n_clusters_: Optional[int] = None
        self.optimal_k_: Optional[int] = None
        self.linkage_matrix_: Optional[np.ndarray] = None
        self.db_scores_: Dict[int, float] = {}
        self.silhouette_scores_: Dict[int, float] = {}

    def fit(
        self,
        som,
        X=None,
        n_clusters: Optional[int] = None
    ) -> Dict[Tuple[int, int], int]:
        """
        Cluster SOM neurons using Vesanto-Alhoniemi method.

        Parameters
        ----------
        som : SEDBGSOM or DBGSOMWrapper
            Trained SOM model.
        X : array-like, optional
            Input data (not used, included for API consistency with other
            clustering methods).
        n_clusters : int, optional
            Target number of clusters. If None, automatically detect optimal k
            using the configured auto_k_method.

        Returns
        -------
        Dict[Tuple[int, int], int]
            Mapping from neuron position (x, y) to cluster label (1-indexed).
        """
        positions, weights = som.get_neuron_weights()
        n_neurons = len(positions)

        # Handle edge cases: empty or single neuron
        if not positions or n_neurons < 2:
            self.labels_ = {pos: 1 for pos in positions} if positions else {}
            self.n_clusters_ = 1 if positions else 0
            self.optimal_k_ = 1 if positions else 0
            return self.labels_

        if n_clusters is not None and n_clusters > n_neurons:
            n_clusters = n_neurons

        self.db_scores_ = {}
        self.silhouette_scores_ = {}

        if self.method == 'ward':
            self.labels_ = self._fit_ward(positions, weights, n_clusters)
        else:
            self.labels_ = self._fit_kmeans(positions, weights, n_clusters)

        self.n_clusters_ = len(set(self.labels_.values()))
        return self.labels_

    def _fit_ward(
        self,
        positions: List[Tuple[int, int]],
        weights: np.ndarray,
        n_clusters: Optional[int]
    ) -> Dict[Tuple[int, int], int]:
        """Fit using Ward's agglomerative hierarchical clustering."""
        distances = pdist(weights, metric='euclidean')
        self.linkage_matrix_ = linkage(distances, method='ward')

        if n_clusters is None:
            n_clusters = self._find_optimal_k(weights, self.linkage_matrix_)

        self.optimal_k_ = n_clusters
        cluster_ids = fcluster(self.linkage_matrix_, n_clusters, criterion='maxclust')

        return {pos: int(cluster_ids[i]) for i, pos in enumerate(positions)}

    def _fit_kmeans(
        self,
        positions: List[Tuple[int, int]],
        weights: np.ndarray,
        n_clusters: Optional[int]
    ) -> Dict[Tuple[int, int], int]:
        """Fit using K-means partitive clustering."""
        self.linkage_matrix_ = None

        if n_clusters is None:
            n_clusters = self._find_optimal_k_kmeans(weights)

        self.optimal_k_ = n_clusters

        km = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        cluster_ids = km.fit_predict(weights) + 1

        return {pos: int(cluster_ids[i]) for i, pos in enumerate(positions)}

    def _find_optimal_k(
        self,
        weights: np.ndarray,
        linkage_matrix: np.ndarray
    ) -> int:
        """
        Find optimal k using Davies-Bouldin or silhouette score.

        Uses the precomputed linkage matrix to cut at different k values.
        """
        n_neurons = len(weights)
        max_k = min(self.max_clusters, n_neurons - 1)

        for k in range(self.min_clusters, max_k + 1):
            cluster_ids = fcluster(linkage_matrix, k, criterion='maxclust')

            if len(set(cluster_ids)) < 2:
                continue

            try:
                self.db_scores_[k] = davies_bouldin_score(weights, cluster_ids)
                self.silhouette_scores_[k] = silhouette_score(weights, cluster_ids)
            except (ValueError, ZeroDivisionError):
                # These metrics can fail with degenerate clusters
                continue

        return self._select_optimal_k()

    def _find_optimal_k_kmeans(self, weights: np.ndarray) -> int:
        """Find optimal k for K-means by trying different values."""
        n_neurons = len(weights)
        max_k = min(self.max_clusters, n_neurons - 1)

        for k in range(self.min_clusters, max_k + 1):
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )
            cluster_ids = km.fit_predict(weights)

            if len(set(cluster_ids)) < 2:
                continue

            try:
                self.db_scores_[k] = davies_bouldin_score(weights, cluster_ids)
                self.silhouette_scores_[k] = silhouette_score(weights, cluster_ids)
            except (ValueError, ZeroDivisionError):
                # These metrics can fail with degenerate clusters
                continue

        return self._select_optimal_k()

    def _select_optimal_k(self) -> int:
        """Select optimal k based on configured method."""
        if not self.db_scores_:
            # No valid k values evaluated; fall back to minimum
            return self.min_clusters

        if self.auto_k_method == 'davies_bouldin':
            return min(self.db_scores_.keys(), key=self.db_scores_.get)
        else:
            return max(self.silhouette_scores_.keys(), key=self.silhouette_scores_.get)

    def get_dendrogram_data(self) -> Optional[np.ndarray]:
        """
        Get linkage matrix for dendrogram visualization.

        Returns
        -------
        ndarray or None
            Linkage matrix compatible with scipy.cluster.hierarchy.dendrogram().
            Returns None if method is 'kmeans'.

        Examples
        --------
        >>> from scipy.cluster.hierarchy import dendrogram
        >>> import matplotlib.pyplot as plt
        >>>
        >>> clustering = SOMVesantoClustering(method='ward')
        >>> clustering.fit(som, X)
        >>>
        >>> linkage = clustering.get_dendrogram_data()
        >>> if linkage is not None:
        ...     dendrogram(linkage)
        ...     plt.show()
        """
        return self.linkage_matrix_

    def get_k_evaluation_profile(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Get evaluation scores for different k values.

        Useful for visualizing how clustering quality varies with k.

        Returns
        -------
        db_scores : Dict[int, float]
            Davies-Bouldin index for each k (lower is better).
        silhouette_scores : Dict[int, float]
            Silhouette scores for each k (higher is better).

        Examples
        --------
        >>> clustering = SOMVesantoClustering()
        >>> clustering.fit(som, X)
        >>>
        >>> db_scores, sil_scores = clustering.get_k_evaluation_profile()
        >>> print(f"Best k by DB: {min(db_scores, key=db_scores.get)}")
        >>> print(f"Best k by silhouette: {max(sil_scores, key=sil_scores.get)}")
        """
        return self.db_scores_.copy(), self.silhouette_scores_.copy()
