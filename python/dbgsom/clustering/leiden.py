"""
Leiden Clustering for SOM neurons.

Uses the Leiden community detection algorithm on a k-NN graph built from
neuron weight vectors. Optionally incorporates SOM topology.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

try:
    import igraph as ig
    import leidenalg as la
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False


class SOMLeidenClustering:
    """
    Leiden community detection on SOM neuron similarity graph.

    Builds a k-NN graph from neuron weights and applies the Leiden algorithm
    to find communities (clusters).

    Parameters
    ----------
    n_neighbors : int, default=10
        Number of neighbors for k-NN graph construction.
    resolution : float, default=1.0
        Leiden resolution parameter. Higher = more clusters.
    use_topology : bool, default=True
        Whether to incorporate SOM grid topology as edges.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    labels_ : dict
        After fit, maps neuron positions to cluster labels.
    modularity_ : float
        Modularity score of the partition.
    n_clusters_ : int
        Number of clusters found.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        resolution: float = 1.0,
        use_topology: bool = True,
        random_state: Optional[int] = None
    ):
        if not HAS_LEIDEN:
            raise ImportError(
                "Leiden clustering requires igraph and leidenalg. "
                "Install with: pip install igraph leidenalg"
            )

        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.use_topology = use_topology
        self.random_state = random_state

        self.labels_: Optional[Dict[Tuple[int, int], int]] = None
        self.modularity_: Optional[float] = None
        self.n_clusters_: Optional[int] = None

    def fit(
        self,
        som,
        X=None,
        n_clusters: Optional[int] = None
    ) -> Dict[Tuple[int, int], int]:
        """
        Cluster SOM neurons using Leiden algorithm.

        Parameters
        ----------
        som : DBGSOMWrapper
            Trained SOM model.
        X : array-like, optional
            Input data (not used, included for API consistency).
        n_clusters : int, optional
            Target number of clusters. If provided, searches for appropriate
            resolution parameter.

        Returns
        -------
        dict
            Mapping from neuron position (x, y) to cluster label (1-indexed).
        """
        positions, weights = som.get_neuron_weights()
        n_neurons = len(positions)
        pos_to_idx = {pos: i for i, pos in enumerate(positions)}

        if n_neurons < 2:
            self.labels_ = {pos: 1 for pos in positions}
            self.n_clusters_ = 1
            return self.labels_

        # If target n_clusters specified, find appropriate resolution
        if n_clusters is not None:
            best_res = self._find_resolution_for_k(
                positions, weights, pos_to_idx, n_neurons, n_clusters
            )
            resolution = best_res
        else:
            resolution = self.resolution

        # Build graph and run Leiden
        g = self._build_graph(positions, weights, pos_to_idx, n_neurons)
        partition = self._run_leiden(g, resolution)

        labels = partition.membership
        self.modularity_ = partition.modularity
        self.n_clusters_ = len(set(labels))

        # Return dict mapping position -> cluster (1-indexed)
        self.labels_ = {pos: int(labels[i]) + 1 for i, pos in enumerate(positions)}
        return self.labels_

    def _build_graph(
        self,
        positions: List[Tuple[int, int]],
        weights: np.ndarray,
        pos_to_idx: Dict[Tuple[int, int], int],
        n_neurons: int
    ) -> 'ig.Graph':
        """Build k-NN graph with optional topology edges."""
        from sklearn.neighbors import kneighbors_graph

        # Build k-NN graph from weight similarity
        k = min(self.n_neighbors, n_neurons - 1)
        knn_graph = kneighbors_graph(
            weights,
            n_neighbors=k,
            mode='distance',
            include_self=False
        )

        # Collect edges and weights
        edges = []
        edge_weights = []
        seen_edges = set()

        # Add k-NN edges
        coo = knn_graph.tocoo()
        for i, j, dist in zip(coo.row, coo.col, coo.data):
            edge_key = (min(i, j), max(i, j))
            if edge_key not in seen_edges:
                edges.append(edge_key)
                # Convert distance to similarity weight
                edge_weights.append(1.0 / (1.0 + dist))
                seen_edges.add(edge_key)

        # Optionally add SOM topology edges
        if self.use_topology:
            for i, pos in enumerate(positions):
                x, y = pos
                for neighbor in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                    if neighbor in pos_to_idx:
                        j = pos_to_idx[neighbor]
                        edge_key = (min(i, j), max(i, j))
                        if edge_key not in seen_edges:
                            edges.append(edge_key)
                            # Weight by inverse weight distance
                            dist = np.linalg.norm(weights[i] - weights[j])
                            edge_weights.append(1.0 / (1.0 + dist))
                            seen_edges.add(edge_key)

        # Create igraph graph
        g = ig.Graph(n=n_neurons, edges=edges)
        g.es['weight'] = edge_weights

        return g

    def _run_leiden(
        self,
        g: 'ig.Graph',
        resolution: float
    ) -> 'la.VertexPartition':
        """Run Leiden algorithm on graph."""
        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=resolution,
            seed=self.random_state
        )

        return partition

    def _find_resolution_for_k(
        self,
        positions: List[Tuple[int, int]],
        weights: np.ndarray,
        pos_to_idx: Dict[Tuple[int, int], int],
        n_neurons: int,
        target_k: int,
        tol: int = 1
    ) -> float:
        """
        Binary search for resolution parameter yielding ~k clusters.

        Parameters
        ----------
        target_k : int
            Target number of clusters.
        tol : int
            Tolerance for cluster count match.

        Returns
        -------
        float
            Resolution parameter.
        """
        g = self._build_graph(positions, weights, pos_to_idx, n_neurons)

        lo, hi = 0.01, 10.0
        best_res = 1.0
        best_diff = float('inf')

        for _ in range(20):  # Binary search iterations
            mid = (lo + hi) / 2
            partition = self._run_leiden(g, mid)
            n_found = len(set(partition.membership))

            diff = abs(n_found - target_k)
            if diff < best_diff:
                best_diff = diff
                best_res = mid

            if diff <= tol:
                return best_res

            if n_found < target_k:
                # Need more clusters, higher resolution
                lo = mid
            else:
                # Need fewer clusters, lower resolution
                hi = mid

        return best_res

    def find_resolution_for_k(
        self,
        som,
        X,
        target_k: int,
        tol: int = 1
    ) -> float:
        """
        Public method to find resolution for target cluster count.

        Parameters
        ----------
        som : DBGSOMWrapper
            Trained SOM model.
        X : array-like
            Input data (not used).
        target_k : int
            Target number of clusters.
        tol : int
            Tolerance for cluster count match.

        Returns
        -------
        float
            Resolution parameter yielding approximately target_k clusters.
        """
        positions, weights = som.get_neuron_weights()
        pos_to_idx = {pos: i for i, pos in enumerate(positions)}
        n_neurons = len(positions)

        return self._find_resolution_for_k(
            positions, weights, pos_to_idx, n_neurons, target_k, tol
        )
