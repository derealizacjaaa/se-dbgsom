"""
ACUTE: Agglomerative Clustering Using Topological Relations.

Adapted for SOM neuron clustering. Uses topological predicates (Disjoint, Meet, Overlap)
to merge spatially connected neurons based on growing radii in weight space.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.spatial.distance import pdist, squareform

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class SOMAcuteClustering:
    """
    ACUTE clustering on SOM neuron weight vectors.

    Uses topological relations to merge neurons as the interaction radius grows.
    Tracks cluster evolution to find stable clusterings.

    Parameters
    ----------
    threshold_method : str, default='auto'
        'auto': Uses sorted critical radii (event-driven).
        'step': Uses fixed step_size increments.
    step_size : float, default=0.1
        Radius increment for 'step' method.
    max_radius : float, optional
        Maximum radius to consider. If None, runs until single cluster.
    outlier_threshold : int, default=1
        Minimum component size to be a cluster. Smaller = noise (-1 label).
    max_steps : int, default=1000
        Maximum radius steps for 'auto' method (for performance).
    auto_method : str, default='silhouette'
        Method for automatic k selection: 'silhouette', 'stability', 'elbow', 'combined'.
    min_clusters : int, default=2
        Minimum number of clusters to consider in auto mode.
    max_clusters : int, default=15
        Maximum number of clusters to consider in auto mode.

    Attributes
    ----------
    labels_ : dict
        After fit, maps neuron positions to cluster labels.
    radius_history_ : list
        Radii at each step.
    cluster_count_history_ : list
        Number of clusters at each step.
    final_radius_ : float
        Radius at which clustering was determined.
    silhouette_history_ : list
        Silhouette scores at each clustering state (if computed).
    optimal_k_ : int
        Automatically determined optimal number of clusters.
    """

    def __init__(
        self,
        threshold_method: str = 'auto',
        step_size: float = 0.1,
        max_radius: Optional[float] = None,
        outlier_threshold: int = 1,
        max_steps: int = 1000,
        auto_method: str = 'combined',
        min_clusters: int = 2,
        max_clusters: int = 15
    ):
        if not HAS_NETWORKX:
            raise ImportError(
                "ACUTE clustering requires networkx. "
                "Install with: pip install networkx"
            )

        self.threshold_method = threshold_method
        self.step_size = step_size
        self.max_radius = max_radius
        self.outlier_threshold = outlier_threshold
        self.max_steps = max_steps
        self.auto_method = auto_method
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

        self.labels_: Optional[Dict[Tuple[int, int], int]] = None
        self.radius_history_: List[float] = []
        self.cluster_count_history_: List[int] = []
        self.silhouette_history_: List[Tuple[int, float, float]] = []  # (k, radius, score)
        self.final_radius_: Optional[float] = None
        self.optimal_k_: Optional[int] = None

    def fit(
        self,
        som,
        X=None,
        n_clusters: Optional[int] = None
    ) -> Dict[Tuple[int, int], int]:
        """
        Cluster SOM neurons using ACUTE algorithm.

        Parameters
        ----------
        som : DBGSOMWrapper
            Trained SOM model.
        X : array-like, optional
            Input data (not used, included for API consistency).
        n_clusters : int, optional
            Target number of clusters. If provided, stops when reached.
            If None, uses automatic selection based on auto_method.

        Returns
        -------
        dict
            Mapping from neuron position (x, y) to cluster label (1-indexed).
        """
        positions, weights = som.get_neuron_weights()
        n_neurons = len(positions)

        if n_neurons < 2:
            self.labels_ = {pos: 1 for pos in positions}
            return self.labels_

        # Compute distance matrix
        dists = pdist(weights, metric='euclidean')
        distance_matrix = squareform(dists)

        # Build radius schedule
        radii_schedule = self._build_radius_schedule(dists)

        if n_clusters is not None:
            # Target specific k
            return self._fit_with_target_k(
                positions, weights, distance_matrix, radii_schedule, n_clusters
            )
        else:
            # Automatic k selection
            return self._fit_auto(
                positions, weights, distance_matrix, radii_schedule
            )

    def _fit_with_target_k(
        self,
        positions: List[Tuple[int, int]],
        weights: np.ndarray,
        distance_matrix: np.ndarray,
        radii_schedule: np.ndarray,
        n_clusters: int
    ) -> Dict[Tuple[int, int], int]:
        """Fit with a specific target number of clusters."""
        n_neurons = len(positions)

        G = nx.Graph()
        G.add_nodes_from(range(n_neurons))

        self.radius_history_ = []
        self.cluster_count_history_ = []

        best_radius = 0.0
        best_labels = None
        best_diff = float('inf')

        for r in radii_schedule:
            interaction_dist = 2 * r
            epsilon = 1e-9

            connected_mask = (
                (np.triu(distance_matrix, k=1) <= interaction_dist + epsilon) &
                (np.triu(distance_matrix, k=1) > 0)
            )

            rows, cols = np.where(connected_mask)
            if len(rows) > 0:
                G.add_edges_from(zip(rows, cols))

            n_components = nx.number_connected_components(G)
            self.radius_history_.append(r)
            self.cluster_count_history_.append(n_components)

            diff = abs(n_components - n_clusters)
            if diff < best_diff:
                best_diff = diff
                best_radius = r
                best_labels = self._extract_labels(G, positions, n_neurons)

            if n_components == n_clusters:
                break
            elif n_components < n_clusters:
                break

            if n_components == 1:
                break

        self.final_radius_ = best_radius
        self.optimal_k_ = n_clusters
        self.labels_ = best_labels if best_labels else self._extract_labels(G, positions, n_neurons)
        return self.labels_

    def _fit_auto(
        self,
        positions: List[Tuple[int, int]],
        weights: np.ndarray,
        distance_matrix: np.ndarray,
        radii_schedule: np.ndarray
    ) -> Dict[Tuple[int, int], int]:
        """Fit with automatic k selection."""
        from sklearn.metrics import silhouette_score

        n_neurons = len(positions)

        G = nx.Graph()
        G.add_nodes_from(range(n_neurons))

        self.radius_history_ = []
        self.cluster_count_history_ = []
        self.silhouette_history_ = []

        # Store clustering states at different k values
        clustering_states = {}  # k -> (radius, labels, silhouette)
        prev_k = n_neurons

        for r in radii_schedule:
            interaction_dist = 2 * r
            epsilon = 1e-9

            connected_mask = (
                (np.triu(distance_matrix, k=1) <= interaction_dist + epsilon) &
                (np.triu(distance_matrix, k=1) > 0)
            )

            rows, cols = np.where(connected_mask)
            if len(rows) > 0:
                G.add_edges_from(zip(rows, cols))

            n_components = nx.number_connected_components(G)
            self.radius_history_.append(r)
            self.cluster_count_history_.append(n_components)

            # When k changes and is in our target range, evaluate
            if n_components != prev_k and self.min_clusters <= n_components <= self.max_clusters:
                labels_dict = self._extract_labels(G, positions, n_neurons)
                labels_array = np.array([labels_dict[pos] for pos in positions])

                # Compute silhouette if we have at least 2 clusters with samples
                unique_labels = set(labels_array)
                if len(unique_labels) >= 2:
                    try:
                        sil_score = silhouette_score(weights, labels_array)
                    except (ValueError, ZeroDivisionError):
                        sil_score = -1.0
                else:
                    sil_score = -1.0

                self.silhouette_history_.append((n_components, r, sil_score))
                clustering_states[n_components] = (r, labels_dict.copy(), sil_score)

            prev_k = n_components

            if n_components == 1:
                break

        # Select optimal k based on method
        if not clustering_states:
            # Fallback: return current state
            self.labels_ = self._extract_labels(G, positions, n_neurons)
            self.optimal_k_ = 1
            self.final_radius_ = radii_schedule[-1] if len(radii_schedule) > 0 else 0
            return self.labels_

        optimal_k = self._select_optimal_k(clustering_states, weights, positions)
        self.optimal_k_ = optimal_k

        if optimal_k in clustering_states:
            self.final_radius_, self.labels_, _ = clustering_states[optimal_k]
        else:
            # Fallback to best silhouette
            best_k = max(clustering_states.keys(), key=lambda k: clustering_states[k][2])
            self.final_radius_, self.labels_, _ = clustering_states[best_k]
            self.optimal_k_ = best_k

        return self.labels_

    def _select_optimal_k(
        self,
        clustering_states: Dict[int, Tuple[float, Dict, float]],
        weights: np.ndarray,
        positions: List[Tuple[int, int]]
    ) -> int:
        """Select optimal k using the configured method."""
        if self.auto_method == 'silhouette':
            return self._select_by_silhouette(clustering_states)
        elif self.auto_method == 'stability':
            return self._select_by_stability(clustering_states)
        elif self.auto_method == 'elbow':
            return self._select_by_elbow(clustering_states)
        else:  # combined
            return self._select_combined(clustering_states)

    def _select_by_silhouette(self, clustering_states: Dict) -> int:
        """Select k with highest silhouette score."""
        valid_states = {k: v for k, v in clustering_states.items() if v[2] > -1}
        if not valid_states:
            return min(clustering_states.keys())
        return max(valid_states.keys(), key=lambda k: valid_states[k][2])

    def _select_by_stability(self, clustering_states: Dict) -> int:
        """Select k based on stability (longest plateau)."""
        if not self.cluster_count_history_:
            return min(clustering_states.keys())

        # Count persistence of each k
        k_persistence = {}
        prev_k = None
        count = 0

        for k in self.cluster_count_history_:
            if k == prev_k:
                count += 1
            else:
                if prev_k is not None and prev_k in clustering_states:
                    k_persistence[prev_k] = k_persistence.get(prev_k, 0) + count
                prev_k = k
                count = 1

        if prev_k is not None and prev_k in clustering_states:
            k_persistence[prev_k] = k_persistence.get(prev_k, 0) + count

        if not k_persistence:
            return min(clustering_states.keys())

        return max(k_persistence.keys(), key=k_persistence.get)

    def _select_by_elbow(self, clustering_states: Dict) -> int:
        """Select k using elbow method on cluster count curve."""
        if len(self.radius_history_) < 3:
            return min(clustering_states.keys())

        # Find the "elbow" - point of maximum curvature
        radii = np.array(self.radius_history_)
        counts = np.array(self.cluster_count_history_)

        # Normalize
        radii_norm = (radii - radii.min()) / (radii.max() - radii.min() + 1e-10)
        counts_norm = (counts - counts.min()) / (counts.max() - counts.min() + 1e-10)

        # Find elbow using perpendicular distance to line from start to end
        start = np.array([radii_norm[0], counts_norm[0]])
        end = np.array([radii_norm[-1], counts_norm[-1]])
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            return min(clustering_states.keys())

        line_unit = line_vec / line_len

        max_dist = 0
        elbow_idx = 0

        for i in range(len(radii)):
            point = np.array([radii_norm[i], counts_norm[i]])
            point_vec = point - start
            proj_len = np.dot(point_vec, line_unit)
            proj = start + proj_len * line_unit
            dist = np.linalg.norm(point - proj)

            if dist > max_dist:
                max_dist = dist
                elbow_idx = i

        elbow_k = counts[elbow_idx]

        # Find closest k in our states
        if elbow_k in clustering_states:
            return elbow_k

        available_ks = sorted(clustering_states.keys())
        closest_k = min(available_ks, key=lambda k: abs(k - elbow_k))
        return closest_k

    def _select_combined(self, clustering_states: Dict) -> int:
        """Combined method: weight silhouette, stability, and elbow."""
        scores = {}

        # Get candidates
        candidates = list(clustering_states.keys())
        if not candidates:
            return self.min_clusters

        # Silhouette scores (normalized)
        sil_scores = {k: clustering_states[k][2] for k in candidates}
        sil_min = min(sil_scores.values())
        sil_max = max(sil_scores.values())
        sil_range = sil_max - sil_min if sil_max > sil_min else 1.0

        # Stability scores
        k_persistence = {}
        if self.cluster_count_history_:
            prev_k = None
            count = 0
            for k in self.cluster_count_history_:
                if k == prev_k:
                    count += 1
                else:
                    if prev_k is not None:
                        k_persistence[prev_k] = k_persistence.get(prev_k, 0) + count
                    prev_k = k
                    count = 1
            if prev_k is not None:
                k_persistence[prev_k] = k_persistence.get(prev_k, 0) + count

        stab_max = max(k_persistence.values()) if k_persistence else 1

        # Combine scores
        for k in candidates:
            # Silhouette (0-1, higher better)
            sil_norm = (sil_scores[k] - sil_min) / sil_range if sil_range > 0 else 0.5

            # Stability (0-1, higher better)
            stab_norm = k_persistence.get(k, 0) / stab_max if stab_max > 0 else 0

            # Prefer moderate k (penalty for very high or very low)
            k_penalty = 0
            if k < 2:
                k_penalty = 0.5
            elif k > 10:
                k_penalty = 0.1 * (k - 10)

            # Combined score (weights: silhouette=0.5, stability=0.3, k_penalty=0.2)
            scores[k] = 0.5 * sil_norm + 0.3 * stab_norm - 0.2 * k_penalty

        return max(scores.keys(), key=lambda k: scores[k])

    def _build_radius_schedule(self, dists: np.ndarray) -> np.ndarray:
        """Build radius schedule based on method."""
        if self.threshold_method == 'auto':
            unique_dists = np.unique(dists)
            unique_dists = unique_dists[unique_dists > 0]
            radii_schedule = unique_dists / 2.0

            if len(radii_schedule) > self.max_steps:
                radii_schedule = np.linspace(
                    np.min(radii_schedule),
                    np.max(radii_schedule),
                    self.max_steps
                )
        else:
            max_dist = np.max(dists) if len(dists) > 0 else 1.0
            radii_schedule = np.arange(0, max_dist, self.step_size)

        if self.max_radius is not None:
            radii_schedule = radii_schedule[radii_schedule <= self.max_radius]

        return np.sort(radii_schedule)

    def _extract_labels(
        self,
        G: 'nx.Graph',
        positions: List[Tuple[int, int]],
        n_neurons: int
    ) -> Dict[Tuple[int, int], int]:
        """Extract cluster labels from graph components."""
        labels = {}
        components = list(nx.connected_components(G))

        cluster_id = 1
        for comp_set in components:
            if len(comp_set) >= self.outlier_threshold:
                for node_idx in comp_set:
                    labels[positions[node_idx]] = cluster_id
                cluster_id += 1
            else:
                for node_idx in comp_set:
                    labels[positions[node_idx]] = 0

        if cluster_id > 1:
            labels = self._assign_outliers_to_nearest(labels, positions, G)

        return labels

    def _assign_outliers_to_nearest(
        self,
        labels: Dict[Tuple[int, int], int],
        positions: List[Tuple[int, int]],
        G: 'nx.Graph'
    ) -> Dict[Tuple[int, int], int]:
        """Assign outlier neurons (label 0) to nearest cluster."""
        cluster_positions = {pos: lbl for pos, lbl in labels.items() if lbl > 0}

        if not cluster_positions:
            return {pos: 1 for pos in labels.keys()}

        for pos, lbl in list(labels.items()):
            if lbl == 0:
                pos_idx = positions.index(pos)
                min_dist = float('inf')
                nearest_label = 1

                for other_pos, other_lbl in cluster_positions.items():
                    if other_lbl > 0:
                        other_idx = positions.index(other_pos)
                        try:
                            dist = nx.shortest_path_length(G, pos_idx, other_idx)
                        except nx.NetworkXNoPath:
                            dist = float('inf')

                        if dist < min_dist:
                            min_dist = dist
                            nearest_label = other_lbl

                labels[pos] = nearest_label

        return labels

    def get_stability_profile(self) -> Tuple[List[float], List[int]]:
        """
        Get the cluster count evolution over radius.

        Returns
        -------
        radii : list of float
            Radius values.
        counts : list of int
            Number of clusters at each radius.
        """
        return self.radius_history_, self.cluster_count_history_

    def get_silhouette_profile(self) -> List[Tuple[int, float, float]]:
        """
        Get silhouette scores at different k values.

        Returns
        -------
        list of tuple
            (n_clusters, radius, silhouette_score) for each evaluated state.
        """
        return self.silhouette_history_
