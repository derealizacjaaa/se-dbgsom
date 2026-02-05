"""
Evaluation metrics for SOM neuron clustering.

Provides metrics to compare clustering quality against ground truth labels.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Protocol, Union
from collections import defaultdict


class SOMProtocol(Protocol):
    """Protocol defining the SOM interface used by metrics."""

    def get_neuron_weights(self) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Return neuron positions and weight vectors."""
        ...

    def predict(self, X: Any) -> List[Tuple[int, int]]:
        """Return BMU positions for input data."""
        ...


def silhouette_score_som(
    som: SOMProtocol,
    clusters: Dict[Tuple[int, int], int]
) -> float:
    """
    Compute silhouette score for SOM neuron clustering.

    Uses neuron weight vectors as data points.

    Parameters
    ----------
    som : DBGSOMWrapper
        Fitted SOM model.
    clusters : dict
        Mapping from neuron position (x, y) to cluster label.

    Returns
    -------
    float
        Silhouette score in range [-1, 1]. Higher is better.
    """
    from sklearn.metrics import silhouette_score

    positions, weights = som.get_neuron_weights()
    labels = [clusters[pos] for pos in positions]

    # Need at least 2 clusters for silhouette
    if len(set(labels)) < 2:
        return 0.0

    return silhouette_score(weights, labels)


def nmi_score(
    bmus: List[Tuple[int, int]],
    clusters: Dict[Tuple[int, int], int],
    true_labels: np.ndarray
) -> float:
    """
    Compute Normalized Mutual Information between clustering and ground truth.

    Maps samples to their neuron's cluster, then compares to true labels.

    Parameters
    ----------
    bmus : list of tuple
        Best Matching Unit positions for each sample.
    clusters : dict
        Mapping from neuron position to cluster label.
    true_labels : array-like
        Ground truth class labels for each sample.

    Returns
    -------
    float
        NMI score in range [0, 1]. Higher is better.
    """
    from sklearn.metrics import normalized_mutual_info_score

    pred_labels = [clusters[tuple(bmu)] for bmu in bmus]

    return normalized_mutual_info_score(true_labels, pred_labels)


def ari_score(
    bmus: List[Tuple[int, int]],
    clusters: Dict[Tuple[int, int], int],
    true_labels: np.ndarray
) -> float:
    """
    Compute Adjusted Rand Index between clustering and ground truth.

    Parameters
    ----------
    bmus : list of tuple
        Best Matching Unit positions for each sample.
    clusters : dict
        Mapping from neuron position to cluster label.
    true_labels : array-like
        Ground truth class labels for each sample.

    Returns
    -------
    float
        ARI score in range [-1, 1]. Higher is better.
    """
    from sklearn.metrics import adjusted_rand_score

    pred_labels = [clusters[tuple(bmu)] for bmu in bmus]

    return adjusted_rand_score(true_labels, pred_labels)


def cluster_purity(
    bmus: List[Tuple[int, int]],
    clusters: Dict[Tuple[int, int], int],
    true_labels: np.ndarray
) -> float:
    """
    Compute cluster purity (fraction of samples matching dominant class).

    Parameters
    ----------
    bmus : list of tuple
        Best Matching Unit positions for each sample.
    clusters : dict
        Mapping from neuron position to cluster label.
    true_labels : array-like
        Ground truth class labels for each sample.

    Returns
    -------
    float
        Purity score in range [0, 1]. Higher is better.
    """
    cluster_samples = defaultdict(list)

    for i, bmu in enumerate(bmus):
        c = clusters[tuple(bmu)]
        cluster_samples[c].append(true_labels[i])

    total_correct = 0
    for c, labels in cluster_samples.items():
        if labels:
            counts = defaultdict(int)
            for l in labels:
                counts[l] += 1
            total_correct += max(counts.values())

    return total_correct / len(true_labels)


def count_clusters(clusters: Dict[Tuple[int, int], int]) -> int:
    """
    Count number of unique clusters.

    Parameters
    ----------
    clusters : dict
        Mapping from neuron position to cluster label.

    Returns
    -------
    int
        Number of unique cluster labels.
    """
    return len(set(clusters.values()))


def compute_all_metrics(
    som: SOMProtocol,
    clusters: Dict[Tuple[int, int], int],
    X: Any,
    true_labels: np.ndarray,
    bmus: List[Tuple[int, int]] = None
) -> Dict[str, Union[float, int]]:
    """
    Compute all evaluation metrics for a clustering.

    Parameters
    ----------
    som : DBGSOMWrapper
        Fitted SOM model.
    clusters : dict
        Mapping from neuron position to cluster label.
    X : array-like
        Input data.
    true_labels : array-like
        Ground truth class labels.
    bmus : list of tuple, optional
        Pre-computed BMU positions. If None, computed from X.

    Returns
    -------
    dict
        Dictionary with keys: 'silhouette', 'nmi', 'ari', 'purity', 'n_clusters'
    """
    if bmus is None:
        bmus = som.predict(X)

    return {
        'silhouette': silhouette_score_som(som, clusters),
        'nmi': nmi_score(bmus, clusters, true_labels),
        'ari': ari_score(bmus, clusters, true_labels),
        'purity': cluster_purity(bmus, clusters, true_labels),
        'n_clusters': count_clusters(clusters)
    }
