"""
Clustering methods for SOM neurons.

Vesanto-Alhoniemi clustering is the main supported method for convex clusters
with dendrogram visualization.

Example
-------
>>> from dbgsom import SEDBGSOM, cluster_vesanto
>>>
>>> # Train SE-DBGSOM
>>> som = SEDBGSOM(lambda_=1.5, max_neurons=100, n_iter=200)
>>> som.fit(X)
>>>
>>> # Vesanto for convex clusters with dendrogram
>>> labels = cluster_vesanto(som, X)
>>>
>>> # Evaluate
>>> from dbgsom.clustering import compute_all_metrics
>>> metrics = compute_all_metrics(som, labels, X, true_labels)
"""

from .vesanto import SOMVesantoClustering
from .metrics import (
    compute_all_metrics,
    silhouette_score_som,
    nmi_score,
    ari_score,
    cluster_purity,
    count_clusters
)

# High-level API functions
from .api import (
    cluster_vesanto,
    assign_cluster_labels,
    get_cluster_info
)

__all__ = [
    # High-level API (recommended)
    "cluster_vesanto",
    "assign_cluster_labels",
    "get_cluster_info",
    # Low-level classes
    "SOMVesantoClustering",
    # Metrics
    "compute_all_metrics",
    "silhouette_score_som",
    "nmi_score",
    "ari_score",
    "cluster_purity",
    "count_clusters",
]
