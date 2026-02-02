"""
Clustering methods for SOM neurons.

Three main clustering approaches:
- ACUTE: Best for non-convex shapes (moons, circles, complex boundaries)
- Leiden: Best for convex/blob-like clusters
- Vesanto: Best for convex clusters with dendrogram visualization

Example
-------
>>> from dbgsom import SEDBGSOM, cluster_acute, cluster_leiden, cluster_vesanto
>>>
>>> # Train SE-DBGSOM
>>> som = SEDBGSOM(lambda_=1.5, max_neurons=100, n_iter=200)
>>> som.fit(X)
>>>
>>> # ACUTE for complex shapes
>>> labels = cluster_acute(som, X)
>>>
>>> # Leiden for blob-like clusters
>>> labels = cluster_leiden(som, X, n_clusters=3)
>>>
>>> # Vesanto for convex clusters with dendrogram
>>> labels = cluster_vesanto(som, X)
>>>
>>> # Evaluate
>>> from dbgsom.clustering import compute_all_metrics
>>> metrics = compute_all_metrics(som, labels, X, true_labels)
"""

from .spectral import SOMSpectralClustering
from .acute import SOMAcuteClustering
from .vesanto import SOMVesantoClustering
from .metrics import (
    compute_all_metrics,
    silhouette_score_som,
    nmi_score,
    ari_score,
    cluster_purity,
    count_clusters
)

# Leiden is optional (requires igraph, leidenalg)
try:
    from .leiden import SOMLeidenClustering
    HAS_LEIDEN = True
except ImportError:
    SOMLeidenClustering = None
    HAS_LEIDEN = False

# High-level API functions
from .api import (
    cluster_acute,
    cluster_leiden,
    cluster_vesanto,
    assign_cluster_labels,
    get_cluster_info
)

__all__ = [
    # High-level API (recommended)
    "cluster_acute",
    "cluster_leiden",
    "cluster_vesanto",
    "assign_cluster_labels",
    "get_cluster_info",
    # Low-level classes
    "SOMSpectralClustering",
    "SOMAcuteClustering",
    "SOMLeidenClustering",
    "SOMVesantoClustering",
    # Metrics
    "compute_all_metrics",
    "silhouette_score_som",
    "nmi_score",
    "ari_score",
    "cluster_purity",
    "count_clusters",
    # Flags
    "HAS_LEIDEN",
]
