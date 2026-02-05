"""
SE-DBGSOM - Statistical Error Dynamic Batch Growing Self-Organizing Map.

A two-part workflow:
1. Training: Use SEDBGSOM (always StatisticalError method)
2. Clustering: Use cluster_vesanto() separately

Example
-------
>>> from dbgsom import SEDBGSOM, cluster_vesanto
>>>
>>> # Part 1: Train the model
>>> som = SEDBGSOM(lambda_=1.5, max_neurons=100, n_iter=200)
>>> som.fit(X)
>>>
>>> # Part 2: Cluster with Vesanto (with dendrogram)
>>> labels = cluster_vesanto(som, X)  # auto k via Davies-Bouldin
>>>
>>> # Get sample labels
>>> from dbgsom import assign_cluster_labels
>>> sample_labels = assign_cluster_labels(som, X, labels)

Legacy API (DBGSOMWrapper) is still available for backwards compatibility.
"""

# New recommended API
from .sedbgsom import SEDBGSOM
from .clustering import (
    cluster_vesanto,
    assign_cluster_labels,
    get_cluster_info,
    compute_all_metrics,
    silhouette_score_som,
    nmi_score,
    ari_score,
    cluster_purity,
    count_clusters,
)

# Legacy API (still supported)
from .wrapper import DBGSOMWrapper
from .clustering import (
    SOMSpectralClustering,
    SOMVesantoClustering,
)

# Shared utilities
from .preprocessor import DataPreprocessor, ColumnType, NormalizationMethod
from .julia_setup import setup_julia_environment, is_julia_ready, get_julia_version
from .visualization import DBGSOMVisualizer, PanelDataVisualizer

__version__ = "0.2.0"

__all__ = [
    # Recommended API - Training
    "SEDBGSOM",
    # Recommended API - Clustering
    "cluster_vesanto",
    "assign_cluster_labels",
    "get_cluster_info",
    # Legacy API
    "DBGSOMWrapper",
    "SOMSpectralClustering",
    "SOMVesantoClustering",
    # Visualization
    "DBGSOMVisualizer",
    "PanelDataVisualizer",
    # Preprocessing
    "DataPreprocessor",
    "ColumnType",
    "NormalizationMethod",
    # Clustering metrics
    "compute_all_metrics",
    "silhouette_score_som",
    "nmi_score",
    "ari_score",
    "cluster_purity",
    "count_clusters",
    # Setup functions
    "setup_julia_environment",
    "is_julia_ready",
    "get_julia_version",
]
