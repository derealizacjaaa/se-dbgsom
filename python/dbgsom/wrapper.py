"""
Python wrapper for Julia DBGSOM implementation.

Provides a scikit-learn-like interface to the Julia BasicDBGSOM module.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from .julia_setup import get_julia, ensure_julia_ready
from .preprocessor import DataPreprocessor
from .utils import (
    numpy_to_julia_matrix,
    julia_to_numpy_matrix,
    julia_dict_to_python,
    julia_vector_of_tuples_to_python
)


class DBGSOMWrapper:
    """
    Python wrapper for Julia DBGSOM (Dynamic Batch Growing Self-Organizing Map).

    This wrapper provides a scikit-learn-like interface to the Julia BasicDBGSOM
    implementation, with automatic data preprocessing and type handling.

    Parameters
    ----------
    sf : float, default=0.5
        Spreading factor controlling growth aggressiveness.
        - sf ~ 0.1: Conservative growth (fewer neurons)
        - sf ~ 0.5: Balanced growth (default)
        - sf ~ 0.9: Aggressive growth (more neurons)
    max_neurons : int, default=100
        Maximum number of neurons allowed.
    n_iter : int, default=100
        Number of training iterations.
    init_size : tuple of int, default=(2, 2)
        Initial grid size (width, height).
    preprocess : bool, default=True
        Whether to automatically preprocess input data.
    preprocessor_kwargs : dict, optional
        Additional arguments passed to DataPreprocessor.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    model_ : Julia DBGSOM object
        The underlying Julia model (after fitting).
    preprocessor_ : DataPreprocessor
        Fitted preprocessor (if preprocess=True).
    n_features_in_ : int
        Number of features seen during fit.
    n_neurons_ : int
        Number of neurons after training.
    feature_names_in_ : list of str
        Feature names from training data.

    Examples
    --------
    >>> import pandas as pd
    >>> from dbgsom import DBGSOMWrapper
    >>>
    >>> df = pd.read_csv('countries.csv')
    >>> som = DBGSOMWrapper(sf=0.6, max_neurons=50, n_iter=100)
    >>> som.fit(df)
    >>>
    >>> # Check detected column types
    >>> print(som.preprocessor_.get_binary_columns())
    >>>
    >>> # Get clusters
    >>> clusters = som.cluster()
    """

    def __init__(
        self,
        sf: float = 0.5,
        max_neurons: int = 100,
        n_iter: int = 100,
        init_size: Tuple[int, int] = (2, 2),
        preprocess: bool = True,
        preprocessor_kwargs: Optional[Dict] = None,
        random_state: Optional[int] = None,
    ):
        self.sf = sf
        self.max_neurons = max_neurons
        self.n_iter = n_iter
        self.init_size = init_size
        self.preprocess = preprocess
        self.preprocessor_kwargs = preprocessor_kwargs or {}
        self.random_state = random_state

        # Fitted attributes
        self.model_ = None
        self.preprocessor_: Optional[DataPreprocessor] = None
        self.n_features_in_: Optional[int] = None
        self.n_neurons_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None
        self._is_fitted = False

    def _ensure_julia(self):
        """Ensure Julia environment is ready."""
        ensure_julia_ready()
        return get_julia()

    def _prepare_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        fit_preprocessor: bool = False
    ) -> np.ndarray:
        """
        Prepare data for Julia.

        Returns data in Julia format: (n_features, n_samples), Float64.
        """
        # Extract feature names
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = None

        if self.preprocess:
            if fit_preprocessor:
                self.preprocessor_ = DataPreprocessor(**self.preprocessor_kwargs)
                X_processed = self.preprocessor_.fit_transform(X, feature_names)
                self.feature_names_in_ = self.preprocessor_.state.feature_names
            else:
                if self.preprocessor_ is None:
                    raise ValueError("Preprocessor not fitted. Call fit() first.")
                X_processed = self.preprocessor_.transform(X)
        else:
            # No preprocessing, just convert to correct format
            if isinstance(X, pd.DataFrame):
                self.feature_names_in_ = list(X.columns) if fit_preprocessor else self.feature_names_in_
                X_processed = X.values.astype(np.float64).T
            else:
                X_processed = np.asarray(X, dtype=np.float64).T
                if fit_preprocessor:
                    self.feature_names_in_ = [f"feature_{i}" for i in range(X_processed.shape[0])]

        return X_processed

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, str],
        y: Optional[Any] = None
    ) -> 'DBGSOMWrapper':
        """
        Fit the DBGSOM model.

        Parameters
        ----------
        X : DataFrame, ndarray, or str
            Training data. Can be:
            - pandas DataFrame
            - numpy array of shape (n_samples, n_features)
            - path to CSV file
        y : ignored
            Not used, present for scikit-learn API compatibility.

        Returns
        -------
        self : DBGSOMWrapper
            Fitted model.
        """
        # Handle CSV file path
        if isinstance(X, str):
            X = pd.read_csv(X)

        return self._fit_direct(X)

    def _fit_direct(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> 'DBGSOMWrapper':
        """Fit the model directly with current parameters."""
        jl = self._ensure_julia()

        # Prepare data
        X_julia = self._prepare_data(X, fit_preprocessor=True)
        n_features, n_samples = X_julia.shape
        self.n_features_in_ = n_features

        # Set random seed if specified
        if self.random_state is not None:
            jl.seval(f'Random.seed!({self.random_state})')

        # Convert to Julia matrix
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        # Create Julia DBGSOM
        init_x, init_y = self.init_size
        self.model_ = jl.BasicDBGSOM.DBGSOM[jl.Float64](
            input_dim=n_features,
            sf=float(self.sf),
            max_neurons=int(self.max_neurons),
            n_iter=int(self.n_iter),
            init_size=(init_x, init_y)
        )

        # Train the model
        jl.BasicDBGSOM.fit_b(self.model_, X_jl)

        # Store fitted attributes
        self.n_neurons_ = int(jl.BasicDBGSOM.n_neurons(self.model_))
        self._is_fitted = True

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray, str]
    ) -> List[Tuple[int, int]]:
        """
        Predict BMU positions for samples.

        Parameters
        ----------
        X : DataFrame, ndarray, or str
            Data to predict.

        Returns
        -------
        bmus : list of tuple
            BMU grid positions (x, y) for each sample.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        if isinstance(X, str):
            X = pd.read_csv(X)

        X_julia = self._prepare_data(X)
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        bmus_jl = jl.BasicDBGSOM.predict(self.model_, X_jl)
        return julia_vector_of_tuples_to_python(bmus_jl)

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray, str]
    ) -> np.ndarray:
        """
        Transform samples to BMU coordinates.

        Parameters
        ----------
        X : DataFrame, ndarray, or str
            Data to transform.

        Returns
        -------
        coords : ndarray of shape (n_samples, 2)
            BMU coordinates (x, y) for each sample.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        if isinstance(X, str):
            X = pd.read_csv(X)

        X_julia = self._prepare_data(X)
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        coords_jl = jl.BasicDBGSOM.transform(self.model_, X_jl)
        return julia_to_numpy_matrix(coords_jl).T

    def fit_predict(
        self,
        X: Union[pd.DataFrame, np.ndarray, str],
        y: Optional[Any] = None
    ) -> List[Tuple[int, int]]:
        """Fit the model and return BMU predictions."""
        return self.fit(X, y).predict(X)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray, str],
        y: Optional[Any] = None
    ) -> np.ndarray:
        """Fit the model and return transformed coordinates."""
        return self.fit(X, y).transform(X)

    def predict_with_distances(
        self,
        X: Union[pd.DataFrame, np.ndarray, str]
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Predict BMU positions with distances.

        Returns
        -------
        bmus : list of tuple
            BMU positions.
        distances : ndarray
            Distances to BMUs.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        if isinstance(X, str):
            X = pd.read_csv(X)

        X_julia = self._prepare_data(X)
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        result = jl.BasicDBGSOM.predict_with_distances(self.model_, X_jl)
        bmus_jl = result[0]
        distances_jl = result[1]

        bmus = julia_vector_of_tuples_to_python(bmus_jl)
        distances = np.array([float(d) for d in distances_jl])

        return bmus, distances

    def cluster(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold_quantile: float = 0.6
    ) -> Dict[Tuple[int, int], int]:
        """
        Cluster neurons using U* matrix (U-matrix + density) watershed approach.

        The U* matrix combines topology (U-matrix) with data density (hit counts):
        U*(i) = U(i) * (1 - normalized_density(i))

        This gives better clustering by considering both neuron weight distances
        and how many data points map to each neuron.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input data used to compute density (hit counts).
        threshold_quantile : float, default=0.6
            Quantile for U* threshold (0.0-1.0). Higher values = fewer clusters.
            - 0.5 (median): More clusters, finer granularity
            - 0.6 (default): Balanced
            - 0.7-0.8: Fewer clusters, coarser grouping

        Returns
        -------
        clusters : dict
            Mapping of neuron position to cluster label.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        X_jl = self._prepare_data(X)

        clusters_jl = jl.BasicDBGSOM.cluster_neurons_umatrix(
            self.model_, X_jl,
            threshold_quantile=float(threshold_quantile)
        )

        return julia_dict_to_python(clusters_jl, value_type='int')

    def cluster_n(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n_clusters: int
    ) -> Dict[Tuple[int, int], int]:
        """
        Cluster neurons into exactly n_clusters using U* matrix + hierarchical merging.

        First uses U* matrix clustering, then hierarchically merges the closest
        clusters (by weight centroid distance) until the target number is reached.

        This is useful when you know the expected number of clusters (e.g., 3 for Iris).

        Parameters
        ----------
        X : DataFrame or ndarray
            Input data used to compute density (hit counts).
        n_clusters : int
            Target number of clusters.

        Returns
        -------
        clusters : dict
            Mapping of neuron position to cluster label (1 to n_clusters).
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        X_jl = self._prepare_data(X)

        clusters_jl = jl.BasicDBGSOM.cluster_neurons_n(
            self.model_, X_jl, int(n_clusters)
        )

        return julia_dict_to_python(clusters_jl, value_type='int')

    def get_umatrix(self) -> Dict[Tuple[int, int], float]:
        """
        Compute the U-matrix (unified distance matrix) using standard procedure.

        The U-matrix shows the average distance between each neuron and its neighbors.
        High values indicate cluster boundaries, low values indicate cluster interiors.

        Algorithm (Standard Procedure):
        For each node n:
        1. Retrieve neighbors N(n) from topology
        2. If N(n) is empty, set U_n = 0
        3. Else compute average Euclidean distance to all neighbors:
           U_n = sum(dist(w_n, w_m) for m in N(n)) / |N(n)|

        Returns
        -------
        umatrix : dict
            Mapping of neuron position to average neighbor distance value.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        umat_jl = jl.BasicDBGSOM.compute_umatrix(self.model_)
        return julia_dict_to_python(umat_jl, value_type='float')

    def get_expanded_umatrix(
        self, log: bool = False
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]]:
        """
        Compute the expanded U-matrix with per-edge distances.

        The expanded U-matrix provides both neuron-level (mean) and edge-level
        (pairwise) distance information for detailed hexagonal visualization.

        Parameters
        ----------
        log : bool, default=False
            If True, apply log transform to distances.

        Returns
        -------
        neuron_umat : dict
            Mapping of neuron position (x, y) to average neighbor distance.
        edge_umat : dict
            Mapping of edge ((x1, y1), (x2, y2)) to pairwise distance between
            adjacent neurons. Keys are sorted so (x1, y1) < (x2, y2).
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        neuron_jl, edge_jl = jl.BasicDBGSOM.compute_expanded_umatrix(self.model_, log=log)

        # Convert neuron dict
        neuron_umat = julia_dict_to_python(neuron_jl, value_type='float')

        # Convert edge dict (keys are tuples of tuples)
        edge_umat = {}
        for key, val in dict(edge_jl).items():
            # key is ((x1, y1), (x2, y2)) from Julia
            pos1 = (int(key[0][0]), int(key[0][1]))
            pos2 = (int(key[1][0]), int(key[1][1]))
            edge_umat[(pos1, pos2)] = float(val)

        return neuron_umat, edge_umat

    def get_component_plane(self, feature_idx: int) -> Dict[Tuple[int, int], float]:
        """
        Get component plane for a specific feature.

        Parameters
        ----------
        feature_idx : int
            Index of the feature (0-based).

        Returns
        -------
        plane : dict
            Mapping of position to weight value for that feature.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        # Julia uses 1-based indexing
        plane_jl = jl.BasicDBGSOM.compute_component_plane(self.model_, feature_idx + 1)
        return julia_dict_to_python(plane_jl, value_type='float')

    def get_hit_counts(
        self,
        X: Union[pd.DataFrame, np.ndarray, str]
    ) -> Dict[Tuple[int, int], int]:
        """
        Count samples assigned to each neuron.

        Parameters
        ----------
        X : DataFrame, ndarray, or str
            Data to count hits for.

        Returns
        -------
        hits : dict
            Mapping of neuron position to hit count.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        if isinstance(X, str):
            X = pd.read_csv(X)

        X_julia = self._prepare_data(X)
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        hits_jl = jl.BasicDBGSOM.compute_hit_counts(self.model_, X_jl)
        return julia_dict_to_python(hits_jl, value_type='int')

    def get_neuron_weights(self) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Get all neuron weights.

        Returns
        -------
        positions : list of tuple
            Neuron grid positions.
        weights : ndarray of shape (n_neurons, n_features)
            Weight vectors for each neuron.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        result = jl.BasicDBGSOM.get_positions_and_weights(self.model_)
        pos_jl = result[0]
        weights_jl = result[1]

        positions = julia_vector_of_tuples_to_python(pos_jl)
        weights = julia_to_numpy_matrix(weights_jl).T  # (n_neurons, n_features)

        return positions, weights

    def get_grid_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get bounding box of the grid.

        Returns
        -------
        bounds : tuple
            ((x_min, x_max), (y_min, y_max))
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        bounds_jl = jl.BasicDBGSOM.grid_bounds(self.model_)

        x_bounds = (int(bounds_jl[0][0]), int(bounds_jl[0][1]))
        y_bounds = (int(bounds_jl[1][0]), int(bounds_jl[1][1]))

        return (x_bounds, y_bounds)

    def quantization_error(
        self,
        X: Union[pd.DataFrame, np.ndarray, str]
    ) -> float:
        """
        Compute mean quantization error.

        Parameters
        ----------
        X : DataFrame, ndarray, or str
            Data to compute error for.

        Returns
        -------
        qe : float
            Mean quantization error.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        if isinstance(X, str):
            X = pd.read_csv(X)

        X_julia = self._prepare_data(X)
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        qe = jl.BasicDBGSOM.compute_quantization_error(self.model_, X_jl)
        return float(qe)

    def summary(self) -> str:
        """
        Get a text summary of the model.

        Returns
        -------
        summary : str
            Formatted summary string.
        """
        self._check_is_fitted()

        lines = [
            "DBGSOM Model Summary",
            "=" * 40,
            "",
            "Configuration:",
            f"  Spreading factor: {self.sf}",
            f"  Max neurons: {self.max_neurons}",
            f"  Iterations: {self.n_iter}",
            "",
            "Trained model:",
            f"  Input dimension: {self.n_features_in_}",
            f"  Neurons: {self.n_neurons_}",
        ]

        bounds = self.get_grid_bounds()
        lines.append(f"  Grid bounds: x=[{bounds[0][0]}, {bounds[0][1]}], y=[{bounds[1][0]}, {bounds[1][1]}]")

        if self.preprocessor_ is not None:
            lines.append("")
            lines.append("Preprocessing:")
            binary = self.preprocessor_.get_binary_columns()
            cont = self.preprocessor_.get_continuous_columns()
            lines.append(f"  Binary features: {len(binary)}")
            lines.append(f"  Continuous features: {len(cont)}")

        return "\n".join(lines)

    def _check_is_fitted(self):
        """Check if model is fitted."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted

    def find_dead_neurons(self, min_hits: int = 1) -> List[Tuple[int, int]]:
        """
        Find neurons with fewer than min_hits activations during training.

        Note: Dead neurons are automatically pruned at the end of fit(),
        so this typically returns an empty list after training.

        Parameters
        ----------
        min_hits : int, default=1
            Minimum activation count to be considered "alive".

        Returns
        -------
        dead : list of tuple
            Positions of dead neurons.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        dead_jl = jl.BasicDBGSOM.find_dead_neurons(self.model_, min_hits=min_hits)
        return julia_vector_of_tuples_to_python(dead_jl)

    def prune_dead_neurons(self, min_hits: int = 1, min_neurons: int = 4) -> int:
        """
        Remove neurons with fewer than min_hits activations.

        Note: Dead neurons are automatically pruned at the end of fit().
        This method is for manual pruning if needed after modifying the model.

        Parameters
        ----------
        min_hits : int, default=1
            Minimum activation count to keep a neuron.
        min_neurons : int, default=4
            Minimum neurons to keep in the map.

        Returns
        -------
        n_removed : int
            Number of neurons removed.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        prune_fn = jl.seval("BasicDBGSOM.prune_dead_neurons!")
        n_removed = prune_fn(
            self.model_,
            min_hits=min_hits,
            min_neurons=min_neurons
        )
        # Update neuron count
        self.n_neurons_ = int(jl.BasicDBGSOM.n_neurons(self.model_))
        return int(n_removed)

    def find_empty_neurons(
        self,
        X: Union[pd.DataFrame, np.ndarray, str]
    ) -> List[Tuple[int, int]]:
        """
        Find neurons with no samples mapped to them.

        Unlike dead neurons (which are pruned during training), empty neurons
        are active neurons that simply have no samples mapping to them from
        the given dataset. This is expected behavior for sparse data regions.

        Parameters
        ----------
        X : DataFrame, ndarray, or str
            Data to check for mapping.

        Returns
        -------
        empty : list of tuple
            Positions of neurons with no samples mapped.
        """
        self._check_is_fitted()

        if isinstance(X, str):
            X = pd.read_csv(X)

        hits = self.get_hit_counts(X)
        positions, _ = self.get_neuron_weights()

        return [pos for pos in positions if hits.get(pos, 0) == 0]

    def get_neuron_stats(
        self,
        X: Union[pd.DataFrame, np.ndarray, str]
    ) -> Dict[str, Any]:
        """
        Get comprehensive neuron statistics.

        Parameters
        ----------
        X : DataFrame, ndarray, or str
            Data to analyze.

        Returns
        -------
        stats : dict
            - 'total': Total number of neurons
            - 'active': Neurons with at least one sample mapped
            - 'empty': Neurons with no samples mapped
            - 'empty_positions': List of empty neuron positions
            - 'empty_fraction': Fraction of neurons that are empty
            - 'hit_distribution': Dict mapping hit count to number of neurons
        """
        self._check_is_fitted()

        if isinstance(X, str):
            X = pd.read_csv(X)

        hits = self.get_hit_counts(X)
        positions, _ = self.get_neuron_weights()

        empty_positions = [pos for pos in positions if hits.get(pos, 0) == 0]
        active_count = len(positions) - len(empty_positions)

        # Hit distribution
        from collections import Counter
        hit_values = [hits.get(pos, 0) for pos in positions]
        hit_dist = dict(Counter(hit_values))

        return {
            'total': len(positions),
            'active': active_count,
            'empty': len(empty_positions),
            'empty_positions': empty_positions,
            'empty_fraction': len(empty_positions) / len(positions),
            'hit_distribution': hit_dist
        }

    def topographic_error(
        self,
        X: Union[pd.DataFrame, np.ndarray, str]
    ) -> float:
        """
        Compute topographic error.

        The proportion of samples for which BMU1 and BMU2 are not adjacent.

        Parameters
        ----------
        X : DataFrame, ndarray, or str
            Data to compute error for.

        Returns
        -------
        error : float
            Topographic error (0 = perfect topology, 1 = worst).
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        if isinstance(X, str):
            X = pd.read_csv(X)

        X_julia = self._prepare_data(X)
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        error = jl.BasicDBGSOM.compute_topographic_error(self.model_, X_jl)
        return float(error)

