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
    bayesian : bool, default=False
        If True, run Bayesian hyperparameter optimization during fit().
        The provided sf, max_neurons, n_iter, init_size are ignored and
        optimal values are found automatically using Julia's native implementation.
        Objective: Minimize QE subject to TE <= bayesian_te_constraint.
    bayesian_objective : str, default='qe_te'
        Objective function for Bayesian optimization (kept for API compatibility).
        Currently only 'qe_te' is supported (minimize QE with TE constraint).
    bayesian_trials : int, default=30
        Number of Bayesian optimization trials (only used if bayesian=True).
    bayesian_te_constraint : float, default=0.25
        Maximum topographic error allowed during optimization.
    bayesian_ranges : dict, optional
        Custom search ranges for Bayesian optimization. Keys:
        - 'init_size': (min, max) for grid size, default (2, 5)
        - 'max_neurons': (min, max), default (20, 200)
        - 'sf': (min, max), default (0.1, 0.9)
        - 'n_iter': (min, max), default (50, 200)

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
        bayesian: bool = False,
        bayesian_objective: str = 'qe_te',
        bayesian_trials: int = 30,
        bayesian_te_constraint: float = 0.25,
        bayesian_ranges: Optional[Dict] = None
    ):
        self.sf = sf
        self.max_neurons = max_neurons
        self.n_iter = n_iter
        self.init_size = init_size
        self.preprocess = preprocess
        self.preprocessor_kwargs = preprocessor_kwargs or {}
        self.random_state = random_state
        self.bayesian = bayesian
        self.bayesian_objective = bayesian_objective
        self.bayesian_trials = bayesian_trials
        self.bayesian_te_constraint = bayesian_te_constraint
        self.bayesian_ranges = bayesian_ranges or {}

        # Fitted attributes
        self.model_ = None
        self.preprocessor_: Optional[DataPreprocessor] = None
        self.n_features_in_: Optional[int] = None
        self.n_neurons_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.optimization_result_: Optional[Dict] = None  # Stores Bayesian opt results
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

        # Run Bayesian optimization if requested
        if self.bayesian:
            return self._fit_with_bayesian(X)

        return self._fit_direct(X)

    def _fit_with_bayesian(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> 'DBGSOMWrapper':
        """Run Bayesian optimization using Julia implementation."""
        jl = self._ensure_julia()

        # Prepare data (fit preprocessor here so Julia gets clean data)
        X_julia = self._prepare_data(X, fit_preprocessor=True)
        n_features, n_samples = X_julia.shape
        self.n_features_in_ = n_features

        # Convert to Julia matrix
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        # Get search ranges (use defaults or user-provided)
        ranges = self.bayesian_ranges
        init_size_range = ranges.get('init_size', (2, 5))
        max_neurons_range = ranges.get('max_neurons', (20, 200))
        sf_range = ranges.get('sf', (0.1, 0.9))
        n_iter_range = ranges.get('n_iter', (50, 200))

        # Set random seed if specified
        if self.random_state is not None:
            jl.seval(f'Random.seed!({self.random_state})')

        # Call Julia bayesian_optimize
        result_jl = jl.BasicDBGSOM.bayesian_optimize(
            X_jl,
            sf_range=(float(sf_range[0]), float(sf_range[1])),
            max_neurons_range=(int(max_neurons_range[0]), int(max_neurons_range[1])),
            n_iter_range=(int(n_iter_range[0]), int(n_iter_range[1])),
            init_size_range=(int(init_size_range[0]), int(init_size_range[1])),
            te_constraint=float(self.bayesian_te_constraint),
            n_trials=int(self.bayesian_trials),
            n_startup=min(10, self.bayesian_trials // 3),
            seed=self.random_state if self.random_state is not None else jl.nothing,
            verbose=True
        )

        # Extract results from Julia
        best_model_jl = result_jl.best_model
        best_params_jl = result_jl.best_params
        best_qe = float(result_jl.best_qe)
        best_te = float(result_jl.best_te)

        if best_model_jl is None or best_model_jl == jl.nothing:
            raise RuntimeError(
                "Bayesian optimization found no feasible solution. "
                "Consider relaxing te_constraint or expanding search ranges."
            )

        # Store the Julia model directly
        self.model_ = best_model_jl

        # Extract best parameters
        best_params = {
            'init_size': (int(best_params_jl[jl.seval(':init_size')][0]),
                         int(best_params_jl[jl.seval(':init_size')][1])),
            'max_neurons': int(best_params_jl[jl.seval(':max_neurons')]),
            'sf': float(best_params_jl[jl.seval(':sf')]),
            'n_iter': int(best_params_jl[jl.seval(':n_iter')])
        }

        # Update self with optimal parameters
        self.sf = best_params['sf']
        self.max_neurons = best_params['max_neurons']
        self.n_iter = best_params['n_iter']
        self.init_size = best_params['init_size']

        # Convert all_trials from Julia to Python
        all_trials_jl = result_jl.all_trials
        all_trials = []
        for t in all_trials_jl:
            params_jl = t[jl.seval(':params')]
            init_size_jl = params_jl[jl.seval(':init_size')]
            all_trials.append({
                'init_size': (int(init_size_jl[0]), int(init_size_jl[1])),
                'max_neurons': int(params_jl[jl.seval(':max_neurons')]),
                'sf': float(params_jl[jl.seval(':sf')]),
                'n_iter': int(params_jl[jl.seval(':n_iter')]),
                'qe': float(t[jl.seval(':qe')]),
                'te': float(t[jl.seval(':te')]),
                'feasible': bool(t[jl.seval(':feasible')]),
                'score': float(t[jl.seval(':score')])
            })

        # Store optimization results
        self.optimization_result_ = {
            'best_params': best_params,
            'best_qe': best_qe,
            'best_te': best_te,
            'objective': self.bayesian_objective,
            'te_constraint': self.bayesian_te_constraint,
            'all_trials': sorted(all_trials, key=lambda x: x['score'])
        }

        # Store fitted attributes
        self.n_neurons_ = int(jl.BasicDBGSOM.n_neurons(self.model_))
        self._is_fitted = True

        return self

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

    @classmethod
    def bayesian_optimize(
        cls,
        X: Union[pd.DataFrame, np.ndarray, str],
        n_trials: int = 50,
        init_size_range: Tuple[int, int] = (2, 5),
        max_neurons_range: Tuple[int, int] = (20, 200),
        sf_range: Tuple[float, float] = (0.1, 0.9),
        n_iter_range: Tuple[int, int] = (50, 200),
        te_constraint: float = 0.25,
        preprocess: bool = True,
        preprocessor_kwargs: Optional[Dict] = None,
        random_state: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Find optimal hyperparameters using Bayesian optimization (Julia implementation).

        Minimizes Quantization Error (QE) subject to Topographic Error (TE) <= te_constraint.

        Parameters
        ----------
        X : DataFrame, ndarray, or str
            Training data.
        n_trials : int, default=50
            Number of optimization trials.
        init_size_range : tuple of int, default=(2, 5)
            Range for initial grid size (both dimensions).
        max_neurons_range : tuple of int, default=(20, 200)
            Range for maximum neurons.
        sf_range : tuple of float, default=(0.1, 0.9)
            Range for spreading factor.
        n_iter_range : tuple of int, default=(50, 200)
            Range for number of iterations.
        te_constraint : float, default=0.25
            Maximum topographic error allowed.
        preprocess : bool, default=True
            Whether to preprocess data.
        preprocessor_kwargs : dict, optional
            Arguments for preprocessor.
        random_state : int, optional
            Random seed for reproducibility.
        verbose : bool, default=True
            Whether to show progress.

        Returns
        -------
        result : dict
            - 'best_params': Best hyperparameters found
            - 'best_qe': Best quantization error achieved
            - 'best_te': Topographic error of best model
            - 'best_model': Fitted DBGSOMWrapper with best parameters
            - 'all_trials': List of all trial results
        """
        # Create a temporary instance to access Julia
        temp_instance = cls(
            preprocess=preprocess,
            preprocessor_kwargs=preprocessor_kwargs,
            random_state=random_state
        )
        jl = temp_instance._ensure_julia()

        # Load data
        if isinstance(X, str):
            X = pd.read_csv(X)

        # Prepare data using the temporary instance
        X_julia = temp_instance._prepare_data(X, fit_preprocessor=True)

        # Convert to Julia matrix
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        # Set random seed if specified
        if random_state is not None:
            jl.seval(f'Random.seed!({random_state})')

        # Call Julia bayesian_optimize
        result_jl = jl.BasicDBGSOM.bayesian_optimize(
            X_jl,
            sf_range=(float(sf_range[0]), float(sf_range[1])),
            max_neurons_range=(int(max_neurons_range[0]), int(max_neurons_range[1])),
            n_iter_range=(int(n_iter_range[0]), int(n_iter_range[1])),
            init_size_range=(int(init_size_range[0]), int(init_size_range[1])),
            te_constraint=float(te_constraint),
            n_trials=int(n_trials),
            n_startup=min(10, n_trials // 3),
            seed=random_state if random_state is not None else jl.nothing,
            verbose=verbose
        )

        # Extract results from Julia
        best_model_jl = result_jl.best_model
        best_params_jl = result_jl.best_params
        best_qe = float(result_jl.best_qe)
        best_te = float(result_jl.best_te)

        if best_model_jl is None or best_model_jl == jl.nothing:
            return {
                'best_params': None,
                'best_qe': float('inf'),
                'best_te': float('inf'),
                'best_model': None,
                'all_trials': [],
                'error': 'No feasible solution found'
            }

        # Extract best parameters
        best_params = {
            'init_size': (int(best_params_jl[jl.seval(':init_size')][0]),
                         int(best_params_jl[jl.seval(':init_size')][1])),
            'max_neurons': int(best_params_jl[jl.seval(':max_neurons')]),
            'sf': float(best_params_jl[jl.seval(':sf')]),
            'n_iter': int(best_params_jl[jl.seval(':n_iter')])
        }

        # Create a wrapper with the best model
        best_wrapper = cls(
            sf=best_params['sf'],
            max_neurons=best_params['max_neurons'],
            n_iter=best_params['n_iter'],
            init_size=best_params['init_size'],
            preprocess=preprocess,
            preprocessor_kwargs=preprocessor_kwargs,
            random_state=random_state
        )
        best_wrapper.model_ = best_model_jl
        best_wrapper.preprocessor_ = temp_instance.preprocessor_
        best_wrapper.n_features_in_ = temp_instance.n_features_in_
        best_wrapper.feature_names_in_ = temp_instance.feature_names_in_
        best_wrapper.n_neurons_ = int(jl.BasicDBGSOM.n_neurons(best_model_jl))
        best_wrapper._is_fitted = True

        # Convert all_trials from Julia to Python
        all_trials_jl = result_jl.all_trials
        all_trials = []
        for t in all_trials_jl:
            params_jl = t[jl.seval(':params')]
            init_size_jl = params_jl[jl.seval(':init_size')]
            all_trials.append({
                'params': {
                    'init_size': (int(init_size_jl[0]), int(init_size_jl[1])),
                    'max_neurons': int(params_jl[jl.seval(':max_neurons')]),
                    'sf': float(params_jl[jl.seval(':sf')]),
                    'n_iter': int(params_jl[jl.seval(':n_iter')])
                },
                'qe': float(t[jl.seval(':qe')]),
                'te': float(t[jl.seval(':te')]),
                'feasible': bool(t[jl.seval(':feasible')]),
                'score': float(t[jl.seval(':score')])
            })

        return {
            'best_params': best_params,
            'best_qe': best_qe,
            'best_te': best_te,
            'best_model': best_wrapper,
            'all_trials': sorted(all_trials, key=lambda x: x['score'])
        }
