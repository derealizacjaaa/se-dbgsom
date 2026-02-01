"""
SE-DBGSOM: Statistical Error Dynamic Batch Growing Self-Organizing Map.

Training-only wrapper using the StatisticalError growth threshold method.
Clustering is handled separately via the clustering module.
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


class SEDBGSOM:
    """
    Statistical Error DBGSOM - Training-focused SOM wrapper.

    Always uses the StatisticalError growth threshold method which adapts
    to data statistics: GT = λ × sqrt(Σ std_i²)

    For clustering, use the separate clustering functions:
    - cluster_acute(som, X) - for non-convex/complex boundaries
    - cluster_leiden(som, X) - for convex/blob-like clusters

    Parameters
    ----------
    lambda_ : float, default=1.0
        Growth threshold scaling factor.
        - λ ~ 0.5: Aggressive growth (more neurons)
        - λ ~ 1.0: Balanced (default)
        - λ ~ 2.0: Conservative (fewer neurons)
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

    Bayesian Optimization Parameters
    --------------------------------
    bayesian : bool, default=False
        If True, run Bayesian optimization to find optimal parameters.
        Objective: Minimize QE subject to TE <= te_constraint.
    bayesian_trials : int, default=30
        Number of optimization trials.
    bayesian_te_constraint : float, default=0.25
        Maximum topographic error allowed (default 25%).
    bayesian_ranges : dict, optional
        Custom search ranges:
        - 'lambda_': (min, max), default (0.3, 2.0)
        - 'max_neurons': (min, max), default (50, 200)
        - 'n_iter': (min, max), default (50, 200)
        - 'init_size': (min, max), default (2, 5)

    Attributes
    ----------
    model_ : Julia DBGSOM object
        The underlying Julia model (after fitting).
    n_neurons_ : int
        Number of neurons after training.
    optimization_result_ : dict
        Results from Bayesian optimization (if used).

    Examples
    --------
    >>> from dbgsom import SEDBGSOM, cluster_acute, cluster_leiden
    >>>
    >>> # Train model
    >>> som = SEDBGSOM(max_neurons=100, bayesian=True)
    >>> som.fit(X)
    >>>
    >>> # Cluster with ACUTE (good for non-convex shapes)
    >>> labels = cluster_acute(som, X)
    >>>
    >>> # Or cluster with Leiden (good for blob-like clusters)
    >>> labels = cluster_leiden(som, X, n_clusters=3)
    """

    def __init__(
        self,
        lambda_: float = 1.0,
        max_neurons: int = 100,
        n_iter: int = 100,
        init_size: Tuple[int, int] = (2, 2),
        preprocess: bool = True,
        preprocessor_kwargs: Optional[Dict] = None,
        random_state: Optional[int] = None,
        bayesian: bool = False,
        bayesian_trials: int = 30,
        bayesian_te_constraint: float = 0.25,
        bayesian_ranges: Optional[Dict] = None
    ):
        self.lambda_ = lambda_
        self.max_neurons = max_neurons
        self.n_iter = n_iter
        self.init_size = init_size
        self.preprocess = preprocess
        self.preprocessor_kwargs = preprocessor_kwargs or {}
        self.random_state = random_state
        self.bayesian = bayesian
        self.bayesian_trials = bayesian_trials
        self.bayesian_te_constraint = bayesian_te_constraint
        self.bayesian_ranges = bayesian_ranges or {}

        # Fitted attributes
        self.model_ = None
        self.preprocessor_: Optional[DataPreprocessor] = None
        self.n_features_in_: Optional[int] = None
        self.n_neurons_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.optimization_result_: Optional[Dict] = None
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
        """Prepare data for Julia (n_features, n_samples), Float64."""
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
    ) -> 'SEDBGSOM':
        """
        Fit the SE-DBGSOM model.

        Parameters
        ----------
        X : DataFrame, ndarray, or str (path to CSV)
            Training data.
        y : ignored
            For sklearn API compatibility.

        Returns
        -------
        self : SEDBGSOM
            Fitted model.
        """
        if isinstance(X, str):
            X = pd.read_csv(X)

        if self.bayesian:
            return self._fit_with_bayesian(X)
        return self._fit_direct(X)

    def _fit_with_bayesian(self, X: Union[pd.DataFrame, np.ndarray]) -> 'SEDBGSOM':
        """Run Bayesian optimization with StatisticalError method."""
        jl = self._ensure_julia()

        X_julia = self._prepare_data(X, fit_preprocessor=True)
        n_features, n_samples = X_julia.shape
        self.n_features_in_ = n_features

        X_jl = numpy_to_julia_matrix(jl, X_julia)

        # Get search ranges
        ranges = self.bayesian_ranges
        lambda_range = ranges.get('lambda_', (0.3, 2.0))
        max_neurons_range = ranges.get('max_neurons', (50, 200))
        n_iter_range = ranges.get('n_iter', (50, 200))
        init_size_range = ranges.get('init_size', (2, 5))

        if self.random_state is not None:
            jl.seval(f'Random.seed!({self.random_state})')

        # Call Julia bayesian_optimize with StatisticalError method
        result_jl = jl.BasicDBGSOM.bayesian_optimize(
            X_jl,
            gt_method_type=jl.seval(':statistical_error'),
            lambda_range=(float(lambda_range[0]), float(lambda_range[1])),
            max_neurons_range=(int(max_neurons_range[0]), int(max_neurons_range[1])),
            n_iter_range=(int(n_iter_range[0]), int(n_iter_range[1])),
            init_size_range=(int(init_size_range[0]), int(init_size_range[1])),
            te_constraint=float(self.bayesian_te_constraint),
            n_trials=int(self.bayesian_trials),
            n_startup=min(10, self.bayesian_trials // 3),
            seed=self.random_state if self.random_state is not None else jl.nothing,
            verbose=True
        )

        best_model_jl = result_jl.best_model
        best_params_jl = result_jl.best_params
        best_qe = float(result_jl.best_qe)
        best_te = float(result_jl.best_te)

        if best_model_jl is None or best_model_jl == jl.nothing:
            raise RuntimeError(
                "Bayesian optimization found no feasible solution. "
                "Consider relaxing te_constraint or expanding search ranges."
            )

        self.model_ = best_model_jl

        # Extract best parameters
        best_params = {
            'init_size': (int(best_params_jl[jl.seval(':init_size')][0]),
                         int(best_params_jl[jl.seval(':init_size')][1])),
            'max_neurons': int(best_params_jl[jl.seval(':max_neurons')]),
            'lambda_': float(best_params_jl[jl.seval(':lambda')]),
            'n_iter': int(best_params_jl[jl.seval(':n_iter')])
        }

        # Update self with optimal parameters
        self.lambda_ = best_params['lambda_']
        self.max_neurons = best_params['max_neurons']
        self.n_iter = best_params['n_iter']
        self.init_size = best_params['init_size']

        # Store optimization results
        self.optimization_result_ = {
            'best_params': best_params,
            'best_qe': best_qe,
            'best_te': best_te,
            'te_constraint': self.bayesian_te_constraint,
        }

        self.n_neurons_ = int(jl.BasicDBGSOM.n_neurons(self.model_))
        self._is_fitted = True

        return self

    def _fit_direct(self, X: Union[pd.DataFrame, np.ndarray]) -> 'SEDBGSOM':
        """Fit directly with current parameters using StatisticalError."""
        jl = self._ensure_julia()

        X_julia = self._prepare_data(X, fit_preprocessor=True)
        n_features, n_samples = X_julia.shape
        self.n_features_in_ = n_features

        if self.random_state is not None:
            jl.seval(f'Random.seed!({self.random_state})')

        X_jl = numpy_to_julia_matrix(jl, X_julia)

        # Create Julia DBGSOM with StatisticalError method
        init_x, init_y = self.init_size
        gt_method = jl.BasicDBGSOM.StatisticalError(float(self.lambda_))

        self.model_ = jl.BasicDBGSOM.DBGSOM[jl.Float64](
            input_dim=n_features,
            gt_method=gt_method,
            max_neurons=int(self.max_neurons),
            n_iter=int(self.n_iter),
            init_size=(init_x, init_y)
        )

        # Train the model
        jl.BasicDBGSOM.fit_b(self.model_, X_jl)

        self.n_neurons_ = int(jl.BasicDBGSOM.n_neurons(self.model_))
        self._is_fitted = True

        return self

    # =========================================================================
    # Prediction / Transform Methods
    # =========================================================================

    def predict(self, X: Union[pd.DataFrame, np.ndarray, str]) -> List[Tuple[int, int]]:
        """
        Predict BMU positions for samples.

        Returns list of (x, y) grid positions.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        if isinstance(X, str):
            X = pd.read_csv(X)

        X_julia = self._prepare_data(X)
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        bmus_jl = jl.BasicDBGSOM.predict(self.model_, X_jl)
        return julia_vector_of_tuples_to_python(bmus_jl)

    def transform(self, X: Union[pd.DataFrame, np.ndarray, str]) -> np.ndarray:
        """
        Transform samples to BMU coordinates.

        Returns ndarray of shape (n_samples, 2).
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        if isinstance(X, str):
            X = pd.read_csv(X)

        X_julia = self._prepare_data(X)
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        coords_jl = jl.BasicDBGSOM.transform(self.model_, X_jl)
        return julia_to_numpy_matrix(coords_jl).T

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray, str],
        y: Optional[Any] = None
    ) -> np.ndarray:
        """Fit and return transformed coordinates."""
        return self.fit(X, y).transform(X)

    # =========================================================================
    # Grid / Neuron Information
    # =========================================================================

    def get_neuron_weights(self) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Get all neuron weights.

        Returns
        -------
        positions : list of (x, y) tuples
        weights : ndarray of shape (n_neurons, n_features)
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        result = jl.BasicDBGSOM.get_positions_and_weights(self.model_)
        pos_jl = result[0]
        weights_jl = result[1]

        positions = julia_vector_of_tuples_to_python(pos_jl)
        weights = julia_to_numpy_matrix(weights_jl).T

        return positions, weights

    def get_grid_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get grid bounding box: ((x_min, x_max), (y_min, y_max))."""
        self._check_is_fitted()
        jl = self._ensure_julia()

        bounds_jl = jl.BasicDBGSOM.grid_bounds(self.model_)
        x_bounds = (int(bounds_jl[0][0]), int(bounds_jl[0][1]))
        y_bounds = (int(bounds_jl[1][0]), int(bounds_jl[1][1]))

        return (x_bounds, y_bounds)

    def get_umatrix(self) -> Dict[Tuple[int, int], float]:
        """
        Compute the U-matrix (average distance to neighbors).

        High values = cluster boundaries, low values = cluster interiors.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        umat_jl = jl.BasicDBGSOM.compute_umatrix(self.model_)
        return julia_dict_to_python(umat_jl, value_type='float')

    def get_expanded_umatrix(
        self, log: bool = False
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]]:
        """
        Compute expanded U-matrix with per-edge distances.

        Parameters
        ----------
        log : bool
            If True, apply log transform to distances.

        Returns
        -------
        neuron_umat : dict
            Neuron position -> average neighbor distance.
        edge_umat : dict
            Edge ((pos1), (pos2)) -> pairwise distance between adjacent neurons.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        neuron_jl, edge_jl = jl.BasicDBGSOM.compute_expanded_umatrix(self.model_, log=log)

        # Convert neuron dict
        neuron_umat = julia_dict_to_python(neuron_jl, value_type='float')

        # Convert edge dict (keys are tuples of tuples)
        edge_umat = {}
        for key, val in dict(edge_jl).items():
            pos1 = (int(key[0][0]), int(key[0][1]))
            pos2 = (int(key[1][0]), int(key[1][1]))
            edge_umat[(pos1, pos2)] = float(val)

        return neuron_umat, edge_umat

    def get_hit_counts(self, X: Union[pd.DataFrame, np.ndarray, str]) -> Dict[Tuple[int, int], int]:
        """Count samples assigned to each neuron."""
        self._check_is_fitted()
        jl = self._ensure_julia()

        if isinstance(X, str):
            X = pd.read_csv(X)

        X_julia = self._prepare_data(X)
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        hits_jl = jl.BasicDBGSOM.compute_hit_counts(self.model_, X_jl)
        return julia_dict_to_python(hits_jl, value_type='int')

    # =========================================================================
    # Quality Metrics
    # =========================================================================

    def quantization_error(self, X: Union[pd.DataFrame, np.ndarray, str]) -> float:
        """Compute mean quantization error."""
        self._check_is_fitted()
        jl = self._ensure_julia()

        if isinstance(X, str):
            X = pd.read_csv(X)

        X_julia = self._prepare_data(X)
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        qe = jl.BasicDBGSOM.compute_quantization_error(self.model_, X_jl)
        return float(qe)

    def topographic_error(self, X: Union[pd.DataFrame, np.ndarray, str]) -> float:
        """
        Compute topographic error.

        Proportion of samples where BMU1 and BMU2 are not adjacent.
        0 = perfect topology, 1 = worst.
        """
        self._check_is_fitted()
        jl = self._ensure_julia()

        if isinstance(X, str):
            X = pd.read_csv(X)

        X_julia = self._prepare_data(X)
        X_jl = numpy_to_julia_matrix(jl, X_julia)

        error = jl.BasicDBGSOM.compute_topographic_error(self.model_, X_jl)
        return float(error)

    # =========================================================================
    # Utilities
    # =========================================================================

    def summary(self) -> str:
        """Get a text summary of the model."""
        self._check_is_fitted()

        lines = [
            "SE-DBGSOM Model Summary",
            "=" * 40,
            "",
            "Configuration (StatisticalError method):",
            f"  Lambda: {self.lambda_}",
            f"  Max neurons: {self.max_neurons}",
            f"  Iterations: {self.n_iter}",
            "",
            "Trained model:",
            f"  Input dimension: {self.n_features_in_}",
            f"  Neurons: {self.n_neurons_}",
        ]

        bounds = self.get_grid_bounds()
        lines.append(f"  Grid bounds: x=[{bounds[0][0]}, {bounds[0][1]}], y=[{bounds[1][0]}, {bounds[1][1]}]")

        if self.optimization_result_:
            lines.append("")
            lines.append("Bayesian optimization:")
            lines.append(f"  Best QE: {self.optimization_result_['best_qe']:.4f}")
            lines.append(f"  Best TE: {self.optimization_result_['best_te']:.4f}")

        return "\n".join(lines)

    def _check_is_fitted(self):
        """Check if model is fitted."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
