"""
Smart data preprocessor for DBGSOM.

Automatically detects binary vs continuous columns and applies
appropriate normalization.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


class ColumnType(Enum):
    """Types of columns detected during preprocessing."""
    BINARY = "binary"
    CONTINUOUS = "continuous"


class NormalizationMethod(Enum):
    """Normalization methods available."""
    MINMAX = "minmax"
    ZSCORE = "zscore"
    NONE = "none"


@dataclass
class ColumnInfo:
    """Information about a single column."""
    name: str
    dtype: ColumnType
    normalization: NormalizationMethod
    original_index: int = 0
    # Parameters for inverse transform
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    std_val: Optional[float] = None


@dataclass
class PreprocessorState:
    """Complete state of the preprocessor."""
    columns: List[ColumnInfo] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    n_features: int = 0
    fitted: bool = False


class DataPreprocessor:
    """
    Smart data preprocessor for DBGSOM.

    Automatically detects binary vs continuous columns and applies
    appropriate normalization. Tracks all transformation parameters
    for inverse transforms.

    Parameters
    ----------
    continuous_normalization : str, default='zscore'
        Normalization method for continuous columns: 'zscore' or 'minmax'.
    binary_normalization : str, default='minmax'
        Normalization method for binary columns: 'minmax' or 'none'.

    Attributes
    ----------
    state : PreprocessorState
        Current state including column info and fit status.

    Examples
    --------
    >>> import pandas as pd
    >>> from dbgsom import DataPreprocessor
    >>>
    >>> df = pd.DataFrame({
    ...     'gdp': [1000, 2000, 3000],
    ...     'has_oil': [0, 1, 1],
    ...     'landlocked': [1, 0, 0]
    ... })
    >>>
    >>> prep = DataPreprocessor()
    >>> X = prep.fit_transform(df)
    >>> print(prep.get_binary_columns())
    ['has_oil', 'landlocked']
    """

    def __init__(
        self,
        continuous_normalization: str = 'zscore',
        binary_normalization: str = 'minmax'
    ):
        self.continuous_normalization = NormalizationMethod(continuous_normalization)
        self.binary_normalization = NormalizationMethod(binary_normalization)
        self.state = PreprocessorState()

    def _detect_column_type(self, values: np.ndarray, name: str) -> ColumnType:
        """
        Detect if a column is binary or continuous.

        Binary detection criteria:
        - Only contains values in {0, 1} or {True, False}
        - Or exactly 2 unique numeric values
        """
        # Handle NaN
        if np.issubdtype(values.dtype, np.floating):
            valid_values = values[~np.isnan(values)]
        else:
            valid_values = values

        if len(valid_values) == 0:
            return ColumnType.CONTINUOUS

        unique_vals = np.unique(valid_values)

        # Check for binary patterns
        if len(unique_vals) <= 2:
            # Check if values are 0/1
            if set(unique_vals).issubset({0, 0.0, 1, 1.0, True, False}):
                return ColumnType.BINARY

        return ColumnType.CONTINUOUS

    def _compute_normalization_params(
        self,
        values: np.ndarray,
        col_type: ColumnType
    ) -> Tuple[NormalizationMethod, Dict]:
        """Compute normalization parameters for a column."""
        # Handle NaN
        if np.issubdtype(values.dtype, np.floating):
            valid_values = values[~np.isnan(values)]
        else:
            valid_values = values.astype(np.float64)

        if len(valid_values) == 0:
            return NormalizationMethod.NONE, {}

        if col_type == ColumnType.BINARY:
            method = self.binary_normalization
        else:
            method = self.continuous_normalization

        params = {}
        if method == NormalizationMethod.MINMAX:
            params['min_val'] = float(np.nanmin(valid_values))
            params['max_val'] = float(np.nanmax(valid_values))
        elif method == NormalizationMethod.ZSCORE:
            params['mean_val'] = float(np.nanmean(valid_values))
            params['std_val'] = float(np.nanstd(valid_values))
            if params['std_val'] == 0:
                params['std_val'] = 1.0  # Avoid division by zero

        return method, params

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> 'DataPreprocessor':
        """
        Fit the preprocessor to data.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input data. If DataFrame, column names are preserved.
            Shape should be (n_samples, n_features).
        feature_names : list of str, optional
            Feature names if X is ndarray.

        Returns
        -------
        self : DataPreprocessor
        """
        # Convert to numpy and extract names
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        n_samples, n_features = X_arr.shape

        self.state.columns = []
        self.state.feature_names = list(feature_names)
        self.state.n_features = n_features

        for i in range(n_features):
            col_values = X_arr[:, i]
            name = feature_names[i]

            col_type = self._detect_column_type(col_values, name)
            norm_method, params = self._compute_normalization_params(col_values, col_type)

            col_info = ColumnInfo(
                name=name,
                dtype=col_type,
                normalization=norm_method,
                original_index=i,
                **params
            )
            self.state.columns.append(col_info)

        self.state.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform data using fitted parameters.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input data with same features as fitting data.

        Returns
        -------
        X_transformed : ndarray of shape (n_features, n_samples)
            Transformed data in Julia-expected format.
        """
        if not self.state.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        n_samples, n_features = X_arr.shape

        if n_features != self.state.n_features:
            raise ValueError(
                f"Expected {self.state.n_features} features, got {n_features}"
            )

        X_transformed = np.zeros_like(X_arr)

        for col_info in self.state.columns:
            i = col_info.original_index
            col_values = X_arr[:, i]

            if col_info.normalization == NormalizationMethod.MINMAX:
                range_val = col_info.max_val - col_info.min_val
                if range_val > 0:
                    X_transformed[:, i] = (col_values - col_info.min_val) / range_val
                else:
                    X_transformed[:, i] = 0.0
            elif col_info.normalization == NormalizationMethod.ZSCORE:
                X_transformed[:, i] = (col_values - col_info.mean_val) / col_info.std_val
            else:
                X_transformed[:, i] = col_values

        # Transpose to (n_features, n_samples) for Julia
        return X_transformed.T.astype(np.float64)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, feature_names).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_samples)
            Transformed data in Julia format.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Data in original scale.
        """
        if not self.state.fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")

        # Transpose from (features, samples) to (samples, features)
        X_arr = X.T.copy()
        n_samples, n_features = X_arr.shape

        X_original = np.zeros_like(X_arr)

        for col_info in self.state.columns:
            i = col_info.original_index
            col_values = X_arr[:, i]

            if col_info.normalization == NormalizationMethod.MINMAX:
                range_val = col_info.max_val - col_info.min_val
                X_original[:, i] = col_values * range_val + col_info.min_val
            elif col_info.normalization == NormalizationMethod.ZSCORE:
                X_original[:, i] = col_values * col_info.std_val + col_info.mean_val
            else:
                X_original[:, i] = col_values

        return X_original

    def get_column_types(self) -> Dict[str, ColumnType]:
        """Get detected column types as a dictionary."""
        return {col.name: col.dtype for col in self.state.columns}

    def get_binary_columns(self) -> List[str]:
        """Get names of columns detected as binary."""
        return [col.name for col in self.state.columns if col.dtype == ColumnType.BINARY]

    def get_continuous_columns(self) -> List[str]:
        """Get names of columns detected as continuous."""
        return [col.name for col in self.state.columns if col.dtype == ColumnType.CONTINUOUS]

    def summary(self) -> str:
        """Get a summary of detected column types."""
        if not self.state.fitted:
            return "Preprocessor not fitted"

        lines = ["DataPreprocessor Summary", "=" * 40, ""]

        binary_cols = self.get_binary_columns()
        cont_cols = self.get_continuous_columns()

        lines.append(f"Total features: {self.state.n_features}")
        lines.append(f"Binary features: {len(binary_cols)}")
        lines.append(f"Continuous features: {len(cont_cols)}")
        lines.append("")

        if binary_cols:
            lines.append("Binary columns:")
            for name in binary_cols:
                lines.append(f"  - {name}")
            lines.append("")

        if cont_cols:
            lines.append("Continuous columns:")
            for name in cont_cols:
                lines.append(f"  - {name}")

        return "\n".join(lines)
