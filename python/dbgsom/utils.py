"""
Utility functions for Python-Julia type conversion.
"""

import numpy as np
from typing import Any, Dict, List, Tuple


def numpy_to_julia_matrix(jl: Any, arr: np.ndarray) -> Any:
    """
    Convert numpy array to Julia Matrix{Float64}.

    Parameters
    ----------
    jl : Julia Main module
        The Julia runtime.
    arr : ndarray
        Numpy array to convert. Should be 2D.

    Returns
    -------
    julia_matrix : Julia Matrix{Float64}
        The converted matrix.
    """
    # Ensure correct dtype and contiguous memory
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    # juliacall handles conversion automatically
    return arr


def julia_to_numpy_matrix(jl_matrix: Any) -> np.ndarray:
    """
    Convert Julia matrix to numpy array.

    Parameters
    ----------
    jl_matrix : Julia Matrix
        Julia matrix to convert.

    Returns
    -------
    arr : ndarray
        Numpy array.
    """
    return np.array(jl_matrix, dtype=np.float64)


def julia_tuple_to_python(jl_tuple: Any) -> Tuple[int, int]:
    """Convert Julia tuple to Python tuple."""
    return (int(jl_tuple[0]), int(jl_tuple[1]))


def julia_dict_to_python(
    jl_dict: Any,
    value_type: str = 'float'
) -> Dict[Tuple[int, int], Any]:
    """
    Convert Julia Dict with tuple keys to Python dict.

    Parameters
    ----------
    jl_dict : Julia Dict
        Julia dictionary to convert.
    value_type : str
        Type of values: 'float', 'int', or 'any'.

    Returns
    -------
    py_dict : dict
        Python dictionary with tuple keys.
    """
    result = {}
    for key in jl_dict.keys():
        py_key = (int(key[0]), int(key[1]))
        val = jl_dict[key]
        if value_type == 'float':
            result[py_key] = float(val)
        elif value_type == 'int':
            result[py_key] = int(val)
        else:
            result[py_key] = val
    return result


def julia_vector_of_tuples_to_python(jl_vec: Any) -> List[Tuple[int, int]]:
    """Convert Julia Vector of tuples to Python list of tuples."""
    return [(int(t[0]), int(t[1])) for t in jl_vec]
