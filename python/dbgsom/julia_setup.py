"""
Julia environment setup utilities for DBGSOM.

Handles lazy loading of Julia and initialization of the BasicDBGSOM module.
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded Julia runtime
_jl = None
_julia_ready = False


def get_package_root() -> Path:
    """Get the root directory of the dbg_som project."""
    # Navigate from python/dbgsom/julia_setup.py to project root
    return Path(__file__).parent.parent.parent


def get_julia() -> Any:
    """
    Lazy-load Julia runtime.

    Returns
    -------
    jl : Julia Main module
        The Julia runtime.
    """
    global _jl
    if _jl is None:
        try:
            from juliacall import Main as jl
            _jl = jl
        except ImportError:
            raise ImportError(
                "juliacall is required. Install with: pip install juliacall"
            )
    return _jl


def setup_julia_environment(
    julia_project_path: Optional[str] = None,
    install_packages: bool = True,
    verbose: bool = False
) -> None:
    """
    Initialize Julia environment for DBGSOM.

    Parameters
    ----------
    julia_project_path : str, optional
        Path to Julia project. Defaults to the dbg_som package root.
    install_packages : bool, default=True
        Whether to instantiate Julia packages if needed.
    verbose : bool, default=False
        Print setup progress.
    """
    global _julia_ready

    jl = get_julia()

    if julia_project_path is None:
        julia_project_path = str(get_package_root())

    # Normalize path for Julia (use forward slashes)
    julia_project_path = julia_project_path.replace('\\', '/')

    if verbose:
        logger.info(f"Setting up Julia environment at: {julia_project_path}")

    # Check if DBGSOM.jl exists
    basic_path = Path(julia_project_path) / "src" / "DBGSOM.jl"
    if not basic_path.exists():
        raise FileNotFoundError(
            f"DBGSOM.jl not found at {basic_path}. "
            "Please provide the correct project path."
        )

    # Load required Julia packages
    jl.seval('using Random')
    jl.seval('using Statistics')

    # Load the BasicDBGSOM module
    basic_path_str = str(basic_path).replace('\\', '/')
    if verbose:
        logger.info(f"Loading BasicDBGSOM from: {basic_path_str}")

    jl.seval(f'include("{basic_path_str}")')
    jl.seval('using .BasicDBGSOM')

    _julia_ready = True

    if verbose:
        logger.info("Julia environment ready!")


def is_julia_ready() -> bool:
    """
    Check if Julia environment is properly initialized.

    Returns
    -------
    ready : bool
        True if Julia is ready with BasicDBGSOM loaded.
    """
    global _julia_ready
    if not _julia_ready:
        return False

    try:
        jl = get_julia()
        # Try to access BasicDBGSOM
        jl.seval('BasicDBGSOM.DBGSOM')
        return True
    except (AttributeError, RuntimeError, ImportError, NameError):
        _julia_ready = False
        return False


def ensure_julia_ready(verbose: bool = False) -> None:
    """
    Ensure Julia is ready, setting up if needed.

    Parameters
    ----------
    verbose : bool, default=False
        Print setup progress if initialization is needed.
    """
    if not is_julia_ready():
        setup_julia_environment(verbose=verbose)


def get_julia_version() -> str:
    """Get the Julia version string."""
    jl = get_julia()
    return str(jl.seval('string(VERSION)'))
