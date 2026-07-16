"""Runtime array/data validators for bayesian_panel_nmf.

Moved verbatim out of validation.py in Phase 3 (Task 3.3): these validate
in-memory data structures (data dicts, MCMC samples, predictions arrays,
filepaths, group lists) at runtime, as opposed to validation.py's config-shape
checks which are now delegated to the pydantic schema in config.py.
"""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from bayesian_panel_nmf.validation import DataError


def _safe_rmtree(path: Path, allowed_parent: Path) -> None:
    """Remove a directory tree only if strictly inside ``allowed_parent``.

    Refuses to remove ``allowed_parent`` itself, its ancestors, or any path
    that is not a descendant of it. Intended for cleaning per-type
    subdirectories under a user-configured output root.
    """
    path = Path(path).resolve()
    allowed_parent = Path(allowed_parent).resolve()

    if path == allowed_parent:
        logger.warning(f"Refusing to remove output root: {path}")
        return

    try:
        rel = path.relative_to(allowed_parent)
    except ValueError:
        logger.warning(
            f"Refusing to remove path outside output root {allowed_parent}: {path}"
        )
        return

    # `relative_to` succeeds even for `.`; guard against empty relative path too
    if rel == Path("."):
        logger.warning(f"Refusing to remove output root: {path}")
        return

    shutil.rmtree(path, ignore_errors=True)


def validate_filepath(filepath: str) -> Path:
    """Return resolved Path if file exists. Raises DataError otherwise."""
    if not isinstance(filepath, (str, Path)):
        raise DataError(f"filepath must be str or Path, got {type(filepath).__name__}")

    path = Path(filepath)
    if not path.exists():
        raise DataError(f"File not found: {filepath}")

    return path


def validate_groups(groups: list[str]) -> None:
    """Require groups to be a non-empty list of strings."""
    if not groups or not isinstance(groups, list):
        raise DataError("groups must be non-empty list")

    if not all(isinstance(g, str) for g in groups):
        raise DataError("groups must contain strings")


def _check_arrays_consistency(
    data_dict: dict, shape: tuple, required_arrays: list
) -> None:
    for key in required_arrays:
        arr = data_dict[key]
        if not isinstance(arr, np.ndarray):
            raise DataError(f"{key} must be numpy array, got {type(arr).__name__}")
        if arr.shape != shape:
            raise DataError(f"{key} shape {arr.shape} != Y shape {shape}")


def validate_data_dict(data_dict: dict) -> None:
    """
    Validate data dictionary structure and shapes.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model data with keys:
        Y, denominators, control_idx_array, missing_idx_array, groups, units, times

    Raises
    ------
    DataError
        If required keys are missing or shapes are inconsistent
    """
    if not isinstance(data_dict, dict):
        raise DataError(f"data_dict must be dict, got {type(data_dict).__name__}")

    required_arrays = ["Y", "denominators", "control_idx_array", "missing_idx_array"]
    required_meta = ["groups", "units", "times"]

    missing = [k for k in required_arrays + required_meta if k not in data_dict]
    if missing:
        raise DataError(f"data_dict missing: {missing}")

    # Check Y is 3D
    shape = data_dict["Y"].shape
    if len(shape) != 3:
        raise DataError(f"Arrays must be 3D (K,D,N), got shape {shape}")

    _check_arrays_consistency(data_dict, shape, required_arrays)

    K, D, N = shape
    if len(data_dict["groups"]) != K:
        raise DataError(f"len(groups)={len(data_dict['groups'])} != K={K}")
    if len(data_dict["units"]) != D:
        raise DataError(f"len(units)={len(data_dict['units'])} != D={D}")
    if len(data_dict["times"]) != N:
        raise DataError(f"len(times)={len(data_dict['times'])} != N={N}")


def validate_rank(rank: Any) -> int:
    """Return `rank` as int if positive; raise DataError otherwise."""
    if not isinstance(rank, (int, np.integer)) or rank <= 0:
        raise DataError(f"rank must be positive integer, got {rank}")
    return int(rank)


def validate_samples(samples: dict[str, np.ndarray]) -> None:
    """
    Validate MCMC samples dictionary.

    Parameters
    ----------
    samples : dict
        MCMC samples from mcmc.get_samples(group_by_chain=True)
        Must contain 'mu_ctrl' key with 5D array (C, S, K, D, N)

    Raises
    ------
    DataError
        If samples are invalid
    """
    if not isinstance(samples, dict):
        raise DataError(f"samples must be dict, got {type(samples).__name__}")

    if "mu_ctrl" not in samples:
        raise DataError("samples missing 'mu_ctrl' key")

    mu_ctrl = samples["mu_ctrl"]
    # Check for array-like with ndim attribute (works for numpy and jax arrays)
    if not hasattr(mu_ctrl, "ndim") or mu_ctrl.ndim != 5:
        raise DataError(
            f"samples['mu_ctrl'] must be 5D array (C,S,K,D,N), got shape {getattr(mu_ctrl, 'shape', 'N/A')}"
        )


def validate_predictions(
    predictions: np.ndarray, samples: dict[str, Any] | None = None
) -> None:
    """
    Validate predictions array.

    Parameters
    ----------
    predictions : np.ndarray
        Posterior predictive samples, shape (C, S, K, D, N)
    samples : dict, optional
        If provided, validates predictions shape matches samples['mu_ctrl']

    Raises
    ------
    DataError
        If predictions have invalid shape
    """
    # Check for array-like with ndim attribute (works for numpy and jax arrays)
    if not hasattr(predictions, "ndim"):
        raise DataError(
            f"predictions must be array-like, got {type(predictions).__name__}"
        )

    if predictions.ndim != 5:
        raise DataError(
            f"predictions must be 5D (C,S,K,D,N), got shape {predictions.shape}"
        )

    if (
        samples is not None
        and "mu_ctrl" in samples
        and predictions.shape != samples["mu_ctrl"].shape
    ):
        raise DataError(
            f"predictions shape {predictions.shape} != "
            f"samples['mu_ctrl'] shape {samples['mu_ctrl'].shape}"
        )
