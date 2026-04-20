"""
Input validation for bayesian_panel_nmf.

This module provides essential validation functions with concise error messages.
Philosophy: Trust users (researchers who understand their data), let pandas/numpy
raise natural errors for type mismatches, keep validation helpful but not verbose.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


# =============================================================================
# Exceptions
# =============================================================================
class ConfigError(ValueError):
    """Configuration is invalid."""
    pass


class DataError(ValueError):
    """Data validation failed."""
    pass


# =============================================================================
# Configuration Validation
# =============================================================================
def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary structure.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with 'data' section containing 'schema'
        
    Raises
    ------
    ConfigError
        If required sections or keys are missing
    """
    if not isinstance(config, dict):
        raise ConfigError(f"config must be dict, got {type(config).__name__}")
    
    if 'data' not in config:
        raise ConfigError("config missing 'data' section")
    
    data_config = config['data']
    if not isinstance(data_config, dict):
        raise ConfigError(f"config['data'] must be dict, got {type(data_config).__name__}")
    
    schema = data_config.get('schema')
    if not schema:
        raise ConfigError("config['data'] missing 'schema' section")
    
    if not isinstance(schema, dict):
        raise ConfigError(f"config['data']['schema'] must be dict, got {type(schema).__name__}")
    
    # Check required schema keys
    required = ['unit_col', 'time_col', 'treatment_col', 'outcomes']
    missing = [k for k in required if k not in schema]
    if missing:
        raise ConfigError(f"schema missing: {missing}")
    
    # Validate outcomes list
    outcomes = schema['outcomes']
    if not isinstance(outcomes, list) or len(outcomes) == 0:
        raise ConfigError("schema['outcomes'] must be non-empty list")
    
    for i, outcome in enumerate(outcomes):
        if not isinstance(outcome, dict):
            raise ConfigError(f"outcomes[{i}] must be dict")
        if 'outcome_col' not in outcome or 'label' not in outcome:
            raise ConfigError(f"outcomes[{i}] must have 'outcome_col' and 'label'")


# =============================================================================
# Filepath Validation
# =============================================================================
def validate_filepath(filepath: str) -> Path:
    """
    Validate that filepath exists.
    
    Parameters
    ----------
    filepath : str
        Path to input file
        
    Returns
    -------
    Path
        Validated Path object
        
    Raises
    ------
    DataError
        If file doesn't exist
    """
    if not isinstance(filepath, (str, Path)):
        raise DataError(f"filepath must be str or Path, got {type(filepath).__name__}")
    
    path = Path(filepath)
    if not path.exists():
        raise DataError(f"File not found: {filepath}")
    
    return path


# =============================================================================
# Groups Validation
# =============================================================================
def validate_groups(groups: List[str]) -> None:
    """
    Validate groups is non-empty list of strings.
    
    Parameters
    ----------
    groups : list of str
        List of outcome group labels to process
        
    Raises
    ------
    DataError
        If groups is invalid
    """
    if not groups or not isinstance(groups, list):
        raise DataError("groups must be non-empty list")
    
    if not all(isinstance(g, str) for g in groups):
        raise DataError("groups must contain strings")


# =============================================================================
# Data Dictionary Validation
# =============================================================================
def validate_data_dict(data_dict: Dict[str, Any]) -> None:
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
    
    required_arrays = ['Y', 'denominators', 'control_idx_array', 'missing_idx_array']
    required_meta = ['groups', 'units', 'times']
    
    missing = [k for k in required_arrays + required_meta if k not in data_dict]
    if missing:
        raise DataError(f"data_dict missing: {missing}")
    
    # Check Y is 3D
    shape = data_dict['Y'].shape
    if len(shape) != 3:
        raise DataError(f"Arrays must be 3D (K,D,N), got shape {shape}")
    
    # Check all arrays have consistent shapes
    for key in required_arrays:
        arr = data_dict[key]
        if not isinstance(arr, np.ndarray):
            raise DataError(f"{key} must be numpy array, got {type(arr).__name__}")
        if arr.shape != shape:
            raise DataError(f"{key} shape {arr.shape} != Y shape {shape}")
    
    # Check metadata lengths match dimensions
    K, D, N = shape
    if len(data_dict['groups']) != K:
        raise DataError(f"len(groups)={len(data_dict['groups'])} != K={K}")
    if len(data_dict['units']) != D:
        raise DataError(f"len(units)={len(data_dict['units'])} != D={D}")
    if len(data_dict['times']) != N:
        raise DataError(f"len(times)={len(data_dict['times'])} != N={N}")


# =============================================================================
# Rank Validation
# =============================================================================
def validate_rank(rank: Any) -> int:
    """
    Validate rank is positive integer.
    
    Parameters
    ----------
    rank : any
        Rank value to validate
        
    Returns
    -------
    int
        Validated rank value
        
    Raises
    ------
    DataError
        If rank is not a positive integer
    """
    if not isinstance(rank, (int, np.integer)) or rank <= 0:
        raise DataError(f"rank must be positive integer, got {rank}")
    return int(rank)


# =============================================================================
# Samples and Predictions Validation (for output formatting)
# =============================================================================
def validate_samples(samples: Dict[str, np.ndarray]) -> None:
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
    
    if 'mu_ctrl' not in samples:
        raise DataError("samples missing 'mu_ctrl' key")
    
    mu_ctrl = samples['mu_ctrl']
    # Check for array-like with ndim attribute (works for numpy and jax arrays)
    if not hasattr(mu_ctrl, 'ndim') or mu_ctrl.ndim != 5:
        raise DataError(f"samples['mu_ctrl'] must be 5D array (C,S,K,D,N), got shape {getattr(mu_ctrl, 'shape', 'N/A')}")


def validate_predictions(predictions: np.ndarray, samples: Optional[Dict[str, Any]] = None) -> None:
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
    if not hasattr(predictions, 'ndim'):
        raise DataError(f"predictions must be array-like, got {type(predictions).__name__}")
    
    if predictions.ndim != 5:
        raise DataError(f"predictions must be 5D (C,S,K,D,N), got shape {predictions.shape}")
    
    if samples is not None and 'mu_ctrl' in samples:
        if predictions.shape != samples['mu_ctrl'].shape:
            raise DataError(
                f"predictions shape {predictions.shape} != "
                f"samples['mu_ctrl'] shape {samples['mu_ctrl'].shape}"
            )


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================
# Keep old exception names as aliases for backwards compatibility
ValidationError = DataError
ConfigValidationError = ConfigError
DataValidationError = DataError
ArrayShapeError = DataError
