"""
Bayesian Panel NMF - Causal inference for panel data using Bayesian hierarchical models.

This package provides tools for analyzing treatment effects in panel data
using Bayesian hierarchical models with low-rank matrix factorization.
"""

__version__ = "0.1.0"

# =============================================================================
# Logging Configuration (call first in scripts)
# =============================================================================
from bayesian_panel_nmf.logging_config import setup_logging, logger

# =============================================================================
# Validation (for advanced users)
# =============================================================================
from bayesian_panel_nmf.validation import (
    # New simplified exceptions (2 classes)
    ConfigError,
    DataError,
    # Backwards compatibility aliases
    ValidationError,
    ConfigValidationError,
    DataValidationError,
    ArrayShapeError,
    # Validation functions
    validate_config,
    validate_data_dict,
    validate_filepath,
    validate_groups,
    validate_rank,
)

# =============================================================================
# New Simplified API (preferred)
# =============================================================================
from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.inference import (
    run_mcmc_inference,
    generate_predictions,
    extract_diagnostics,
    check_convergence,
)
from bayesian_panel_nmf.output import format_draws
from bayesian_panel_nmf.models import model

# Visualization (PPC plots and time series)
from bayesian_panel_nmf.visualization import (
    make_abs_ppc_plot,
    make_acf_ppc_plot,
    make_rmse_ppc_plot,
    make_unit_corr_ppc_plot,
    make_all_ppc_plots,
    # Time series visualization
    make_raw_rate_plot,
    make_group_comparison_plot,
)

__all__ = [
    # Logging
    'setup_logging',
    'logger',
    # Validation - New simplified exceptions
    'ConfigError',
    'DataError',
    # Validation - Backwards compatibility aliases
    'ValidationError',
    'ConfigValidationError',
    'DataValidationError',
    'ArrayShapeError',
    # Validation - Functions
    'validate_config',
    'validate_data_dict',
    'validate_filepath',
    'validate_groups',
    'validate_rank',
    # Data loading
    'load_and_prepare',
    # Inference
    'run_mcmc_inference',
    'generate_predictions',
    'extract_diagnostics',
    'check_convergence',
    # Output
    'format_draws',
    # Models
    'model',
    # Visualization (PPC)
    'make_abs_ppc_plot',
    'make_acf_ppc_plot',
    'make_rmse_ppc_plot',
    'make_unit_corr_ppc_plot',
    'make_all_ppc_plots',
    # Visualization (Time Series)
    'make_raw_rate_plot',
    'make_group_comparison_plot',
]
