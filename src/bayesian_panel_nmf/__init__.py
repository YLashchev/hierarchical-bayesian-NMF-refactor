"""Bayesian hierarchical panel model with low-rank factorization for causal inference on treatment effects."""

__version__ = "0.1.0"

from bayesian_panel_nmf.logging_config import setup_logging, logger
from bayesian_panel_nmf.validation import (
    ConfigError,
    DataError,
    # Backwards compatibility aliases
    ValidationError,
    ConfigValidationError,
    DataValidationError,
    ArrayShapeError,
    validate_config,
    validate_data_dict,
    validate_filepath,
    validate_groups,
    validate_rank,
)
from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.inference import (
    run_mcmc_inference,
    generate_predictions,
    convergence_summary,
)
from bayesian_panel_nmf.output import format_draws
from bayesian_panel_nmf.parallel import (
    get_requested_analysis_workers,
    resolve_analysis_workers,
)
from bayesian_panel_nmf.models import model

# Visualization (optional — requires pip install bayesian_panel_nmf[viz])
try:
    import importlib.util

    _HAS_VIZ = importlib.util.find_spec("bayesian_panel_nmf.visualization") is not None
    if _HAS_VIZ:
        from bayesian_panel_nmf.visualization import (  # noqa: F401
            make_abs_ppc_plot,
            make_acf_ppc_plot,
            make_rmse_ppc_plot,
            make_unit_corr_ppc_plot,
            make_all_ppc_plots,
            make_raw_rate_plot,
            make_group_comparison_plot,
        )
except ImportError:
    _HAS_VIZ = False

__all__ = [
    # Logging
    "setup_logging",
    "logger",
    # Validation - New simplified exceptions
    "ConfigError",
    "DataError",
    # Validation - Backwards compatibility aliases
    "ValidationError",
    "ConfigValidationError",
    "DataValidationError",
    "ArrayShapeError",
    # Validation - Functions
    "validate_config",
    "validate_data_dict",
    "validate_filepath",
    "validate_groups",
    "validate_rank",
    # Data loading
    "load_and_prepare",
    # Inference
    "run_mcmc_inference",
    "generate_predictions",
    "convergence_summary",
    # Output
    "format_draws",
    # Parallel
    "get_requested_analysis_workers",
    "resolve_analysis_workers",
    # Models
    "model",
]

# Extend __all__ with viz names only if available
_VIZ_NAMES = [
    "make_abs_ppc_plot",
    "make_acf_ppc_plot",
    "make_rmse_ppc_plot",
    "make_unit_corr_ppc_plot",
    "make_all_ppc_plots",
    "make_raw_rate_plot",
    "make_group_comparison_plot",
]
if _HAS_VIZ:
    __all__.extend(_VIZ_NAMES)
