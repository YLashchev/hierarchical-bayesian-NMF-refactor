"""Bayesian hierarchical panel model with low-rank factorization for causal inference on treatment effects."""

__version__ = "0.1.0"

from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.inference import (
    convergence_summary,
    generate_predictions,
    run_mcmc_inference,
)
from bayesian_panel_nmf.logging_config import logger, setup_logging
from bayesian_panel_nmf.models import model
from bayesian_panel_nmf.output import format_draws
from bayesian_panel_nmf.validation import (
    ConfigError,
    DataError,
    validate_config,
    validate_data_dict,
)

# Visualization (optional — requires pip install bayesian_panel_nmf[viz])
try:
    import importlib.util

    _HAS_VIZ = importlib.util.find_spec("bayesian_panel_nmf.visualization") is not None
    if _HAS_VIZ:
        from bayesian_panel_nmf.visualization import (  # noqa: F401
            make_all_ppc_plots,
            make_group_comparison_plot,
            make_raw_rate_plot,
        )
except ImportError:
    _HAS_VIZ = False

__all__ = [
    # Logging
    "setup_logging",
    "logger",
    # Validation - Exceptions
    "ConfigError",
    "DataError",
    # Validation - Functions
    "validate_config",
    "validate_data_dict",
    # Data loading
    "load_and_prepare",
    # Inference
    "run_mcmc_inference",
    "generate_predictions",
    "convergence_summary",
    # Output
    "format_draws",
    # Models
    "model",
]

# Extend __all__ with viz names only if available
_VIZ_NAMES = [
    "make_all_ppc_plots",
    "make_raw_rate_plot",
    "make_group_comparison_plot",
]
if _HAS_VIZ:
    __all__.extend(_VIZ_NAMES)
