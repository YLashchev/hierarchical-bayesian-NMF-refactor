"""Bayesian hierarchical panel model with low-rank factorization for causal inference on treatment effects."""

__version__ = "0.1.0"

from bayesian_panel_nmf.config import Config
from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.diagnostics import convergence_summary
from bayesian_panel_nmf.inference import (
    generate_predictions,
    run_mcmc_inference,
)
from bayesian_panel_nmf.logging_config import logger, setup_logging
from bayesian_panel_nmf.models import model
from bayesian_panel_nmf.plots import (
    make_all_ppc_plots,
    make_group_comparison_plot,
    make_raw_rate_plot,
)
from bayesian_panel_nmf.results import format_draws
from bayesian_panel_nmf.validation import (
    ConfigError,
    DataError,
    validate_config,
    validate_data_dict,
)

__all__ = [
    # Logging
    "setup_logging",
    "logger",
    # Validation - Exceptions
    "ConfigError",
    "DataError",
    # Config
    "Config",
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
    # Visualization
    "make_all_ppc_plots",
    "make_raw_rate_plot",
    "make_group_comparison_plot",
]
