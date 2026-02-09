"""
Bayesian Panel NMF - Causal inference for panel data using Bayesian hierarchical models.

This package provides tools for analyzing treatment effects in panel data
using Bayesian hierarchical models with low-rank matrix factorization.
"""

__version__ = "0.1.0"

# =============================================================================
# New Simplified API (preferred)
# =============================================================================
from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.inference import run_mcmc_inference, generate_predictions
from bayesian_panel_nmf.output import format_draws
from bayesian_panel_nmf.models import model

__all__ = [
    'load_and_prepare',
    'run_mcmc_inference',
    'generate_predictions',
    'format_draws',
    'model',
]
