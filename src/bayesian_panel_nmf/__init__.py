"""
Bayesian Panel NMF - Causal inference for panel data using Bayesian hierarchical models.

This package provides tools for analyzing treatment effects in panel data
using Bayesian hierarchical models with low-rank matrix factorization.
"""

__version__ = "0.1.0"

from .data import load_panel_data, wide_to_long, preprocess_pipeline, DataSchema
from .models import model
from .inference import run_mcmc_inference, generate_predictions, merge_draws_and_data

__all__ = [
    'load_panel_data',
    'wide_to_long',
    'preprocess_pipeline',
    'DataSchema',
    'model',
    'run_mcmc_inference',
    'generate_predictions',
    'merge_draws_and_data',
]
