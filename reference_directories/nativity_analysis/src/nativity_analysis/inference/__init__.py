"""
Inference module for Bayesian model fitting.

This module handles MCMC sampling, posterior processing, and 
conversion to tidy data formats.
"""

from .sampler import run_mcmc_inference, generate_predictions
from .postprocessing import (
    dict_to_tidybayes,
    merge_draws_and_data,
)

__all__ = [
    "run_mcmc_inference",
    "generate_predictions",
    "dict_to_tidybayes",
    "merge_draws_and_data",
]
