"""
Bayesian hierarchical panel model for nativity analysis.

This module contains the NumPyro implementation of the panel NMF model
adapted for nativity subgroup analysis.
"""

from .panel_nmf_model import model
from .utils import missingness_adjustment

__all__ = [
    "model",
    "missingness_adjustment",
]
