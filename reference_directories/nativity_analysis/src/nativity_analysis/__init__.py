"""
Nativity Analysis Package
========================

A professional statistical software package for analyzing birth patterns 
by nativity status using Bayesian hierarchical panel models.

Modules:
--------
- data: Data loading and preprocessing utilities
- models: Bayesian panel NMF model implementation
- inference: MCMC sampling and posterior processing
- inference: MCMC sampling and posterior processing
"""

__version__ = "0.1.0"
__author__ = "Dobbs Fertility Research Team"

from . import data
from . import models
from . import inference

__all__ = [
    "data",
    "models", 
    "inference",
]
