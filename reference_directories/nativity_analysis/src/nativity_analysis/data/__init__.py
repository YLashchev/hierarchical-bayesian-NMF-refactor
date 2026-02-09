"""
Data processing module for nativity analysis.

This module handles loading, preprocessing, and transforming birth data
by nativity status.
"""

from .loader import load_nativity_data
from .preprocessing import (
    wide_to_long,
    aggregate_to_bimonthly,
    prepare_model_data,
    create_exposure_codes,
    filter_time_period
)

__all__ = [
    "load_nativity_data",
    "wide_to_long",
    "aggregate_to_bimonthly", 
    "prepare_model_data",
    "create_exposure_codes",
    "filter_time_period",
]
