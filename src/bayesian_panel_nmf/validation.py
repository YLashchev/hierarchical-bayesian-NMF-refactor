"""Config and data validation. Raises ConfigError or DataError with concise messages."""


class ConfigError(ValueError):
    """Configuration is invalid."""


class DataError(ValueError):
    """Data validation failed."""


def validate_config(config: dict) -> None:
    """Back-compat shim: validate a raw config dict via the pydantic schema.

    Raises ConfigError on any problem (unknown keys, wrong types, bad XOR).
    """
    from bayesian_panel_nmf.config import Config

    Config.model_validate(config)


# Re-exported for back-compat: these moved to checks.py in Phase 3 (Task 3.3).
from bayesian_panel_nmf.checks import (  # noqa: E402
    validate_data_dict,
    validate_filepath,
    validate_groups,
    validate_predictions,
    validate_rank,
    validate_samples,
)

__all__ = [
    "ConfigError",
    "DataError",
    "validate_config",
    "validate_data_dict",
    "validate_filepath",
    "validate_groups",
    "validate_predictions",
    "validate_rank",
    "validate_samples",
]
