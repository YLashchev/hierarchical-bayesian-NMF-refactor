"""Package-wide error types.

These two exceptions are imported almost everywhere, so they live in a
dependency-free leaf module to avoid import cycles. Config validation is
``Config.model_validate`` (config.py); runtime array/data validators live in
``checks.py``.
"""


class ConfigError(ValueError):
    """Configuration is invalid."""


class DataError(ValueError):
    """Data validation failed."""


__all__ = [
    "ConfigError",
    "DataError",
]
