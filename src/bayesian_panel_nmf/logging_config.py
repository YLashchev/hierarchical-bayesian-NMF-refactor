"""
Logging configuration for bayesian_panel_nmf.

Usage:
    from bayesian_panel_nmf import setup_logging
    setup_logging()  # INFO level to console
    setup_logging(level="DEBUG")  # Verbose
    setup_logging(log_file="analysis.log")  # Also log to file
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Track handler IDs we own so we only remove ours on re-call.
_state: dict[str, list[int]] = {"owned_handler_ids": []}


def _remove_default_handler() -> None:
    """Remove Loguru's built-in stderr handler if still present."""
    try:
        logger.remove(0)
    except ValueError:
        pass


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for bayesian_panel_nmf.

    Only removes handlers previously added by this function. Does not
    touch handlers created by the user or other libraries.

    Parameters
    ----------
    level : str, default="INFO"
        Logging level: DEBUG, INFO, WARNING, ERROR
    log_file : str, optional
        Path to log file. If provided, logs are also written to file.

    Examples
    --------
    >>> setup_logging()  # Default INFO to console
    >>> setup_logging(level="DEBUG")  # Verbose output
    >>> setup_logging(log_file="logs/run.log")  # Also save to file
    """
    _remove_default_handler()

    # Remove only handlers we previously added
    for hid in _state["owned_handler_ids"]:
        try:
            logger.remove(hid)
        except ValueError:
            pass  # already removed
    _state["owned_handler_ids"].clear()

    # Console handler with color
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    hid = logger.add(
        sys.stderr, format=console_format, level=level.upper(), colorize=True
    )
    _state["owned_handler_ids"].append(hid)

    # File handler if requested
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        )
        hid = logger.add(
            log_file,
            format=file_format,
            level="DEBUG",
            rotation="100 MB",
            retention="10 days",
        )
        _state["owned_handler_ids"].append(hid)


# Re-export logger for direct use
__all__ = ["setup_logging", "logger"]
