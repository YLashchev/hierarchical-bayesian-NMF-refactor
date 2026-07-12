"""Logging setup for bayesian_panel_nmf."""

import sys

from loguru import logger


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper())
    if log_file:
        logger.add(log_file, level="DEBUG")


__all__ = ["setup_logging", "logger"]
