"""
Utilities for analysis-level parallelism.

Provides helpers to read the ``parallel.analysis_workers`` config knob and
resolve it against hardware constraints so that
``workers × chains`` does not oversubscribe the machine.
"""

import os
from typing import Optional

from loguru import logger


def get_requested_analysis_workers(config: dict) -> int:
    """Read analysis worker count from config, supporting the legacy alias.

    Parameters
    ----------
    config : dict
        Full analysis config dict.

    Returns
    -------
    int
        Requested worker count (1 = sequential, -1 = auto, N = explicit).

    Raises
    ------
    ValueError
        If the value is 0 or < -1.
    """
    parallel_config = config.get("parallel", {})
    analysis_workers = parallel_config.get("analysis_workers")

    if analysis_workers is None and parallel_config.get("num_workers") is not None:
        logger.warning(
            "parallel.num_workers is deprecated; use parallel.analysis_workers instead"
        )
        analysis_workers = parallel_config["num_workers"]

    if analysis_workers is None:
        return 1

    analysis_workers = int(analysis_workers)
    if analysis_workers == 0 or analysis_workers < -1:
        raise ValueError("parallel.analysis_workers must be a positive integer or -1")

    return analysis_workers


def resolve_analysis_workers(
    requested_workers: int,
    num_chains: int,
    num_model_types: int,
    cpu_count: Optional[int] = None,
) -> int:
    """Cap analysis workers so workers × chains does not oversubscribe the machine.

    Parameters
    ----------
    requested_workers : int
        Value returned by :func:`get_requested_analysis_workers`.
    num_chains : int
        Number of MCMC chains per model fit.
    num_model_types : int
        Number of model types to run.
    cpu_count : int, optional
        Override for ``os.cpu_count()`` (useful in tests).

    Returns
    -------
    int
        Safe number of analysis workers (>= 1).
    """
    available_cores = cpu_count or os.cpu_count() or 1
    max_workers_for_chains = max(1, available_cores // max(1, num_chains))

    if requested_workers == -1:
        requested_workers = max_workers_for_chains

    return max(1, min(requested_workers, max_workers_for_chains, num_model_types))
