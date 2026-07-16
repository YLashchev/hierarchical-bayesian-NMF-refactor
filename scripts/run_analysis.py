"""
Main script for running the Bayesian Panel NMF analysis pipeline.

This script:
1. Loads and preprocesses panel data using schema-based configuration
2. Runs Bayesian inference for specified model types and ranks
3. Saves results in tidy format with standardized column names
4. Always computes and saves an ArviZ-based convergence gate (ESS, R-hat,
   divergences) next to the draws CSV

Usage:
    python scripts/run_analysis.py --config configs/nativity_config.yaml
    python scripts/run_analysis.py --config configs/nativity_config.yaml --type groups --rank 10
    python scripts/run_analysis.py --config configs/nativity_config.yaml --verbose
    python scripts/run_analysis.py --config configs/nativity_config.yaml --log-file logs/analysis.log
    python scripts/run_analysis.py --config configs/nativity_config.yaml --chains 4 --chain-method sequential
"""

import argparse
import os
from pathlib import Path

# numpyro.set_host_device_count() only takes effect before JAX's backend is
# lazily initialized (it sets an XLA_FLAGS env var that XLA reads once, at
# first use). It MUST run before any jax/numpyro/arviz import — including
# the imports below and inside bayesian_panel_nmf.inference/pipeline — or it
# is a silent no-op and NUTS's MCMC(..., chain_method="parallel") falls back
# to sequential chain execution (root cause of the mcmc.num_chains chains not
# actually running concurrently on multi-core CPUs).
import numpyro  # noqa: E402

numpyro.set_host_device_count(os.cpu_count() or 1)

from loguru import logger  # noqa: E402

from bayesian_panel_nmf.config import Config  # noqa: E402
from bayesian_panel_nmf.logging_config import setup_logging  # noqa: E402
from bayesian_panel_nmf.pipeline import (  # noqa: E402
    _run_sequential,
    _select_types_to_run,
    _validate_run_analysis_config,
)


def load_config(config_path: str) -> Config:
    """Load and validate configuration from a YAML file."""
    return Config.from_yaml(config_path)


def _apply_mcmc_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Return ``config`` with --chains / --chain-method overrides applied to
    its ``mcmc`` section.

    --chain-method forces auto_parallelism=false (manual mode) and sets
    chain_method. --chains then sets num_chains (manual) or max_chains (auto).
    Without --chain-method, --chains just overrides max_chains under auto.

    ``Config`` is immutable, so overrides are applied by dumping to a dict,
    mutating the ``mcmc`` sub-dict, and re-validating.
    """
    if args.chains is None and args.chain_method is None:
        return config

    data = config.model_dump()
    mcmc = data["mcmc"]
    if args.chain_method is not None:
        mcmc["auto_parallelism"] = False
        mcmc["chain_method"] = args.chain_method
        if args.chains is not None:
            mcmc["num_chains"] = args.chains
        logger.info(
            f"CLI override: auto_parallelism=false, "
            f"num_chains={mcmc.get('num_chains', 4)}, "
            f"chain_method={args.chain_method!r}"
        )
    else:
        mcmc["max_chains"] = args.chains
        logger.info(
            f"CLI override: max_chains={args.chains} (auto_parallelism stays on)"
        )
    return Config.model_validate(data)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse run_analysis.py's CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Bayesian Panel NMF analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nativity_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="Model type to run; if not specified, runs all",
    )
    parser.add_argument(
        "--rank", type=int, default=None, help="Model rank (overrides config)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG level) logging",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (enables file logging)",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save full posterior draws as NetCDF sidecar (arviz InferenceData)",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=None,
        help=(
            "Override MCMC chain count. With --chain-method, sets the literal "
            "num_chains; without it, overrides mcmc.max_chains under auto_parallelism."
        ),
    )
    parser.add_argument(
        "--chain-method",
        type=str,
        choices=["sequential", "parallel", "vectorized"],
        default=None,
        help=(
            "Force chain_method (disables auto_parallelism). Use for timing "
            "comparisons, e.g. --chains 4 --chain-method sequential."
        ),
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def main():
    args = _parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, log_file=args.log_file)

    # Load and validate configuration
    config = load_config(args.config)
    _validate_run_analysis_config(config)

    # Setup output directory
    output_dir = Path(config.data.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {output_dir}: {e}")
        raise

    # Apply CLI overrides to the mcmc config section.
    config = _apply_mcmc_overrides(config, args)

    # Select which model types to run
    types_to_run = _select_types_to_run(config.model.types, args.type)

    # Resolve save_traces: CLI flag overrides config
    save_traces = args.save_traces or config.output.save_traces

    logger.info(f"Running {len(types_to_run)} model type(s)")

    _run_sequential(types_to_run, config, args.rank, save_traces, log_level)

    logger.info(f"Analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
