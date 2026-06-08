"""
Main script for running the Bayesian Panel NMF analysis pipeline.

This script:
1. Loads and preprocesses panel data using schema-based configuration
2. Runs Bayesian inference for specified model types and ranks
3. Extracts and saves MCMC diagnostics (ESS, R-hat, divergences)
4. Saves results in tidy format with standardized column names

Usage:
    python scripts/run_analysis.py --config configs/nativity_config.yaml
    python scripts/run_analysis.py --config configs/nativity_config.yaml --type groups --rank 10
    python scripts/run_analysis.py --config configs/nativity_config.yaml --verbose
    python scripts/run_analysis.py --config configs/nativity_config.yaml --log-file logs/analysis.log
    python scripts/run_analysis.py --config configs/nativity_config.yaml --save-diagnostics
"""

import argparse
import json
import shutil
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.output import format_draws
from bayesian_panel_nmf.inference import (
    run_mcmc_inference,
    run_two_stage_inference,
    generate_predictions,
    extract_diagnostics,
    check_convergence,
)
from bayesian_panel_nmf.models import model
from bayesian_panel_nmf.validation import ConfigError
from bayesian_panel_nmf.logging_config import setup_logging
from bayesian_panel_nmf.parallel import (
    get_requested_analysis_workers,
    resolve_analysis_workers,
)


def _validate_run_analysis_config(config: dict) -> None:
    """Validate that config has required sections for run_analysis."""
    if "model" not in config:
        raise ConfigError("config missing 'model' section")
    if "types" not in config.get("model", {}):
        raise ConfigError("config['model'] missing 'types' section")


def _safe_rmtree(path: Path, allowed_parent: Path) -> None:
    """Remove a directory tree only if strictly inside ``allowed_parent``.

    Refuses to remove ``allowed_parent`` itself, its ancestors, or any path
    that is not a descendant of it. Intended for cleaning per-type
    subdirectories under a user-configured output root.
    """
    path = Path(path).resolve()
    allowed_parent = Path(allowed_parent).resolve()

    if path == allowed_parent:
        logger.warning(f"Refusing to remove output root: {path}")
        return

    try:
        rel = path.relative_to(allowed_parent)
    except ValueError:
        logger.warning(
            f"Refusing to remove path outside output root {allowed_parent}: {path}"
        )
        return

    # `relative_to` succeeds even for `.`; guard against empty relative path too
    if rel == Path("."):
        logger.warning(f"Refusing to remove output root: {path}")
        return

    shutil.rmtree(path, ignore_errors=True)


def _run_reporting(
    draws_df,
    output_dir,
    output_config: dict,
) -> None:
    """Generate figures + tables under ``<output_dir>/figs/``."""
    from bayesian_panel_nmf.reporting import generate_reports

    generate_reports(
        draws_df,
        output_dir=output_dir,
        target_unit=output_config.get("target_unit"),
        groups=output_config.get("report_groups"),
        print_tables=output_config.get("print_tables", True),
    )


def run_model_type(
    model_type: str,
    type_config: dict,
    config: dict,
    rank_override: int | None = None,
    save_diagnostics: bool = False,
    log_level: str = "INFO",
    configure_logging: bool = True,
    disable_progress_bar: bool = False,
) -> None:
    """Run analysis for a single model type across specified ranks."""
    if configure_logging:
        setup_logging(level=log_level)

    if disable_progress_bar:
        # Mutate a shallow copy so parallel workers don't race on shared state
        config = {**config, "mcmc": {**config.get("mcmc", {}), "progress_bar": False}}

    base_output_dir = Path(config["data"]["output_dir"])
    output_config = config.get("output", {})
    type_output_dir = base_output_dir / model_type

    # Optional cleanup of this type's subtree before writing
    if output_config.get("clean", False) and type_output_dir.exists():
        logger.info(f"clean=true: removing existing {type_output_dir}")
        _safe_rmtree(type_output_dir, base_output_dir)

    type_output_dir.mkdir(parents=True, exist_ok=True)

    groups = type_config["groups"]
    ranks = [rank_override] if rank_override else type_config.get("ranks_to_test", [10])
    exclude_units = type_config.get("exclude_units")

    # Load and prepare data
    data_dict = load_and_prepare(
        filepath=config["data"]["input_file"],
        config=config,
        groups=groups,
        exclude_units=exclude_units,
        type_config=type_config,
    )

    df_preprocessed = data_dict["df_preprocessed"]
    df_preprocessed.to_csv(type_output_dir / f"df_{model_type}.csv", index=False)

    treatment_mode = config.get("model", {}).get("treatment_mode", "two_stage")

    for rank in ranks:
        if treatment_mode == "two_stage":
            # Stage 1 (control-only factors) + Stage 2 (per-imputation treatment
            # effect) with the counterfactual fixed as an offset; cuts feedback.
            samples, predictions, diagnostics = run_two_stage_inference(
                data_dict, model, rank, config
            )
        else:
            mcmc = run_mcmc_inference(data_dict, model, rank, config)
            diagnostics = extract_diagnostics(mcmc)
            check_convergence(diagnostics)
            predictions = generate_predictions(mcmc, data_dict, model, rank, config)
            samples = mcmc.get_samples(group_by_chain=True)

        if save_diagnostics:
            dist = config["model"].get("outcome_distribution", "NB")
            pattern = output_config.get(
                "filename_pattern", "{distribution}_{type}_{rank}"
            )
            diag_filename = pattern.format(
                distribution=dist, outcome="births", type=model_type, rank=rank
            )
            diagnostics_file = type_output_dir / f"{diag_filename}_diagnostics.json"
            diag_output = {
                "n_eff_min": diagnostics["n_eff_min"],
                "n_eff_mean": diagnostics["n_eff_mean"],
                "rhat_max": diagnostics["rhat_max"],
                "rhat_mean": diagnostics["rhat_mean"],
                "divergences": diagnostics["divergences"],
                "converged": diagnostics["converged"],
            }
            with open(diagnostics_file, "w") as f:
                json.dump(diag_output, f, indent=2)

        draws_df = format_draws(samples, predictions, data_dict)

        dist = config["model"].get("outcome_distribution", "NB")
        pattern = output_config.get("filename_pattern", "{distribution}_{type}_{rank}")
        filename = pattern.format(
            distribution=dist, outcome="births", type=model_type, rank=rank
        )
        draws_df.to_csv(type_output_dir / f"{filename}.csv", index=False)

        # Multi-rank runs nest under rank_<rank>/ so figs don't collide
        report_dir = (
            type_output_dir / f"rank_{rank}" if len(ranks) > 1 else type_output_dir
        )
        report_dir.mkdir(parents=True, exist_ok=True)

        if output_config.get("figures", False):
            _run_reporting(draws_df, report_dir, output_config)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
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
        "--no-aggregate", action="store_true", help="Skip temporal aggregation"
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
        "--save-diagnostics",
        action="store_true",
        help="Save MCMC diagnostics to JSON file",
    )
    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, log_file=args.log_file)

    # Load configuration
    config = load_config(args.config)
    logger.info("=" * 60)
    logger.info("BAYESIAN PANEL NMF ANALYSIS")
    logger.info("=" * 60)
    logger.debug(f"Config file: {args.config}")
    logger.debug(f"Log level: {log_level}")

    # Handle --no-aggregate flag by modifying config
    if args.no_aggregate:
        config = config.copy()
        config["data"] = config["data"].copy()
        config["data"]["aggregation"] = {"enabled": False}
        logger.info("Temporal aggregation disabled via --no-aggregate flag")

    # Setup output directory
    output_dir = Path(config["data"]["output_dir"])
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {output_dir}: {e}")
        raise
    logger.debug(f"Output directory: {output_dir}")

    # Select which model types to run
    all_types = config["model"]["types"]
    if args.type:
        if args.type not in all_types:
            raise ConfigError(
                f"--type={args.type!r} not found in config; available: {list(all_types)}"
            )
        types_to_run = {args.type: all_types[args.type]}
    else:
        types_to_run = all_types

    # Resolve analysis-level parallelism against num_chains and CPU count
    num_chains = int(config.get("mcmc", {}).get("num_chains", 4))
    requested_workers = get_requested_analysis_workers(config)
    workers = resolve_analysis_workers(
        requested_workers,
        num_chains=num_chains,
        num_model_types=len(types_to_run),
    )
    logger.info(
        f"Running {len(types_to_run)} model type(s) with {workers} worker(s) "
        f"(requested={requested_workers}, num_chains={num_chains})"
    )

    # Sequential path (preserves rich logging for single-type or workers=1)
    if workers == 1:
        for model_type, type_config in types_to_run.items():
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"RUNNING MODEL TYPE: {model_type.upper()}")
            logger.info("=" * 60)
            run_model_type(
                model_type=model_type,
                type_config=type_config,
                config=config,
                rank_override=args.rank,
                save_diagnostics=args.save_diagnostics,
                log_level=log_level,
                configure_logging=False,
            )
    else:
        # Parallel path: each worker reconfigures its own logger + progress bars disabled
        # to keep stdout readable when N>1 subprocesses print concurrently.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_model_type,
                    model_type=model_type,
                    type_config=type_config,
                    config=config,
                    rank_override=args.rank,
                    save_diagnostics=args.save_diagnostics,
                    log_level=log_level,
                    configure_logging=True,
                    disable_progress_bar=True,
                ): model_type
                for model_type, type_config in types_to_run.items()
            }
            for fut in as_completed(futures):
                mt = futures[fut]
                try:
                    fut.result()
                    logger.info(f"✓ {mt} finished")
                except Exception as e:
                    logger.error(f"✗ {mt} failed: {e}")
                    raise

    logger.info("")
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
