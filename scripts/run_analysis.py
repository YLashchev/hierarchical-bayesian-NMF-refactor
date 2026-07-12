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
"""

import argparse
import json
import os
import shutil
import sys
import time
import arviz as az
import yaml  # type: ignore[import-untyped]
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.output import format_draws
from bayesian_panel_nmf.inference import (
    run_mcmc_inference,
    generate_predictions,
    convergence_summary,
)
from bayesian_panel_nmf.models import model
from bayesian_panel_nmf.validation import ConfigError, validate_config
from bayesian_panel_nmf.logging_config import setup_logging


def _validate_run_analysis_config(config: dict) -> None:
    """Validate generic schema plus sections required by run_analysis."""
    if "model" not in config:
        raise ConfigError("config missing 'model' section")
    validate_config(config)
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


def _format_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _resolve_workers(config: dict, num_chains: int, n_types: int) -> int:
    """Resolve ``parallel.analysis_workers`` against CPU count and chain count.

    ``analysis_workers``: 1 = sequential (default), -1 = auto (max safe), N = explicit.
    Caps so ``workers x num_chains`` does not oversubscribe the machine, and never
    exceeds the number of model types being run.
    """
    requested = int(config.get("parallel", {}).get("analysis_workers", 1) or 1)
    max_safe = max(1, (os.cpu_count() or 1) // max(1, num_chains))
    workers = max_safe if requested == -1 else max(1, requested)
    return max(1, min(workers, max_safe, n_types))


def _get_outcome_name(config: dict) -> str:
    """Derive outcome name for filename placeholder from config.

    Priority:
    1. Explicit ``data.outcome`` if set.
    2. Strip trailing underscore from ``outcomes_from_prefixes.outcome_prefix``.
    3. Fall back to ``"births"`` for backward compatibility.
    """
    data_cfg = config.get("data", {})
    explicit = data_cfg.get("outcome")
    if explicit:
        return str(explicit)

    schema = data_cfg.get("schema", {})
    prefixes = schema.get("outcomes_from_prefixes")
    if prefixes:
        prefix = prefixes.get("outcome_prefix", "")
        if prefix.endswith("_"):
            return prefix[:-1]
        return prefix or "births"

    return "births"


def _draws_filename(config: dict, model_type: str, rank: int) -> str:
    """Fixed scheme: {distribution}_{outcome}_{type}_{rank}."""
    dist = config.get("model", {}).get("outcome_distribution", "NB")
    return f"{dist}_{_get_outcome_name(config)}_{model_type}_{rank}"


def _clean_scoped_samples(mcmc, model_type: str, rank: int) -> dict:
    """Drop sample keys containing '/' (from numpyro.handlers.scope).

    xarray DataTree rejects '/' in variable names (path-separator conflict),
    so both the convergence gate and the trace sidecar need this filter
    before handing samples to az.from_dict.
    """
    raw_samples = mcmc.get_samples(group_by_chain=True)
    clean_samples = {k: v for k, v in raw_samples.items() if "/" not in k}
    if len(clean_samples) < len(raw_samples):
        dropped = set(raw_samples) - set(clean_samples)
        logger.debug(
            f"{model_type} rank {rank}: excluded {len(dropped)} scoped "
            f"sample keys: {sorted(dropped)}"
        )
    return clean_samples


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
        print_target_table=output_config.get("print_target_table", True),
        aggregate_units=output_config.get("aggregate_units"),
        ppc_units=output_config.get("ppc_units"),
        ppc_acf_lags=output_config.get("ppc_acf_lags", [6]),
        ppc_unit_corr_max_time=output_config.get("ppc_unit_corr_max_time"),
        ppc_exclude_units=output_config.get("ppc_exclude_units"),
    )


def run_model_type(
    model_type: str,
    type_config: dict,
    config: dict,
    rank_override: int | None = None,
    save_traces: bool = False,
    log_level: str = "INFO",
    configure_logging: bool = True,
    disable_progress_bar: bool = False,
) -> None:
    """Run analysis for a single model type across specified ranks."""
    if configure_logging:
        setup_logging(level=log_level)

    if disable_progress_bar:
        # Mutate a shallow copy so parallel workers don't race on shared state.
        config = {
            **config,
            "mcmc": {**config.get("mcmc", {}), "progress_bar": False},
        }
        # Parallel worker: this process's own logger only, filtered to WARNING
        # so per-model stage logs don't interleave across concurrent workers.
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

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

    model_started_at = time.monotonic()
    load_started_at = time.monotonic()
    data_dict = load_and_prepare(
        filepath=config["data"]["input_file"],
        config=config,
        groups=groups,
        exclude_units=exclude_units,
        type_config=type_config,
    )

    logger.info(
        f"{model_type}: data ready in "
        f"{_format_elapsed(time.monotonic() - load_started_at)} "
        f"(K={len(data_dict['groups'])}, D={len(data_dict['units'])}, "
        f"N={len(data_dict['times'])})"
    )

    df_preprocessed = data_dict["df_preprocessed"]
    preprocessed_file = type_output_dir / f"df_{model_type}.csv"
    df_preprocessed.to_csv(preprocessed_file, index=False)

    for rank in ranks:
        filename = _draws_filename(config, model_type, rank)

        mcmc_started_at = time.monotonic()
        mcmc = run_mcmc_inference(data_dict, model, rank, config)
        logger.info(
            f"{model_type} rank {rank}: MCMC finished in "
            f"{_format_elapsed(time.monotonic() - mcmc_started_at)}"
        )

        clean_samples = _clean_scoped_samples(mcmc, model_type, rank)
        extra_fields = mcmc.get_extra_fields()
        idata_dict = {"posterior": clean_samples}
        if "diverging" in extra_fields:
            # get_extra_fields() is flat (chains*samples,); reshape to match the
            # (chain, draw) samples returned by get_samples(group_by_chain=True).
            diverging = extra_fields["diverging"].reshape(mcmc.num_chains, -1)
            idata_dict["sample_stats"] = {"diverging": diverging}
        idata = az.from_dict(idata_dict)
        gate = convergence_summary(idata)
        if not gate["converged"]:
            logger.warning(
                f"{model_type} rank {rank}: convergence gate FAILED — "
                f"max R-hat={gate['rhat_max']:.4f}, min bulk ESS={gate['ess_bulk_min']:.0f}, "
                f"min tail ESS={gate['ess_tail_min']:.0f}, divergences={gate['divergences']}"
            )
        convergence_file = type_output_dir / f"{filename}_convergence.json"
        with open(convergence_file, "w") as f:
            json.dump(gate, f, indent=2)

        if save_traces:
            traces_file = type_output_dir / f"{filename}_traces.nc"
            trace_clean_samples = _clean_scoped_samples(mcmc, model_type, rank)
            trace_idata = az.from_dict({"posterior": trace_clean_samples})
            trace_idata.to_netcdf(str(traces_file), engine="h5netcdf")
            size_mb = traces_file.stat().st_size / 1024**2
            logger.info(
                f"{model_type} rank {rank}: wrote trace sidecar to {traces_file} "
                f"({size_mb:.1f} MB)"
            )

        predictions = generate_predictions(mcmc, data_dict, model, rank, config)

        samples = mcmc.get_samples(group_by_chain=True)
        draws_df = format_draws(samples, predictions, data_dict)

        draws_file = type_output_dir / f"{filename}.csv"
        draws_df.to_csv(draws_file, index=False)
        size_mb = draws_file.stat().st_size / 1024**2
        logger.info(f"{model_type} rank {rank}: wrote draws to {draws_file} ({size_mb:.1f} MB)")

        # Multi-rank runs nest under rank_<rank>/ so figs don't collide
        report_dir = (
            type_output_dir / f"rank_{rank}" if len(ranks) > 1 else type_output_dir
        )
        report_dir.mkdir(parents=True, exist_ok=True)

        if output_config.get("figures", False):
            _run_reporting(draws_df, report_dir, output_config)

    logger.info(
        f"{model_type}: complete in {_format_elapsed(time.monotonic() - model_started_at)}"
    )


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
    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, log_file=args.log_file)

    # Load and validate configuration
    config = load_config(args.config)
    _validate_run_analysis_config(config)

    # Setup output directory
    output_dir = Path(config["data"]["output_dir"])
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {output_dir}: {e}")
        raise

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

    # Resolve save_traces: CLI flag overrides config
    save_traces = args.save_traces or config.get("output", {}).get("save_traces", False)

    # Resolve analysis-level parallelism against num_chains and CPU count
    num_chains = int(config.get("mcmc", {}).get("num_chains", 4))
    workers = _resolve_workers(config, num_chains=num_chains, n_types=len(types_to_run))
    logger.info(
        f"Running {len(types_to_run)} model type(s) with {workers} worker(s) "
        f"(num_chains={num_chains})"
    )

    # Sequential path (preserves rich logging for single-type or workers=1)
    if workers == 1:
        total_count = len(types_to_run)
        for index, (model_type, type_config) in enumerate(
            types_to_run.items(), start=1
        ):
            logger.info(
                f"RUNNING MODEL TYPE: {model_type.upper()} ({index}/{total_count})"
            )
            run_model_type(
                model_type=model_type,
                type_config=type_config,
                config=config,
                rank_override=args.rank,
                save_traces=save_traces,
                log_level=log_level,
                configure_logging=False,
            )
    else:
        # Parallel path: each worker reconfigures its own logger + progress bars
        # disabled to keep stdout readable when N>1 subprocesses print concurrently.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_model_type,
                    model_type=model_type,
                    type_config=type_config,
                    config=config,
                    rank_override=args.rank,
                    save_traces=save_traces,
                    log_level=log_level,
                    configure_logging=True,
                    disable_progress_bar=True,
                ): model_type
                for model_type, type_config in types_to_run.items()
            }
            for i, fut in enumerate(as_completed(futures), 1):
                mt = futures[fut]
                fut.result()  # re-raises worker exceptions
                logger.info(f"{mt} finished ({i}/{len(futures)})")

    logger.info(f"Analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
