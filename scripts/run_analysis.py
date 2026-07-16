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
import json
import os
import shutil
import time
from pathlib import Path

# numpyro.set_host_device_count() only takes effect before JAX's backend is
# lazily initialized (it sets an XLA_FLAGS env var that XLA reads once, at
# first use). It MUST run before any jax/numpyro/arviz import — including
# the imports below and inside bayesian_panel_nmf.inference — or it is a
# silent no-op and NUTS's MCMC(..., chain_method="parallel") falls back to
# sequential chain execution (root cause of the mcmc.num_chains chains not
# actually running concurrently on multi-core CPUs).
import numpyro  # noqa: E402

numpyro.set_host_device_count(os.cpu_count() or 1)

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from bayesian_panel_nmf.config import Config, OutputConfig, TypeConfig  # noqa: E402
from bayesian_panel_nmf.data import load_and_prepare  # noqa: E402
from bayesian_panel_nmf.inference import (  # noqa: E402
    convergence_summary,
    generate_predictions,
    run_mcmc_inference,
)
from bayesian_panel_nmf.logging_config import setup_logging  # noqa: E402
from bayesian_panel_nmf.models import model  # noqa: E402
from bayesian_panel_nmf.results import format_draws  # noqa: E402
from bayesian_panel_nmf.validation import ConfigError  # noqa: E402


def _write_draws(df: pd.DataFrame, stem: Path, fmt: str) -> Path:
    """Write the (large) draws artifact as csv or parquet; return the path written.

    Additive, opt-in via ``output.draws_format`` (default ``"csv"``). Only
    ever applied to the two big draws artifacts, never the small human-facing
    summary/table CSVs.
    """
    if fmt == "parquet":
        path = stem.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    else:
        path = stem.with_suffix(".csv")
        df.to_csv(path, index=False)
    return path


def _read_draws(stem: Path) -> pd.DataFrame:
    """Read a draws artifact written by ``_write_draws``, trying .parquet then .csv."""
    parquet_path = stem.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = stem.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"no draws file found at {stem} (.parquet or .csv)")


def _validate_run_analysis_config(config: Config) -> None:
    """Run-analysis-specific check beyond generic schema validation (already
    enforced by ``Config.from_yaml``/``Config.model_validate``): at least one
    model type must be configured."""
    if not config.model.types:
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


def _get_outcome_name(config: Config) -> str:
    """Derive outcome name for filename placeholder from config.

    Priority:
    1. Explicit ``data.outcome`` if set.
    2. Strip trailing underscore from ``outcomes_from_prefixes.outcome_prefix``.
    3. Fall back to ``"births"`` for backward compatibility.
    """
    explicit = config.data.outcome
    if explicit:
        return str(explicit)

    prefixes = config.data.schema_.outcomes_from_prefixes
    if prefixes:
        prefix = prefixes.outcome_prefix or ""
        if prefix.endswith("_"):
            return prefix[:-1]
        return prefix or "births"

    return "births"


def _draws_filename(config: Config, model_type: str, rank: int) -> str:
    """Fixed scheme: {distribution}_{outcome}_{type}_{rank}."""
    dist = config.model.outcome_distribution
    return f"{dist}_{_get_outcome_name(config)}_{model_type}_{rank}"


def _clean_scoped_samples(mcmc, model_type: str, rank: int) -> dict:
    """Filter scoped ('/') sample keys via output.drop_scoped_samples, with
    per-run debug logging of what was dropped."""
    from bayesian_panel_nmf.results import drop_scoped_samples

    raw_samples = mcmc.get_samples(group_by_chain=True)
    clean_samples = drop_scoped_samples(raw_samples)
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
    output_config: OutputConfig,
    ppc_draws_df=None,
) -> None:
    """Generate figures + tables under ``<output_dir>/figs/``."""
    from bayesian_panel_nmf.reporting import generate_reports

    aggregate_units = (
        [spec.model_dump() for spec in output_config.aggregate_units]
        if output_config.aggregate_units
        else None
    )
    generate_reports(
        draws_df,
        output_dir=output_dir,
        target_unit=output_config.target_unit,
        groups=output_config.report_groups,
        print_tables=output_config.print_tables,
        print_target_table=output_config.print_target_table,
        aggregate_units=aggregate_units,
        ppc_units=output_config.ppc_units,
        ppc_acf_lags=output_config.ppc_acf_lags or [6],
        ppc_unit_corr_max_time=output_config.ppc_unit_corr_max_time,
        ppc_exclude_units=output_config.ppc_exclude_units,
        ppc_draws_df=ppc_draws_df,
    )


def _prepare_type_output_dir(
    base_output_dir: Path, model_type: str, output_config: OutputConfig
) -> Path:
    """Return this model type's output directory, optionally clearing it
    first when output.clean is true."""
    type_output_dir = base_output_dir / model_type

    if output_config.clean and type_output_dir.exists():
        logger.info(f"clean=true: removing existing {type_output_dir}")
        _safe_rmtree(type_output_dir, base_output_dir)

    type_output_dir.mkdir(parents=True, exist_ok=True)
    return type_output_dir


def _run_single_rank(
    rank: int,
    data_dict: dict,
    model_type: str,
    config: Config,
    type_output_dir: Path,
    ranks: list[int],
    save_traces: bool,
    output_config: OutputConfig,
) -> None:
    """Run MCMC inference for one rank, write the convergence gate,
    optionally save a trace sidecar, write draws, and optionally dispatch
    reporting/figure generation."""
    if config.model.inference_mode == "cut":
        _run_cut_rank(
            rank,
            data_dict,
            model_type,
            config,
            type_output_dir,
            ranks,
            save_traces,
            output_config,
        )
        return

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
    try:
        with open(convergence_file, "w") as f:
            json.dump(gate, f, indent=2)
    except OSError as e:
        logger.error(f"Failed to write convergence file {convergence_file}: {e}")
        raise

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

    draws_file = _write_draws(
        draws_df, type_output_dir / filename, output_config.draws_format
    )
    size_mb = draws_file.stat().st_size / 1024**2
    logger.info(
        f"{model_type} rank {rank}: wrote draws to {draws_file} ({size_mb:.1f} MB)"
    )

    # Multi-rank runs nest under rank_<rank>/ so figs don't collide
    report_dir = type_output_dir / f"rank_{rank}" if len(ranks) > 1 else type_output_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    if output_config.figures:
        _run_reporting(draws_df, report_dir, output_config)


def _publish_cut_artifacts(
    staging: Path,
    dest_dir: Path,
    filename: str,
    save_traces: bool,
    draws_suffix: str = ".csv",
) -> None:
    """Atomically promote staged cut artifacts into the type output dir."""
    names = [
        f"{filename}{draws_suffix}",
        f"{filename}_stage1_ppc.csv",
        f"{filename}_convergence.json",
    ]
    if save_traces:
        names.append(f"{filename}_stage1_traces.nc")
    for name in names:
        os.replace(staging / name, dest_dir / name)
    if save_traces:
        dest_traces = dest_dir / f"{filename}_stage2_traces"
        if dest_traces.exists():
            _safe_rmtree(dest_traces, dest_dir)
        os.replace(staging / f"{filename}_stage2_traces", dest_traces)
    _safe_rmtree(staging, dest_dir)


def _run_cut_rank(
    rank: int,
    data_dict: dict,
    model_type: str,
    config: Config,
    type_output_dir: Path,
    ranks: list[int],
    save_traces: bool,
    output_config: OutputConfig,
) -> None:
    """Two-stage pure cut-posterior path for one rank.

    Stage 1 fits the untreated baseline; a chain-stratified subset of its
    draws each conditions a complete multi-chain Stage-2 fit. Artifacts are
    built in a staging directory and published atomically only after every
    component succeeds. Convergence-gate failures warn and continue;
    execution/data/shape errors abort with staging cleaned and previously
    published artifacts untouched.
    """
    import arviz as az
    from jax import random as jax_random

    from bayesian_panel_nmf.cut import (
        resolve_cut_settings,
        run_stage1_mcmc,
        run_stage2_mcmc,
        sample_untreated_predictions,
        select_stage1_draws,
        subsample_component_draws,
        summarize_mcmc,
        validate_cut_data,
    )
    from bayesian_panel_nmf.results import (
        build_cut_convergence_manifest,
        format_cut_component_draws,
        format_stage1_ppc_draws,
    )
    from bayesian_panel_nmf.validation import DataError

    filename = f"{_draws_filename(config, model_type, rank)}_cut"
    staging = type_output_dir / f".tmp_cut_{filename}"
    if staging.exists():
        logger.warning(f"removing stale cut staging directory {staging}")
        _safe_rmtree(staging, type_output_dir)
    staging.mkdir(parents=True)

    outcome_dist = config.model.outcome_distribution
    sample_disp = config.model.sample_disp
    nb_disp = config.model.nb_disp

    try:
        validate_cut_data(data_dict)
        settings = resolve_cut_settings(config)

        # ---- Stage 1 ---------------------------------------------------
        started = time.monotonic()
        stage1 = run_stage1_mcmc(data_dict, rank, config)
        logger.info(
            f"{model_type} rank {rank} cut: Stage 1 finished in "
            f"{_format_elapsed(time.monotonic() - started)}"
        )
        stage1_diag = summarize_mcmc(stage1)
        if not stage1_diag["converged"]:
            logger.warning(
                f"{model_type} rank {rank} cut: Stage-1 convergence gate FAILED — "
                f"max R-hat={stage1_diag['rhat_max']:.4f}, "
                f"min bulk ESS={stage1_diag['ess_bulk_min']:.0f}, "
                f"divergences={stage1_diag['divergences']}"
            )

        # ---- Full Stage-1 PPC product (independent +1 stream) ----------
        mu1 = stage1.samples["mu_ctrl"]
        if outcome_dist == "NB":
            if sample_disp and "disp" in stage1.samples:
                conc1 = (1.0 / stage1.samples["disp"])[:, :, None, :, None]
            else:
                conc1 = (np.ones(mu1.shape[3]) / nb_disp)[:, None]
        else:
            conc1 = None
        _, ppc_key = jax_random.split(
            jax_random.PRNGKey(int(config.mcmc.random_seed) + 1)
        )
        stage1_ypred = sample_untreated_predictions(mu1, conc1, outcome_dist, ppc_key)
        stage1_ppc_df = format_stage1_ppc_draws(mu1, stage1_ypred, data_dict)
        stage1_ppc_df.to_csv(staging / f"{filename}_stage1_ppc.csv", index=False)
        del stage1_ppc_df, stage1_ypred

        if save_traces:
            az.from_dict(
                {
                    "posterior": stage1.samples,
                    "sample_stats": {"diverging": stage1.diverging},
                }
            ).to_netcdf(
                str(staging / f"{filename}_stage1_traces.nc"), engine="h5netcdf"
            )

        refs = select_stage1_draws(stage1.samples, settings, config.model.model_dump())
        del stage1, mu1  # release the full Stage-1 posterior; refs hold copies

        # ---- Stage 2: one conditional multi-chain fit per component ----
        fit_root, pred_root = jax_random.split(jax_random.PRNGKey(settings.stage2_seed))
        fit_keys = jax_random.split(fit_root, len(refs))
        pred_keys = jax_random.split(pred_root, len(refs))

        traces_dir = staging / f"{filename}_stage2_traces"
        if save_traces:
            traces_dir.mkdir()

        draws_format = output_config.draws_format
        combined_stem = staging / filename
        combined_path = combined_stem.with_suffix(".csv")
        component_records: list[dict] = []
        component_dfs: list[pd.DataFrame] = []
        output_counts: set[int] = set()
        draw_offset = 0

        for i, ref in enumerate(refs):
            started = time.monotonic()
            fit = run_stage2_mcmc(
                data_dict, ref, config, settings.stage2_mcmc, fit_keys[i]
            )
            diag = summarize_mcmc(fit)
            logger.info(
                f"{model_type} rank {rank} cut: component {ref.component}/"
                f"{len(refs)} (stage1 chain {ref.stage1_chain}, "
                f"iter {ref.stage1_iteration}) in "
                f"{_format_elapsed(time.monotonic() - started)}"
            )
            if not diag["converged"]:
                logger.warning(
                    f"{model_type} rank {rank} cut: component {ref.component} "
                    f"convergence gate FAILED — max R-hat={diag['rhat_max']:.4f}"
                )

            if save_traces:
                az.from_dict(
                    {
                        "posterior": fit.samples,
                        "sample_stats": {"diverging": fit.diverging},
                    }
                ).to_netcdf(
                    str(traces_dir / f"component_{ref.component:04d}.nc"),
                    engine="h5netcdf",
                )

            idx_per_chain = subsample_component_draws(
                fit.num_chains, fit.num_retained, settings.stage2_draws_per_component
            )
            te_flat = np.concatenate(
                [
                    fit.samples["te"][c][ix]
                    for c, ix in enumerate(idx_per_chain)
                    if len(ix)
                ]
            )
            chain_ids = np.concatenate(
                [
                    np.full(len(ix), c + 1, dtype=np.int8)
                    for c, ix in enumerate(idx_per_chain)
                    if len(ix)
                ]
            )
            n_out = int(te_flat.shape[0])

            conc_b = (
                None if ref.nb_concentration is None else ref.nb_concentration[:, None]
            )
            mu_grid = np.broadcast_to(ref.mu_ctrl, te_flat.shape)
            ypred = sample_untreated_predictions(
                mu_grid, conc_b, outcome_dist, pred_keys[i]
            )

            df = format_cut_component_draws(
                te_flat, ypred, chain_ids, ref, data_dict, draw_offset
            )
            if draws_format == "parquet":
                # parquet has no cheap append mode; buffer components and
                # write once below via _write_draws.
                component_dfs.append(df)
            else:
                df.to_csv(
                    combined_path,
                    mode="w" if i == 0 else "a",
                    header=(i == 0),
                    index=False,
                )

            component_records.append(
                {
                    "component": int(ref.component),
                    "stage1_draw": int(ref.stage1_draw),
                    "stage1_chain": int(ref.stage1_chain),
                    "stage1_iteration": int(ref.stage1_iteration),
                    **diag,
                    "retained_draws": int(fit.num_chains * fit.num_retained),
                    "output_draws": n_out,
                }
            )
            output_counts.add(n_out)
            draw_offset += n_out
            if draws_format != "parquet":
                del df
            del fit, te_flat, ypred

        if draws_format == "parquet":
            combined_path = _write_draws(
                pd.concat(component_dfs, ignore_index=True), combined_stem, "parquet"
            )
            component_dfs.clear()

        if len(output_counts) > 1:
            raise DataError(
                f"unequal output draw counts across cut components: "
                f"{sorted(output_counts)} — equal counts preserve equal weights"
            )

        manifest = build_cut_convergence_manifest(stage1_diag, component_records)
        with open(staging / f"{filename}_convergence.json", "w") as f:
            json.dump(manifest, f, indent=2)
        if not manifest["converged"]:
            logger.warning(
                f"{model_type} rank {rank} cut: overall convergence gate FAILED "
                "(see manifest)"
            )

        _publish_cut_artifacts(
            staging, type_output_dir, filename, save_traces, combined_path.suffix
        )
    except Exception:
        _safe_rmtree(staging, type_output_dir)
        raise

    logger.info(
        f"{model_type} rank {rank} cut: wrote draws to "
        f"{type_output_dir / (filename + combined_path.suffix)}"
    )

    # ---- Reporting from published artifacts (same as re-render path) ----
    report_dir = type_output_dir / f"rank_{rank}" if len(ranks) > 1 else type_output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    if output_config.figures:
        draws_df = _read_draws(type_output_dir / filename)
        ppc_df = pd.read_csv(type_output_dir / f"{filename}_stage1_ppc.csv")
        _run_reporting(draws_df, report_dir, output_config, ppc_draws_df=ppc_df)


def run_model_type(
    model_type: str,
    type_config: TypeConfig,
    config: Config,
    rank_override: int | None = None,
    save_traces: bool = False,
    log_level: str = "INFO",
    configure_logging: bool = True,
) -> None:
    """Run analysis for a single model type across specified ranks."""
    if configure_logging:
        setup_logging(level=log_level)

    base_output_dir = Path(config.data.output_dir)
    output_config = config.output
    type_output_dir = _prepare_type_output_dir(
        base_output_dir, model_type, output_config
    )

    groups = type_config.groups
    ranks = [rank_override] if rank_override else type_config.ranks_to_test
    exclude_units = type_config.exclude_units

    model_started_at = time.monotonic()
    load_started_at = time.monotonic()
    data_dict = load_and_prepare(
        filepath=config.data.input_file,
        config=config,
        groups=groups,
        exclude_units=exclude_units,
        type_config=type_config.model_dump(),
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
        _run_single_rank(
            rank,
            data_dict,
            model_type,
            config,
            type_output_dir,
            ranks,
            save_traces,
            output_config,
        )

    logger.info(
        f"{model_type}: complete in {_format_elapsed(time.monotonic() - model_started_at)}"
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


def _select_types_to_run(
    all_types: dict[str, TypeConfig], requested_type: str | None
) -> dict[str, TypeConfig]:
    """Return the subset of configured model types to run: just the
    requested one if --type was given, otherwise every configured type.

    Raises
    ------
    ConfigError
        If requested_type is set but not present in all_types.
    """
    if requested_type:
        if requested_type not in all_types:
            raise ConfigError(
                f"--type={requested_type!r} not found in config; "
                f"available: {list(all_types)}"
            )
        return {requested_type: all_types[requested_type]}
    return all_types


def _run_sequential(
    types_to_run: dict[str, TypeConfig],
    config: Config,
    args: argparse.Namespace,
    save_traces: bool,
    log_level: str,
) -> None:
    """Run each model type one at a time in this process (preserves rich
    per-type logging; used when workers == 1)."""
    total_count = len(types_to_run)
    for index, (model_type, type_config) in enumerate(types_to_run.items(), start=1):
        logger.info(f"RUNNING MODEL TYPE: {model_type.upper()} ({index}/{total_count})")
        run_model_type(
            model_type=model_type,
            type_config=type_config,
            config=config,
            rank_override=args.rank,
            save_traces=save_traces,
            log_level=log_level,
            configure_logging=False,
        )


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

    _run_sequential(types_to_run, config, args, save_traces, log_level)

    logger.info(f"Analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
