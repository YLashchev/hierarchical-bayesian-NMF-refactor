"""Per-type / per-rank analysis pipeline: data loading, MCMC dispatch,
convergence gating, draws serialization, reporting dispatch.

CLI parsing, config loading, and the module-top
``numpyro.set_host_device_count()`` call stay in ``cli.py``. This module
imports ``inference``/``cut`` (which import jax), so it must never run before
that call -- see the ordering note in ``cli.py``.
"""

import json
import os
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from loguru import logger

from bayesian_panel_nmf.checks import _safe_rmtree
from bayesian_panel_nmf.config import Config, OutputConfig, TypeConfig
from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.diagnostics import (
    ConvergenceThresholds,
    convergence_summary,
    parameter_diagnostics,
)
from bayesian_panel_nmf.inference import generate_predictions, run_mcmc_inference
from bayesian_panel_nmf.logging_config import setup_logging
from bayesian_panel_nmf.models import model
from bayesian_panel_nmf.results import format_draws
from bayesian_panel_nmf.validation import ConfigError


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
    3. Fall back to the domain-neutral ``"outcome"``.
    """
    explicit = config.data.outcome
    if explicit:
        return str(explicit)

    prefixes = config.data.schema_.outcomes_from_prefixes
    if prefixes:
        prefix = prefixes.outcome_prefix or ""
        if prefix.endswith("_"):
            return prefix[:-1]
        return prefix or "outcome"

    return "outcome"


def _draws_filename(config: Config, model_type: str, rank: int) -> str:
    """Fixed scheme: {distribution}_{outcome}_{type}_{rank}."""
    dist = config.model.outcome_distribution
    return f"{dist}_{_get_outcome_name(config)}_{model_type}_{rank}"


def _clean_scoped_samples(mcmc, model_type: str, rank: int) -> dict:
    """Filter scoped ('/') sample keys via results.drop_scoped_samples, with
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
    figures_override: list[str] | None = None,
) -> None:
    """Generate figures + tables under ``<output_dir>/figs/``.

    ``figures_override`` (when not ``None``) replaces ``output_config.figures``
    for this render -- e.g. ``[]`` from ``viz --tables-only`` to write/print
    tables while skipping every figure.
    """
    from bayesian_panel_nmf.reports import generate_reports

    aggregate_units = (
        list(output_config.aggregate_units) if output_config.aggregate_units else None
    )
    figures = (
        figures_override if figures_override is not None else output_config.figures
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
        figures=figures,
        fit_gap_per_unit=output_config.fit_gap_per_unit,
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
    thresholds = ConvergenceThresholds(**config.mcmc.convergence.model_dump())
    gate = convergence_summary(
        idata, params=config.mcmc.gate_params, thresholds=thresholds
    )
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

    from bayesian_panel_nmf.tables import print_run_summary_panel

    print_run_summary_panel(
        model_type=model_type,
        rank=rank,
        num_chains=mcmc.num_chains,
        chain_method=str(mcmc.chain_method),
        outcome_distribution=config.model.outcome_distribution,
        convergence=gate,
        figures=output_config.figures,
        artifact_paths=[str(draws_file), str(convergence_file)]
        + ([str(report_dir)] if output_config.figures else []),
        diagnostic_rows=parameter_diagnostics(
            idata, gate_params=config.mcmc.gate_params, thresholds=thresholds
        ),
    )


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


def _cut_stage1(
    rank: int,
    data_dict: dict,
    model_type: str,
    config: Config,
    settings,
    staging: Path,
    filename: str,
    save_traces: bool,
):
    """Fit the untreated baseline, write its PPC product (+ optional traces),
    and select the chain-stratified Stage-1 draws that seed Stage 2.

    Returns ``(stage1_diag, refs, stage1_num_chains)``. Releases the full
    Stage-1 posterior before returning; ``refs`` hold the selected copies.
    The Stage-1 PPC uses the independent ``mcmc.random_seed + 1`` stream.
    """
    import arviz as az
    from jax import random as jax_random

    from bayesian_panel_nmf.cut import (
        run_stage1_mcmc,
        sample_untreated_predictions,
        select_stage1_draws,
        summarize_mcmc,
    )
    from bayesian_panel_nmf.results import format_stage1_ppc_draws

    outcome_dist = config.model.outcome_distribution
    sample_disp = config.model.sample_disp
    nb_disp = config.model.nb_disp

    started = time.monotonic()
    stage1 = run_stage1_mcmc(data_dict, rank, config)
    logger.info(
        f"{model_type} rank {rank} cut: Stage 1 finished in "
        f"{_format_elapsed(time.monotonic() - started)}"
    )
    stage1_diag = summarize_mcmc(
        stage1,
        params=config.mcmc.gate_params,
        thresholds=ConvergenceThresholds(**config.mcmc.convergence.model_dump()),
    )
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
    _, ppc_key = jax_random.split(jax_random.PRNGKey(int(config.mcmc.random_seed) + 1))
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
        ).to_netcdf(str(staging / f"{filename}_stage1_traces.nc"), engine="h5netcdf")

    refs = select_stage1_draws(stage1.samples, settings, config.model)
    stage1_num_chains = stage1.num_chains
    del stage1, mu1  # release the full Stage-1 posterior; refs hold copies
    return stage1_diag, refs, stage1_num_chains


def _cut_stage2_components(
    refs,
    data_dict: dict,
    model_type: str,
    rank: int,
    config: Config,
    settings,
    staging: Path,
    filename: str,
    combined_stem: Path,
    draws_format: str,
    save_traces: bool,
):
    """Run one conditional multi-chain Stage-2 fit per selected baseline draw.

    Writes the combined draws artifact (csv incremental-append or a single
    parquet write) and returns ``(component_records, combined_path)``. Uses
    the ``settings.stage2_seed`` stream, split into per-component fit/predict
    keys. Raises ``DataError`` if components yield unequal output-draw counts
    (equal counts preserve the equal-weight nested Monte Carlo).
    """
    import arviz as az
    from jax import random as jax_random

    from bayesian_panel_nmf.cut import (
        run_stage2_mcmc,
        sample_untreated_predictions,
        subsample_component_draws,
        summarize_mcmc,
    )
    from bayesian_panel_nmf.results import format_cut_component_draws
    from bayesian_panel_nmf.validation import DataError

    outcome_dist = config.model.outcome_distribution

    fit_root, pred_root = jax_random.split(jax_random.PRNGKey(settings.stage2_seed))
    fit_keys = jax_random.split(fit_root, len(refs))
    pred_keys = jax_random.split(pred_root, len(refs))

    traces_dir = staging / f"{filename}_stage2_traces"
    if save_traces:
        traces_dir.mkdir()

    combined_path = combined_stem.with_suffix(".csv")
    component_records: list[dict] = []
    component_dfs: list[pd.DataFrame] = []
    output_counts: set[int] = set()
    draw_offset = 0

    for i, ref in enumerate(refs):
        started = time.monotonic()
        fit = run_stage2_mcmc(data_dict, ref, config, settings.stage2_mcmc, fit_keys[i])
        diag = summarize_mcmc(
            fit,
            params=config.mcmc.gate_params,
            thresholds=ConvergenceThresholds(**config.mcmc.convergence.model_dump()),
        )
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
            [fit.samples["te"][c][ix] for c, ix in enumerate(idx_per_chain) if len(ix)]
        )
        chain_ids = np.concatenate(
            [
                np.full(len(ix), c + 1, dtype=np.int8)
                for c, ix in enumerate(idx_per_chain)
                if len(ix)
            ]
        )
        n_out = int(te_flat.shape[0])

        conc_b = None if ref.nb_concentration is None else ref.nb_concentration[:, None]
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

    return component_records, combined_path


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

    Thin orchestrator: stage 1 (``_cut_stage1``) fits the untreated baseline
    and selects the draws that each seed a complete Stage-2 fit
    (``_cut_stage2_components``). Artifacts are built in a staging directory
    and published atomically only after every component succeeds.
    Convergence-gate failures warn and continue; execution/data/shape errors
    abort with staging cleaned and previously published artifacts untouched.
    """
    from bayesian_panel_nmf.cut import (
        _resolve_chains,
        resolve_cut_settings,
        validate_cut_data,
    )
    from bayesian_panel_nmf.results import build_cut_convergence_manifest
    from bayesian_panel_nmf.tables import print_run_summary_panel

    filename = f"{_draws_filename(config, model_type, rank)}_cut"
    staging = type_output_dir / f".tmp_cut_{filename}"
    if staging.exists():
        logger.warning(f"removing stale cut staging directory {staging}")
        _safe_rmtree(staging, type_output_dir)
    staging.mkdir(parents=True)

    outcome_dist = config.model.outcome_distribution

    try:
        validate_cut_data(data_dict)
        settings = resolve_cut_settings(config)

        stage1_diag, refs, stage1_num_chains = _cut_stage1(
            rank,
            data_dict,
            model_type,
            config,
            settings,
            staging,
            filename,
            save_traces,
        )

        combined_stem = staging / filename
        component_records, combined_path = _cut_stage2_components(
            refs,
            data_dict,
            model_type,
            rank,
            config,
            settings,
            staging,
            filename,
            combined_stem,
            output_config.draws_format,
            save_traces,
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
    artifact_paths = [
        str(type_output_dir / f"{filename}{combined_path.suffix}"),
        str(type_output_dir / f"{filename}_convergence.json"),
    ]
    if output_config.figures:
        draws_df = _read_draws(type_output_dir / filename)
        ppc_df = pd.read_csv(type_output_dir / f"{filename}_stage1_ppc.csv")
        _run_reporting(draws_df, report_dir, output_config, ppc_draws_df=ppc_df)
        artifact_paths.append(str(report_dir))

    _, cut_chain_method = _resolve_chains(config.mcmc.model_dump())
    print_run_summary_panel(
        model_type=model_type,
        rank=rank,
        num_chains=stage1_num_chains,
        chain_method=cut_chain_method,
        outcome_distribution=outcome_dist,
        convergence=manifest,
        figures=output_config.figures,
        artifact_paths=artifact_paths,
        component_fits=manifest["stage2"]["fits"],
    )


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
    rank_override: int | None,
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
            rank_override=rank_override,
            save_traces=save_traces,
            log_level=log_level,
            configure_logging=False,
        )
