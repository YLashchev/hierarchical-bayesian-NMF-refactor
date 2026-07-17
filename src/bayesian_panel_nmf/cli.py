"""``bpnmf`` command-line entry point.

Consolidates the four former standalone scripts
(``scripts/run_analysis.py``, ``scripts/generate_full_viz.py``,
``scripts/analyze_traces.py``, ``scripts/make_trace_plots.py``, all deleted
in Phase 9.2 of the legibility refactor) into one installed console script
with subcommands: ``run``, ``viz``, ``traces``, ``init``.

Module-top import order is a hard runtime requirement, carried over
unchanged from ``scripts/run_analysis.py``:
``numpyro.set_host_device_count()`` only takes effect before JAX's backend
is lazily initialized (it sets an XLA_FLAGS env var that XLA reads once, at
first use). It MUST run before any jax/numpyro/arviz import — including
imports inside ``bayesian_panel_nmf.inference``/``pipeline`` — or it is a
silent no-op and NUTS's ``MCMC(..., chain_method="parallel")`` falls back to
sequential chain execution (root cause of ``mcmc.num_chains`` chains not
actually running concurrently on multi-core CPUs).
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpyro  # noqa: E402

numpyro.set_host_device_count(os.cpu_count() or 1)

import arviz as az  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from bayesian_panel_nmf.config import Config  # noqa: E402
from bayesian_panel_nmf.logging_config import setup_logging  # noqa: E402
from bayesian_panel_nmf.pipeline import (  # noqa: E402
    _read_draws,
    _run_sequential,
    _select_types_to_run,
    _validate_run_analysis_config,
)
from bayesian_panel_nmf.reports import generate_reports  # noqa: E402
from bayesian_panel_nmf.validation import ConfigError, DataError  # noqa: E402

matplotlib.use("Agg")  # headless — no display needed

_MAX_TRACE_SUBPLOTS = 20
_BASE_CONFIG_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "configs" / "base_config.yaml"
)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


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


def _run_command(args: argparse.Namespace) -> None:
    """``bpnmf run`` — port of ``scripts/run_analysis.py``'s ``main()``."""
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, log_file=args.log_file)

    config = load_config(args.config)
    _validate_run_analysis_config(config)

    output_dir = Path(config.data.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {output_dir}: {e}")
        raise

    config = _apply_mcmc_overrides(config, args)

    types_to_run = _select_types_to_run(config.model.types, args.type)

    save_traces = args.save_traces or config.output.save_traces

    logger.info(f"Running {len(types_to_run)} model type(s)")

    _run_sequential(types_to_run, config, args.rank, save_traces, log_level)

    logger.info(f"Analysis complete. Results saved to: {output_dir}")


def _add_run_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "run", help="Run the Bayesian Panel NMF analysis pipeline"
    )
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
    parser.set_defaults(func=_run_command)


# ---------------------------------------------------------------------------
# viz
# ---------------------------------------------------------------------------


def _viz_command(args: argparse.Namespace) -> None:
    """``bpnmf viz`` — port of ``scripts/generate_full_viz.py``."""
    results_path = Path(args.results)
    stem = results_path.with_suffix("")
    draws_df = _read_draws(stem)
    output_dir = results_path.parent
    ppc_draws_df = pd.read_csv(args.ppc_results) if args.ppc_results else None
    generate_reports(
        draws_df,
        output_dir=output_dir,
        target_unit=args.target,
        groups=args.group,
        ppc_draws_df=ppc_draws_df,
    )


def _add_viz_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "viz", help="Regenerate figures + tables from an existing draws artifact"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results/total/NB_births_total_5.csv",
        help="Path to the draws CSV/parquet produced by `bpnmf run`",
    )
    parser.add_argument(
        "--ppc-results",
        type=str,
        default=None,
        help=(
            "Optional Stage-1 PPC draws CSV (cut mode: <stem>_cut_stage1_ppc.csv); "
            "routes the JAMA PPC suite to the full Stage-1 posterior"
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target unit for fit/gap/summary plots (auto-detected if omitted)",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        action="append",
        help="Group label for per-unit plots. Repeat for multiple groups; "
        "omit to render every group present in the draws CSV.",
    )
    parser.set_defaults(func=_viz_command)


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------


def _print_trace_diagnostics(nc_path: Path, param_filter: str | None) -> bool:
    """Numeric R-hat/ESS pass-fail table, ported from
    ``scripts/analyze_traces.py``. Returns True iff every parameter group
    passed the gate."""
    logger.info(f"Loading NetCDF traces: {nc_path.name}")
    idata = az.from_netcdf(nc_path)

    var_names = None
    if param_filter:
        prefixes = param_filter.split(",")
        var_names = [
            v
            for v in idata.posterior.data_vars
            if any(v.startswith(p) for p in prefixes)
        ]
        if not var_names:
            logger.error("No parameters matched --param-filter.")
            return False

    logger.info("Computing metrics via ArviZ...")
    summary_df = az.summary(idata, var_names=var_names, filter_vars="like")
    summary_df["base_param"] = summary_df.index.to_series().apply(
        lambda x: x.split("[")[0]
    )

    all_ok = True
    for name, group in summary_df.groupby("base_param"):
        rhat = float(group["r_hat"].max())
        ess = float(group["ess_bulk"].min())

        status = "PASS" if rhat < 1.01 and ess > 400 else "WARN"
        if rhat >= 1.01 or ess < 100:
            status = "FAIL"
            all_ok = False

        print(f"\n{name}: {status}")
        print(f"  max R-hat: {rhat:.4f}")
        print(f"  min ESS:   {ess:.0f}")

    return all_ok


def _select_variables_to_plot(
    all_vars: list[str], param_filters: list[str] | None
) -> list[str]:
    """Choose which posterior variables to trace-plot.

    With --param-filter: any variable whose name starts with one of the
    given prefixes, falling back to the first 5 available variables if
    nothing matches.

    Without --param-filter: scalar/low-dimensional parameters (dispersion,
    treatment/category scales) first, then higher-dimensional matrix
    parameters (treatment effects, factors, fixed effects), in a fixed
    priority order. Falls back to the first 5 available variables if none
    of the priority names are present.
    """
    if param_filters:
        vars_to_plot = [
            v for v in all_vars if any(v.startswith(f) for f in param_filters)
        ]
        if not vars_to_plot:
            logger.warning(
                f"No parameters matched filters {param_filters}. Available: {all_vars}"
            )
            vars_to_plot = all_vars[:5]
        return vars_to_plot

    scalar_priority = [
        "disp",
        "treatment_state_scale",
        "treatment_category_scale",
        "state_category_scale",
    ]
    matrix_params = [
        "te",
        "state_treatment_effect",
        "category_treatment_effect",
        "time_fac",
        "state_fe",
        "time_fe",
        "unit_weight",
    ]
    vars_to_plot = [v for v in scalar_priority if v in all_vars] + [
        v for v in matrix_params if v in all_vars
    ]
    if not vars_to_plot:
        vars_to_plot = all_vars[:5]
    return vars_to_plot


def _select_informative_coords(
    da, max_subplots: int, rng: np.random.Generator
) -> dict | None:
    """Pick a subset of coordinates per non-chain/draw dimension so the
    total subplot count stays near max_subplots, preferring cells with
    non-zero posterior variance (structurally-zero cells like `te` on
    control units produce uninformative flatline traces).

    Returns None if the variable is structurally zero everywhere (caller
    should skip it entirely). Returns {} if no subsampling is needed.
    Returns a populated dict otherwise.
    """
    param_dims = [d for d in da.dims if d not in ("chain", "draw")]
    coords: dict = {}
    if not param_dims:
        return coords

    cell_var = da.var(dim=("chain", "draw"))
    nonzero_mask = cell_var > 1e-12
    n_nonzero = int(nonzero_mask.sum())
    n_total = int(cell_var.size)
    logger.info(f"  {da.name}: {n_nonzero}/{n_total} cells have non-zero variance")

    if n_nonzero == 0:
        return None

    per_dim = max(1, int(max_subplots ** (1.0 / len(param_dims))))

    if n_nonzero < n_total:
        for axis, dim in enumerate(param_dims):
            other_axes = tuple(i for i in range(len(param_dims)) if i != axis)
            per_dim_mask = (
                nonzero_mask.values.any(axis=other_axes)
                if other_axes
                else nonzero_mask.values
            )
            informative_idx = np.flatnonzero(per_dim_mask)
            if informative_idx.size == 0:
                continue
            k = min(per_dim, informative_idx.size)
            chosen = sorted(rng.choice(informative_idx, size=k, replace=False).tolist())
            coords[dim] = da[dim].values[chosen]
            logger.debug(
                f"  {da.name}: dim '{dim}' → {k} informative coords of {informative_idx.size}"
            )
    else:
        for dim in param_dims:
            size = da.sizes[dim]
            if size > per_dim:
                chosen = sorted(rng.choice(size, size=per_dim, replace=False).tolist())
                coords[dim] = da[dim].values[chosen]
                logger.debug(f"  {da.name}: subsampling dim '{dim}' {size}→{per_dim}")

    return coords


def _render_and_save_trace(idata, var: str, coords: dict, out_dir: Path) -> Path:
    """Render one variable's trace plot and save it as a PNG. Caller handles
    exceptions."""
    az.plot_trace(idata, var_names=[var], coords=coords or None, backend="matplotlib")
    fig = plt.gcf()
    suffix = f" (subset {_MAX_TRACE_SUBPLOTS})" if coords else ""
    fig.suptitle(f"Trace — {var}{suffix}", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path = out_dir / f"trace_{var}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _make_trace_plots_from_netcdf(
    nc_path: Path,
    param_filters: list[str] | None,
    out_dir: Path,
) -> list[Path]:
    """Load InferenceData from NetCDF sidecar and trace real latent parameters."""
    logger.info(f"Loading NetCDF sidecar: {nc_path}")
    idata = az.from_netcdf(str(nc_path))

    posterior = idata.posterior
    all_vars = list(posterior.data_vars)
    logger.info(f"Available parameters: {all_vars}")

    vars_to_plot = _select_variables_to_plot(all_vars, param_filters)
    logger.info(f"Tracing parameters: {vars_to_plot}")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    rng = np.random.default_rng(42)
    for var in vars_to_plot:
        logger.info(f"Plotting {var} ...")
        try:
            da = posterior[var]
            coords = _select_informative_coords(
                da, max_subplots=_MAX_TRACE_SUBPLOTS, rng=rng
            )
            if coords is None:
                logger.warning(
                    f"Skipped {var}: all cells structurally zero (flatline)."
                )
                continue
            out_path = _render_and_save_trace(idata, var, coords, out_dir)
            logger.info(f"Saved {out_path}")
            saved.append(out_path)
        except Exception as e:
            logger.warning(f"Skipped {var}: {e}")
            plt.close("all")

    return saved


def _traces_command(args: argparse.Namespace) -> None:
    """``bpnmf traces`` — folds ``scripts/analyze_traces.py`` (default,
    numeric table) and ``scripts/make_trace_plots.py`` (``--plots``, PNGs)."""
    nc_path = Path(args.nc_path)
    if not nc_path.exists():
        logger.error(f"File not found: {nc_path}")
        sys.exit(1)

    if not args.plots:
        all_ok = _print_trace_diagnostics(nc_path, args.param_filter)
        sys.exit(0 if all_ok else 1)

    if nc_path.suffix != ".nc":
        sys.exit(
            f"Expected a NetCDF traces file (*.nc), got {nc_path}. "
            "Run with --save-traces."
        )

    out_dir = (
        Path(args.out_dir) if args.out_dir else nc_path.parent / "figs" / "diagnostics"
    )
    filters = (
        [f.strip() for f in args.param_filter.split(",")] if args.param_filter else None
    )
    saved = _make_trace_plots_from_netcdf(nc_path, filters, out_dir)

    if not saved:
        sys.exit(1)

    print(f"\nTrace plots written ({len(saved)} file(s)):")
    for p in saved:
        print(f"  {p}")


def _add_traces_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "traces",
        help="Compute R-hat/ESS diagnostics or render trace plots from a NetCDF sidecar",
    )
    parser.add_argument(
        "nc_path", help="Path to NetCDF traces file (e.g. ..._traces.nc)"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Render PNG trace plots instead of the numeric R-hat/ESS table",
    )
    parser.add_argument(
        "--param-filter", help="Comma-separated list of parameter prefixes"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for --plots (default: <input_dir>/figs/diagnostics/)",
    )
    parser.set_defaults(func=_traces_command)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def _init_command(args: argparse.Namespace) -> None:
    """``bpnmf init`` — copy the commented base_config.yaml template."""
    target = Path(args.path)
    if target.exists() and not args.force:
        logger.error(f"{target} already exists; use --force to overwrite")
        sys.exit(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_BASE_CONFIG_TEMPLATE, target)
    print(f"Wrote starter config: {target}")


def _add_init_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "init", help="Write a starter config file (copy of configs/base_config.yaml)"
    )
    parser.add_argument(
        "path", nargs="?", default="config.yaml", help="Destination path for the config"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing file"
    )
    parser.set_defaults(func=_init_command)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bpnmf", description="Bayesian Panel NMF analysis CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    _add_viz_parser(subparsers)
    _add_traces_parser(subparsers)
    _add_init_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ConfigError, DataError) as e:
        # User-fixable input errors (bad config, missing file/column): show a
        # clean one-line message and exit non-zero, not a raw traceback.
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
