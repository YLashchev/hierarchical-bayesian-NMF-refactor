"""``bpnmf`` command-line entry point. Subcommands: run, viz, traces, init.

Module-top import order is a hard runtime requirement:
``numpyro.set_host_device_count()`` sets an XLA_FLAGS env var that XLA reads
once, at first use, so it MUST run before any jax/numpyro/arviz import
(including those inside ``inference``/``pipeline``). Run it late and it is a
silent no-op: ``chain_method="parallel"`` then falls back to sequential, so
``mcmc.num_chains`` chains never run concurrently on multi-core CPUs.
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
from rich.console import Console  # noqa: E402
from rich.prompt import Confirm, Prompt  # noqa: E402

from bayesian_panel_nmf.config import Config  # noqa: E402
from bayesian_panel_nmf.diagnostics import parameter_diagnostics  # noqa: E402
from bayesian_panel_nmf.logging_config import setup_logging  # noqa: E402
from bayesian_panel_nmf.pipeline import (  # noqa: E402
    _read_draws,
    _run_sequential,
    _select_types_to_run,
    _validate_run_analysis_config,
)
from bayesian_panel_nmf.reports import generate_reports  # noqa: E402
from bayesian_panel_nmf.tables import render_diagnostics_table  # noqa: E402
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


def _interactive_run_setup(args: argparse.Namespace) -> argparse.Namespace:
    """Fill config/type/save_traces by prompting when --config is omitted.

    Explicit --config bypasses every prompt. Non-TTY without --config is an
    error (scripts must pass --config; we never guess a default config).
    """
    if args.config is not None:
        return args
    if not sys.stdin.isatty():
        raise ConfigError("no --config given and not a terminal; pass --config")

    configs = sorted(Path("configs").glob("*.yaml")) if Path("configs").is_dir() else []
    if not configs:
        raise ConfigError("No config files found under ./configs/; pass --config")

    console = Console(stderr=True)
    console.print("[bold]Config:[/bold]")
    for i, p in enumerate(configs, 1):
        console.print(f"  {i}) {p.stem}")
    idx = int(Prompt.ask("Config", choices=[str(i) for i in range(1, len(configs) + 1)]))
    args.config = str(configs[idx - 1])

    types = list(Config.from_yaml(args.config).model.types)
    console.print("[bold]Model type:[/bold]")
    for i, t in enumerate(types, 1):
        console.print(f"  {i}) {t}")
    console.print(f"  {len(types) + 1}) (all)")
    t_idx = int(
        Prompt.ask("Type", choices=[str(i) for i in range(1, len(types) + 2)])
    )
    args.type = types[t_idx - 1] if t_idx <= len(types) else None

    args.save_traces = args.save_traces or Confirm.ask("Save traces?", default=False)
    return args


def _run_command(args: argparse.Namespace) -> None:
    """``bpnmf run`` — port of ``scripts/run_analysis.py``'s ``main()``."""
    args = _interactive_run_setup(args)
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
        default=None,
        help="Path to config file (omit for an interactive picker)",
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


def _discover_draws_files() -> list[Path]:
    """Find run-produced draws files under ./results*/ (CSV or parquet).

    Draws live directly in a type dir (never under figs/) and are the only
    CSVs there apart from df_<type>.csv and the *_stage1_ppc.csv sidecar.
    """
    found = []
    for p in sorted(Path(".").glob("results*/**/*")):
        if p.suffix not in (".csv", ".parquet") or "figs" in p.parts:
            continue
        if p.name.startswith("df_") or p.stem.endswith("_ppc"):
            continue
        found.append(p)
    return found


def _interactive_viz_setup(args: argparse.Namespace) -> argparse.Namespace:
    """Fill results/ppc_results by prompting when --results is omitted.

    Explicit --results bypasses prompting. Picking a cut run auto-attaches
    its *_stage1_ppc.csv sidecar so the PPC suite uses the full Stage-1
    posterior without the user typing the second path.
    """
    if args.results is not None:
        return args
    if not sys.stdin.isatty():
        raise ConfigError("no --results given and not a terminal; pass --results")

    draws = _discover_draws_files()
    if not draws:
        raise ConfigError("No draws files found under ./results*/; pass --results")

    console = Console(stderr=True)
    console.print("[bold]Draws file:[/bold]")
    for i, p in enumerate(draws, 1):
        console.print(f"  {i}) {p}")
    idx = int(Prompt.ask("Draws", choices=[str(i) for i in range(1, len(draws) + 1)]))
    chosen = draws[idx - 1]
    args.results = str(chosen)

    if args.ppc_results is None:
        sidecar = chosen.with_name(f"{chosen.stem}_stage1_ppc.csv")
        if sidecar.exists():
            args.ppc_results = str(sidecar)
            console.print(f"  (cut run — attaching {sidecar.name})")
    return args


def _viz_command(args: argparse.Namespace) -> None:
    """``bpnmf viz`` — port of ``scripts/generate_full_viz.py``."""
    args = _interactive_viz_setup(args)
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
        default=None,
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


def _diag_rows(idata, param_filter: str | None) -> list[dict]:
    """parameter_diagnostics with the CLI's comma-string filter + clean error."""
    prefixes = param_filter.split(",") if param_filter else None
    try:
        return parameter_diagnostics(idata, params=prefixes)
    except ValueError as e:
        raise ConfigError(str(e)) from e


def _print_trace_diagnostics(nc_path: Path, param_filter: str | None) -> bool:
    """Rich R-hat/ESS table for one sidecar. True iff no parameter FAILs."""
    logger.info(f"Loading NetCDF traces: {nc_path.name}")
    rows = _diag_rows(az.from_netcdf(nc_path), param_filter)
    return render_diagnostics_table(rows, title=nc_path.stem)


def _print_component_summary(comp_dir: Path, param_filter: str | None) -> bool:
    """One row per cut Stage-2 component: worst parameter's diagnostics.

    Diagnostics stay per-component (never pooled across conditional
    targets); this only tabulates each component's own gate side by side.
    """
    from rich.table import Table

    nc_files = sorted(comp_dir.glob("*.nc"))
    if not nc_files:
        raise ConfigError(f"No component .nc files in {comp_dir}")

    style = {"PASS": "bold green", "WARN": "yellow", "FAIL": "bold red", "fixed": "dim"}
    t = Table(title=f"{comp_dir.name} — {len(nc_files)} components")
    for col in ("Component", "worst param", "max R-hat", "min bulk ESS", "status"):
        t.add_column(col, justify="right" if "R-hat" in col or "ESS" in col else "left")

    all_ok, n_fail = True, 0
    for nc in nc_files:
        rows = _diag_rows(az.from_netcdf(nc), param_filter)
        varying = [r for r in rows if r["status"] != "fixed"]
        worst = varying[0] if varying else rows[0]
        status = worst["status"] if varying else "fixed"
        if status == "FAIL":
            all_ok = False
            n_fail += 1
        t.add_row(
            nc.stem, worst["param"],
            "—" if worst["rhat"] is None else f"{worst['rhat']:.4f}",
            "—" if worst["ess_bulk"] is None else f"{worst['ess_bulk']:.0f}",
            status, style=style.get(status),
        )
    console = Console()
    console.print(t)
    console.print(
        f"{len(nc_files) - n_fail}/{len(nc_files)} components pass "
        "(use --param-filter to narrow; full per-component tables: "
        "point at one component .nc)"
    )
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


def _discover_trace_targets() -> list[Path]:
    """Trace sidecars (*.nc) and cut stage2 component dirs under ./results*/."""
    targets: list[Path] = []
    for p in sorted(Path(".").glob("results*/**/*")):
        if (
            p.is_file()
            and p.suffix == ".nc"
            and not any(part.endswith("_stage2_traces") for part in p.parts)
        ) or p.is_dir() and p.name.endswith("_stage2_traces"):
            targets.append(p)
    return targets


def _interactive_traces_setup(args: argparse.Namespace) -> argparse.Namespace:
    """Fill nc_path by prompting when omitted; explicit path bypasses."""
    if args.nc_path is not None:
        return args
    if not sys.stdin.isatty():
        raise ConfigError("no nc_path given and not a terminal; pass a path")

    targets = _discover_trace_targets()
    if not targets:
        raise ConfigError("No trace sidecars found under ./results*/; "
                          "run with --save-traces first")

    console = Console(stderr=True)
    console.print("[bold]Traces:[/bold]")
    for i, p in enumerate(targets, 1):
        label = f"{p}  (per-component)" if p.is_dir() else str(p)
        console.print(f"  {i}) {label}")
    idx = int(
        Prompt.ask("Traces", choices=[str(i) for i in range(1, len(targets) + 1)])
    )
    args.nc_path = str(targets[idx - 1])
    return args


def _traces_command(args: argparse.Namespace) -> None:
    """``bpnmf traces`` — folds ``scripts/analyze_traces.py`` (default,
    numeric table) and ``scripts/make_trace_plots.py`` (``--plots``, PNGs)."""
    args = _interactive_traces_setup(args)
    nc_path = Path(args.nc_path)
    if not nc_path.exists():
        logger.error(f"File not found: {nc_path}")
        sys.exit(1)

    if nc_path.is_dir():
        # cut stage2 component dir -> per-component summary table
        all_ok = _print_component_summary(nc_path, args.param_filter)
        sys.exit(0 if all_ok else 1)

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
        "nc_path",
        nargs="?",
        default=None,
        help="NetCDF traces file, or a *_stage2_traces/ dir for a cut "
        "per-component summary. Omit (in a terminal) for an interactive picker.",
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
