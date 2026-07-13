"""Generate MCMC trace plots from a NetCDF trace sidecar.

Requires a NetCDF sidecar (produced with --save-traces), giving access to
all latent parameters (te, unit_fe, time_fac, treatment_state_scale, etc.).

Usage:
    uv run python scripts/make_trace_plots.py results/total/NB_births_total_3_traces.nc

    # Filter to specific parameters:
    uv run python scripts/make_trace_plots.py results/total/NB_births_total_3_traces.nc \\
        --param-filter te,state_treatment_effect,treatment_state_scale
"""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    import xarray as xr

matplotlib.use("Agg")  # headless — no display needed

_MAX_TRACE_SUBPLOTS = 20


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
    da: "xr.DataArray", max_subplots: int, rng: np.random.Generator
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
            chosen = sorted(
                rng.choice(informative_idx, size=k, replace=False).tolist()
            )
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MCMC trace plots from a NetCDF trace sidecar"
    )
    parser.add_argument(
        "input",
        help="Path to NetCDF trace sidecar (.nc), giving access to full latent parameters.",
    )
    parser.add_argument(
        "--param-filter",
        default=None,
        help=(
            "Comma-separated exact variable name(s) to trace "
            "(e.g. te,treatment_state_scale). "
            "Default: auto-select treatment effects."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <input_dir>/figs/diagnostics/)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    if input_path.suffix != ".nc":
        sys.exit(
            f"Expected a NetCDF traces file (*.nc), got {input_path}. "
            "Run with --save-traces."
        )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else input_path.parent / "figs" / "diagnostics"
    )

    filters = (
        [f.strip() for f in args.param_filter.split(",")] if args.param_filter else None
    )

    saved = _make_trace_plots_from_netcdf(input_path, filters, out_dir)

    if not saved:
        sys.exit(1)

    print(f"\nTrace plots written ({len(saved)} file(s)):")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
