"""Generate MCMC trace plots from draws CSV or NetCDF trace sidecar.

When given a NetCDF sidecar (produced with --save-traces), all latent
parameters are available (te, unit_fe, time_fac, treatment_state_scale, etc.).

When given a CSV (fallback), only marginal columns saved to disk are available
(mu, mu_treated, ypred).  A random subset of cells is traced to avoid OOM.

Usage:
    # From NetCDF sidecar (preferred — full latent access):
    uv run python scripts/make_trace_plots.py results/total/NB_births_total_3_traces.nc

    # Filter to specific parameters:
    uv run python scripts/make_trace_plots.py results/total/NB_births_total_3_traces.nc \\
        --param-filter te,state_treatment_effect,treatment_state_scale

    # From CSV (marginals only):
    uv run python scripts/make_trace_plots.py results/total/NB_births_total_3.csv
    uv run python scripts/make_trace_plots.py results/total/NB_births_total_3.csv \\
        --param-filter mu --num-vars 8 --seed 42
"""

import argparse
import sys
from pathlib import Path

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

matplotlib.use("Agg")  # headless — no display needed


def _load_and_reshape(
    csv_path: str,
    param_filters: list[str],
) -> tuple[dict[str, np.ndarray], int, int]:
    """Load selected CSV columns and reshape to (chains, draws, cells)."""
    import pandas as pd

    header = pd.read_csv(csv_path, nrows=0).columns
    value_cols = [
        col
        for col in ["mu", "mu_treated", "ypred"]
        if col in header and any(col.startswith(f) for f in param_filters)
    ]
    if not value_cols:
        return {}, 0, 0

    usecols = [".chain", ".iteration", *value_cols]
    df = pd.read_csv(
        csv_path,
        usecols=lambda col: col in usecols,
        dtype={".chain": int, ".iteration": int},
    )
    chains = df[".chain"].nunique()
    draws_per_chain = df[".iteration"].nunique()
    params: dict[str, np.ndarray] = {}
    for col in value_cols:
        grouped = df.groupby([".chain", ".iteration"], sort=True)[col].apply(
            lambda s: s.values
        )
        n_obs = len(grouped.iloc[0])
        arr = np.stack(grouped.to_list()).reshape(chains, draws_per_chain, n_obs)
        params[col] = arr
    return params, chains, draws_per_chain


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

    if param_filters:
        vars_to_plot = [
            v for v in all_vars if any(v.startswith(f) for f in param_filters)
        ]
        if not vars_to_plot:
            logger.warning(
                f"No parameters matched filters {param_filters}. Available: {all_vars}"
            )
            vars_to_plot = all_vars[:5]  # fallback: first 5
    else:
        # Default: plot scalar/low-dim params first (treatment effects, scales)
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

    logger.info(f"Tracing parameters: {vars_to_plot}")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    rng = np.random.default_rng(42)
    for var in vars_to_plot:
        logger.info(f"Plotting {var} ...")
        try:
            da = posterior[var]
            # For high-dim params, subset each non-chain/draw dim so that the
            # total number of subplots (product of all dim sizes) stays ≤ max_subplots.
            max_subplots = 20
            param_dims = [d for d in da.dims if d not in ("chain", "draw")]
            coords: dict = {}
            if param_dims:
                # Compute per-cell variance across chain+draw. Cells that are
                # structurally 0 (e.g. `te` in control units/pre-treatment times)
                # have zero variance and produce uninformative flatline traces.
                # Prefer cells with actual posterior variability.
                cell_var = da.var(dim=("chain", "draw"))
                nonzero_mask = cell_var > 1e-12
                n_nonzero = int(nonzero_mask.sum())
                n_total = int(cell_var.size)
                logger.info(
                    f"  {var}: {n_nonzero}/{n_total} cells have non-zero variance"
                )
                if n_nonzero == 0:
                    logger.warning(
                        f"Skipped {var}: all cells structurally zero (flatline)."
                    )
                    continue

                if n_nonzero < n_total:
                    # Pick coords from informative (non-flatline) cells only.
                    # `te` is structurally 0 on control cells, so random sampling
                    # across the full grid almost always produces flatlines.
                    # Per dim, pick up to `per_dim` unique indices from cells
                    # where at least one slice has non-zero variance.
                    per_dim = max(1, int(max_subplots ** (1.0 / len(param_dims))))
                    for axis, dim in enumerate(param_dims):
                        # Indices along this dim that contain at least one
                        # informative cell (collapse other dims with .any()).
                        other_axes = tuple(
                            i for i in range(len(param_dims)) if i != axis
                        )
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
                            f"  {var}: dim '{dim}' → {k} informative coords of {informative_idx.size}"
                        )
                else:
                    # All cells informative: uniform random sample across dims.
                    per_dim = max(1, int(max_subplots ** (1.0 / len(param_dims))))
                    for dim in param_dims:
                        size = da.sizes[dim]
                        if size > per_dim:
                            chosen = sorted(
                                rng.choice(size, size=per_dim, replace=False).tolist()
                            )
                            coords[dim] = da[dim].values[chosen]
                            logger.debug(
                                f"  {var}: subsampling dim '{dim}' {size}→{per_dim}"
                            )
            az.plot_trace(
                idata, var_names=[var], coords=coords or None, backend="matplotlib"
            )
            fig = plt.gcf()
            suffix = f" (subset {max_subplots})" if coords else ""
            fig.suptitle(f"Trace — {var}{suffix}", fontsize=11, y=1.01)
            fig.tight_layout()
            out_path = out_dir / f"trace_{var}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved {out_path}")
            saved.append(out_path)
        except Exception as e:
            logger.warning(f"Skipped {var}: {e}")
            plt.close("all")

    return saved


def _make_trace_plots_from_csv(
    csv_path: Path,
    param_filters: list[str],
    num_vars: int,
    seed: int,
    out_dir: Path,
) -> list[Path]:
    """Load draws CSV, subset cells, generate trace plots, save PNGs."""
    logger.info(f"Loading CSV: {csv_path}")
    params, chains, draws_per_chain = _load_and_reshape(str(csv_path), param_filters)

    if not params:
        logger.error("No matching columns found — check --param-filter")
        return []

    logger.info(f"Chains: {chains}, draws/chain: {draws_per_chain}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    saved: list[Path] = []

    for name, arr in params.items():
        # arr shape: (chains, draws, n_obs)
        n_obs = arr.shape[2]
        k = min(num_vars, n_obs)
        idx = rng.choice(n_obs, size=k, replace=False)
        idx.sort()

        posterior: dict[str, np.ndarray] = {
            f"{name}[{i}]": arr[:, :, j] for j, i in enumerate(idx)
        }

        dt = az.from_dict({"posterior": posterior})  # nested dict per ArviZ >=0.19 API
        az.plot_trace(dt, backend="matplotlib")
        fig = plt.gcf()
        fig.suptitle(
            f"Trace — {name}  ({k} of {n_obs} random cells, seed={seed})",
            fontsize=11,
            y=1.01,
        )
        fig.tight_layout()

        out_path = out_dir / f"trace_{name}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved {out_path}  ({k} cells)")
        saved.append(out_path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MCMC trace plots from draws CSV or NetCDF trace sidecar"
    )
    parser.add_argument(
        "input",
        help="Path to draws CSV or NetCDF sidecar (.nc). "
        "NetCDF gives access to full latent parameters.",
    )
    parser.add_argument(
        "--param-filter",
        default=None,
        help=(
            "Comma-separated parameter name(s) to trace. "
            "NetCDF: exact variable names (e.g. te,treatment_state_scale). "
            "CSV: column prefixes (e.g. mu,ypred). "
            "Default: auto-select treatment effects (NetCDF) or mu (CSV)."
        ),
    )
    parser.add_argument(
        "--num-vars",
        type=int,
        default=5,
        help="[CSV only] Number of random cells to trace per parameter (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="[CSV only] RNG seed for reproducible cell selection (default: 0)",
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

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else input_path.parent / "figs" / "diagnostics"
    )

    filters = (
        [f.strip() for f in args.param_filter.split(",")] if args.param_filter else None
    )

    if input_path.suffix == ".nc":
        saved = _make_trace_plots_from_netcdf(input_path, filters, out_dir)
    else:
        # CSV path: default to mu if no filter given
        saved = _make_trace_plots_from_csv(
            input_path, filters or ["mu"], args.num_vars, args.seed, out_dir
        )

    if not saved:
        sys.exit(1)

    print(f"\nTrace plots written ({len(saved)} file(s)):")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
