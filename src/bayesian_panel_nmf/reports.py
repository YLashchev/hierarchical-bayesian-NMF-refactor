"""Generate figures and summary tables from MCMC posterior draws.

Shared entry point used by both `bpnmf run` (when
``output.figures: true`` in config) and `bpnmf viz`
(for standalone regeneration from a results CSV).

Outputs placed under ``<output_dir>/figs/``:

  - fit_<target>.png / gap_<target>.png       per-unit diagnostics
  - interval.png                               causal intervals (% change)
  - raw_rate.png / group_comparison.png        descriptive panel plots
  - ppc/ppc_*.png + ppc_pvalues.csv            posterior predictive checks
  - summary_table.csv                          headline target effect
  - expected_vs_observed.csv                   per-(unit, time) detail
  - post_treatment_summary.csv                 per-unit post-treatment totals
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
from loguru import logger

from .aggregate_units import add_aggregate_units
from .plots import (
    PLOT_REGISTRY,
    make_all_ppc_plots,
    make_group_comparison_plot,
    make_interval_plot,
    make_raw_rate_plot,
    make_unit_fit_plot,
    make_unit_gap_plot,
)
from .tables import (
    _auto_detect_target,
    _compute_per_unit_post_treatment,
    _compute_quantiles,
    _print_rich_tables,
    _slug,
    make_summary_table,
)

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def generate_reports(
    draws_df: pd.DataFrame,
    output_dir: str | Path,
    target_unit: str | None = None,
    groups: list[str] | None = None,
    print_tables: bool = True,
    print_target_table: bool = True,
    aggregate_units: list[dict] | None = None,
    ppc_units: list[str] | None = None,
    ppc_acf_lags: list[int] | None = None,
    ppc_unit_corr_max_time: str | None = None,
    ppc_exclude_units: list[str] | None = None,
    ppc_draws_df: pd.DataFrame | None = None,
    figures: list[str] | None = None,
) -> dict:
    """Generate figures and tables from posterior draws.

    For each outcome group present in ``draws_df`` (or each group in
    ``groups`` if provided), per-unit plots (fit, gap, raw_rate) are
    written under ``figs/<group>/`` so every group defined in the config
    gets its own artifacts. Cross-group plots/tables (interval,
    group_comparison, ppc/, summary_table, expected_vs_observed,
    post_treatment_summary) live directly under ``figs/``.

    Parameters
    ----------
    draws_df : pd.DataFrame
        Posterior draws in long format (output of ``format_draws``).
        Required columns: ``.draw, .chain, unit, time, group, outcome,
        denominator, treatment, ypred, mu, mu_treated``.
    output_dir : str or Path
        Directory for this model run. Figures go to ``<output_dir>/figs/``.
    target_unit : str, optional
        Unit for per-unit and headline plots/tables. Auto-detected if ``None``.
    groups : list of str, optional
        Subset of groups to plot per-unit artifacts for. Defaults to every
        unique group present in ``draws_df``.
    print_tables : bool, default True
        Render rich tables to the terminal.
    print_target_table : bool, default True
        Print the target-unit headline table (Table 1). Ignored if
        ``print_tables`` is ``False``.
    ppc_draws_df : pd.DataFrame, optional
        Alternative posterior product for the PPC suite only (cut mode: the
        full Stage-1 posterior with untreated ypred). All other figures and
        tables continue to use ``draws_df``. Reporting-only aggregate units
        are applied to it before plotting.
    figures : list of str, optional
        Subset of ``PLOT_REGISTRY`` names to render (``"unit_fit", "unit_gap",
        "raw_rate", "interval", "group_comparison", "ppc"``). ``None``
        (default) renders every registry entry, matching the pre-selection
        behavior. An empty list renders none. Unknown names raise
        ``ValueError``. Tables (summary_table, expected_vs_observed,
        post_treatment_summary) are always written regardless of this
        selection -- only PLOT_REGISTRY figures are gated.

    Returns
    -------
    dict with keys: ``summary``, ``per_unit``, ``detail``, ``target_unit``,
    ``groups``, ``figs_dir``, ``treated_units``.
    """
    if figures is None:
        selected_figures = set(PLOT_REGISTRY)
    else:
        unknown_figures = [name for name in figures if name not in PLOT_REGISTRY]
        if unknown_figures:
            raise ValueError(
                f"figures contains unknown name(s) {unknown_figures}; valid "
                f"names are {sorted(PLOT_REGISTRY)}"
            )
        selected_figures = set(figures)
    # Lazy import: AGENTS.md forbids top-level matplotlib/pyplot outside
    # plots.py. matplotlib is already loaded transitively via
    # ``from .plots import ...`` above; this binds local names for
    # the Agg backend call and ``plt.close()`` calls below.
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    output_dir = Path(output_dir)
    figs_dir = output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    draws_for_reporting = add_aggregate_units(draws_df, aggregate_units or [])

    if target_unit is None:
        target_unit = _auto_detect_target(draws_df)
    if target_unit is None:
        raise ValueError("No treated units in draws_df and no target_unit specified.")

    all_groups = (
        cast(pd.Series, draws_for_reporting["group"]).drop_duplicates().tolist()
    )
    if groups is None:
        report_groups = all_groups
    else:
        unknown = [g for g in groups if g not in all_groups]
        if unknown:
            raise ValueError(
                f"groups={unknown} not present in draws_df (have {all_groups})"
            )
        report_groups = groups

    logger.info(
        f"Generating reports: target_unit={target_unit!r}, groups={report_groups}, "
        f"output_dir={output_dir}"
    )

    quantiles_df = _compute_quantiles(draws_for_reporting)
    quantiles_df["treated_unit"] = quantiles_df["unit"] == target_unit
    if not pd.api.types.is_datetime64_any_dtype(quantiles_df["time"]):
        quantiles_df["time"] = pd.to_datetime(quantiles_df["time"])

    target_slug = _slug(target_unit)

    # ---- Per-group unit-level figures ----------------------------------------
    # When only one group, write directly under figs/ (no extra nesting).
    for grp in report_groups:
        grp_figs_dir = figs_dir / grp if len(report_groups) > 1 else figs_dir
        grp_figs_dir.mkdir(parents=True, exist_ok=True)

        if "unit_fit" in selected_figures:
            logger.debug(f"  [{grp}] fit")
            fig, _ = make_unit_fit_plot(quantiles_df, target_unit, group=grp)
            fig.savefig(
                grp_figs_dir / f"fit_{target_slug}.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

        if "unit_gap" in selected_figures:
            logger.debug(f"  [{grp}] gap")
            fig, _ = make_unit_gap_plot(quantiles_df, target_unit, group=grp)
            fig.savefig(
                grp_figs_dir / f"gap_{target_slug}.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

        if "raw_rate" in selected_figures:
            logger.debug(f"  [{grp}] raw_rate")
            fig, _ = make_raw_rate_plot(draws_df, group=grp, separate_unit=target_unit)
            fig.savefig(grp_figs_dir / "raw_rate.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ---- Cross-group figures (interval, comparison, PPC) ---------------------
    if "interval" in selected_figures:
        logger.debug("  interval (percent change)")
        fig, ax = make_interval_plot(
            draws_df, group_var="unit", estimand="ratio", method="mu"
        )
        ax.set_xlabel("Percent Change (%)", fontsize=12)
        fig.savefig(figs_dir / "interval.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if "group_comparison" in selected_figures:
        logger.debug("  group_comparison")
        fig, _ = make_group_comparison_plot(draws_df)
        fig.savefig(figs_dir / "group_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if "ppc" in selected_figures:
        logger.debug("  ppc/*")
        ppc_dir = figs_dir / "ppc"
        ppc_dir.mkdir(parents=True, exist_ok=True)
        if ppc_draws_df is not None:
            ppc_source = add_aggregate_units(ppc_draws_df, aggregate_units or [])
        else:
            ppc_source = draws_for_reporting
        make_all_ppc_plots(
            ppc_source,
            output_dir=str(ppc_dir),
            acf_lags=ppc_acf_lags or [6],
            max_treat_date=ppc_unit_corr_max_time,
            ppc_units=ppc_units,
            ppc_exclude_units=ppc_exclude_units,
        )

    # ---- Tables ---------------------------------------------------------------
    summary = make_summary_table(draws_for_reporting, target_unit)
    summary.to_csv(figs_dir / "summary_table.csv", index=False)

    # Per-treated-state summary tables (matches JAMA supplement structure)
    treated_units = (
        cast(pd.Series, draws_df.loc[draws_df["treatment"] == 1, "unit"])
        .unique()
        .tolist()
    )
    for tu in treated_units:
        tu_summary = make_summary_table(draws_for_reporting, tu)
        tu_summary.to_csv(figs_dir / f"summary_table_{_slug(tu)}.csv", index=False)

    detail = quantiles_df.rename(
        columns={
            "outcome": "observed",
            "ypred_mean": "expected_mean",
            "ypred_median": "expected_median",
            "ypred_lower": "expected_lower_95",
            "ypred_upper": "expected_upper_95",
        }
    )
    detail["gap"] = detail["observed"] - detail["expected_mean"]
    detail["gap_pct"] = detail["gap"] / detail["expected_mean"] * 100
    detail_cols = [
        "unit",
        "time",
        "group",
        "treatment",
        "treated_unit",
        "observed",
        "expected_mean",
        "expected_median",
        "expected_lower_95",
        "expected_upper_95",
        "gap",
        "gap_pct",
    ]
    detail = cast(
        pd.DataFrame,
        detail.loc[:, detail_cols].sort_values(by=["unit", "time"]),
    )
    detail.to_csv(figs_dir / "expected_vs_observed.csv", index=False)

    per_unit = _compute_per_unit_post_treatment(
        draws_for_reporting, figs_dir / "post_treatment_summary.csv"
    )

    if print_tables:
        _print_rich_tables(
            summary, per_unit, draws_for_reporting, target_unit, print_target_table
        )

    return {
        "summary": summary,
        "per_unit": per_unit,
        "detail": detail,
        "target_unit": target_unit,
        "groups": report_groups,
        "figs_dir": figs_dir,
        "treated_units": treated_units,
    }
