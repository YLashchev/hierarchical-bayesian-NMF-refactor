"""Generate figures and summary tables from MCMC posterior draws.

Shared entry point used by both `scripts/run_analysis.py` (when
``output.figures: true`` in config) and `scripts/generate_full_viz.py`
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
from typing import Optional, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from .aggregate_units import add_aggregate_units
from .visualization import (
    make_all_ppc_plots,
    make_group_comparison_plot,
    make_interval_plot,
    make_raw_rate_plot,
    make_summary_table,
    make_unit_fit_plot,
    make_unit_gap_plot,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _compute_quantiles(draws_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate posterior draws to mean/median/95% CI per (unit, time, group)."""
    return (
        draws_df.groupby(["unit", "time", "group", "outcome", "treatment"])["ypred"]
        .agg(
            ypred_mean="mean",
            ypred_lower=lambda x: np.percentile(x, 2.5),
            ypred_upper=lambda x: np.percentile(x, 97.5),
            ypred_median="median",
        )
        .reset_index()
    )


def _auto_detect_target(draws_df: pd.DataFrame) -> Optional[str]:
    """Pick the unit with the most post-treatment observations."""
    treated = draws_df[draws_df["treatment"] == 1]
    if treated.empty:
        return None
    # Count unique (time) per unit to avoid inflating by draws
    counts = treated.groupby("unit")["time"].nunique()
    return str(counts.idxmax())


def _slug(s: str) -> str:
    return s.replace(" ", "_").lower()


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def generate_reports(
    draws_df: pd.DataFrame,
    output_dir: str | Path,
    target_unit: Optional[str] = None,
    groups: Optional[list[str]] = None,
    print_tables: bool = True,
    print_target_table: bool = True,
    aggregate_units: Optional[list[dict]] = None,
    ppc_units: Optional[list[str]] = None,
    ppc_acf_lags: Optional[list[int]] = None,
    ppc_unit_corr_max_time: Optional[str] = None,
    ppc_exclude_units: Optional[list[str]] = None,
) -> dict:
    """Generate all figures and tables from posterior draws.

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

    Returns
    -------
    dict with keys: ``summary``, ``per_unit``, ``detail``, ``target_unit``,
    ``groups``, ``figs_dir``, ``treated_units``.
    """
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
        logger.debug(f"  [{grp}] fit + gap + raw_rate")

        fig, _ = make_unit_fit_plot(quantiles_df, target_unit, group=grp)
        fig.savefig(
            grp_figs_dir / f"fit_{target_slug}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        fig, _ = make_unit_gap_plot(quantiles_df, target_unit, group=grp)
        fig.savefig(
            grp_figs_dir / f"gap_{target_slug}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        fig, _ = make_raw_rate_plot(draws_df, group=grp, separate_unit=target_unit)
        fig.savefig(grp_figs_dir / "raw_rate.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- Cross-group figures (interval, comparison, PPC) ---------------------
    logger.debug("  interval (percent change)")
    fig, ax = make_interval_plot(
        draws_df, group_var="unit", estimand="ratio", method="mu"
    )
    ax.set_xlabel("Percent Change (%)", fontsize=12)
    fig.savefig(figs_dir / "interval.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.debug("  group_comparison")
    fig, _ = make_group_comparison_plot(draws_df)
    fig.savefig(figs_dir / "group_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.debug("  ppc/*")
    ppc_dir = figs_dir / "ppc"
    ppc_dir.mkdir(parents=True, exist_ok=True)
    make_all_ppc_plots(
        draws_for_reporting,
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


# -----------------------------------------------------------------------------
# Per-unit post-treatment table (mu-based, matches reference make_fertility_table)
# -----------------------------------------------------------------------------


def _compute_per_unit_post_treatment(
    draws_df: pd.DataFrame, csv_path: Path
) -> pd.DataFrame:
    """Per-(unit, group) post-treatment totals using mu-based counterfactual.

    Matches reference R ``make_fertility_table``:
        expected  = sum(exp(mu))        # counterfactual without treatment
        excess    = observed - expected

    CI is the draw-level distribution of sum(exp(mu)).
    """
    post = draws_df[draws_df["treatment"] == 1].copy()
    if post.empty:
        empty = pd.DataFrame(
            columns=[
                "unit",
                "group",
                "n_periods",
                "observed",
                "expected_mean",
                "expected_lower_95",
                "expected_upper_95",
                "excess_mean",
                "excess_lower_95",
                "excess_upper_95",
                "excess_pct_mean",
                "excess_pct_lower_95",
                "excess_pct_upper_95",
            ]
        )
        empty.to_csv(csv_path, index=False)
        return empty

    def _draw_sums(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "expected": float(np.sum(np.exp(g["mu"].to_numpy(dtype=float)))),
            }
        )

    draw_sums = cast(
        pd.DataFrame,
        post.groupby(["unit", "group", ".draw"], observed=True)
        .apply(_draw_sums, include_groups=False)
        .reset_index(),
    )

    # observed is constant across draws per (unit, group, time);
    # drop duplicate time rows to avoid double-counting, then sum.
    observed_totals = cast(
        pd.DataFrame,
        post.drop_duplicates(["unit", "group", "time"])
        .groupby(["unit", "group"], observed=True)
        .agg(n_periods=("time", "nunique"), observed=("outcome", "sum"))
        .reset_index(),
    )

    # Merge observed for per-draw excess and excess_pct (proper posterior uncertainty)
    draw_sums = draw_sums.merge(
        observed_totals[["unit", "group", "observed"]],
        on=["unit", "group"],
        how="left",
    )
    draw_sums["excess"] = draw_sums["observed"] - draw_sums["expected"]
    draw_sums["excess_pct"] = draw_sums["excess"] / draw_sums["expected"] * 100

    stats = cast(
        pd.DataFrame,
        draw_sums.groupby(["unit", "group"], observed=True)
        .agg(
            expected_mean=("expected", "mean"),
            expected_lower_95=("expected", lambda x: float(np.quantile(x, 0.025))),
            expected_upper_95=("expected", lambda x: float(np.quantile(x, 0.975))),
            excess_mean=("excess", "mean"),
            excess_lower_95=("excess", lambda x: float(np.quantile(x, 0.025))),
            excess_upper_95=("excess", lambda x: float(np.quantile(x, 0.975))),
            excess_pct_mean=("excess_pct", "mean"),
            excess_pct_lower_95=("excess_pct", lambda x: float(np.quantile(x, 0.025))),
            excess_pct_upper_95=("excess_pct", lambda x: float(np.quantile(x, 0.975))),
        )
        .reset_index(),
    )

    per_unit = cast(
        pd.DataFrame, observed_totals.merge(stats, on=["unit", "group"], how="left")
    )
    per_unit = cast(
        pd.DataFrame, per_unit.sort_values("excess_pct_mean", ascending=False)
    )
    per_unit = per_unit[
        [
            "unit",
            "group",
            "n_periods",
            "observed",
            "expected_mean",
            "expected_lower_95",
            "expected_upper_95",
            "excess_mean",
            "excess_lower_95",
            "excess_upper_95",
            "excess_pct_mean",
            "excess_pct_lower_95",
            "excess_pct_upper_95",
        ]
    ]
    per_unit.to_csv(csv_path, index=False)
    return cast(pd.DataFrame, per_unit)


# -----------------------------------------------------------------------------
# Terminal rendering
# -----------------------------------------------------------------------------


def _print_rich_tables(
    summary: pd.DataFrame,
    per_unit: pd.DataFrame,
    draws_df: pd.DataFrame,
    target_unit: str,
    print_target_table: bool = True,
) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Headline summary (Table 1)
    if print_target_table:
        t = Table(title=f"{target_unit} — Observed vs Expected", show_lines=False)
        for col in summary.columns:
            t.add_column(col, justify="right" if col != "Group" else "left")
        for _, row in summary.iterrows():
            t.add_row(*[str(v) for v in row.tolist()])
        console.print(t)

    # Per-unit post-treatment totals (Table 2)
    t = Table(
        title="Post-treatment totals by unit (ranked by % excess)",
        show_lines=False,
    )
    t.add_column("Unit", justify="left")
    t.add_column("Group", justify="left")
    t.add_column("Periods", justify="right")
    t.add_column("Observed", justify="right")
    t.add_column("Expected (95% CI)", justify="right")
    t.add_column("Excess (95% CI)", justify="right")
    t.add_column("Excess % (95% CI)", justify="right")
    for _, r in per_unit.iterrows():
        row = r.to_dict()
        style = "bold green" if row["unit"] == target_unit else None
        exp_ci = (
            f"{row['expected_mean']:,.0f} "
            f"({row['expected_lower_95']:,.0f}, {row['expected_upper_95']:,.0f})"
        )
        excess_ci = (
            f"{row['excess_mean']:+,.0f} "
            f"({row['excess_lower_95']:+,.0f}, {row['excess_upper_95']:+,.0f})"
        )
        excess_pct_ci = (
            f"{row['excess_pct_mean']:+.2f}% "
            f"({row['excess_pct_lower_95']:+.2f}%, {row['excess_pct_upper_95']:+.2f}%)"
        )
        t.add_row(
            str(row["unit"]),
            str(row["group"]),
            f"{int(row['n_periods'])}",
            f"{row['observed']:,.0f}",
            exp_ci,
            excess_ci,
            excess_pct_ci,
            style=style,
        )
    console.print(t)
