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
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

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


def _aggregate_treated_draws(draws_df: pd.DataFrame) -> pd.DataFrame:
    """Create an 'All treated states' synthetic unit summing over (unit, time) pairs
    where treatment == 1.

    With staggered adoption the number of included units varies by time period.
    outcome, ypred, denominator are summed (count data).
    mu and mu_treated are log-sum-exp'd so exp(mu_agg) == sum_units(exp(mu)).
    """
    tx = draws_df[draws_df["treatment"] == 1].copy()
    if tx.empty:
        return pd.DataFrame()

    def _logsumexp(x: np.ndarray) -> float:
        m = float(np.max(x))
        return m + float(np.log(np.sum(np.exp(x - m))))

    draw_cols = [c for c in [".draw", ".chain", ".iteration"] if c in tx.columns]
    group_cols = draw_cols + ["time", "group"]

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "outcome": g["outcome"].sum(),
                "ypred": g["ypred"].sum(),
                "denominator": g["denominator"].sum(),
                "treatment": 1,
                "mu": _logsumexp(g["mu"].values),
                "mu_treated": _logsumexp(g["mu_treated"].values),
            }
        )

    agg = tx.groupby(group_cols, sort=False).apply(_agg).reset_index()
    agg["unit"] = "All treated states"
    return agg


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def generate_reports(
    draws_df: pd.DataFrame,
    output_dir: str | Path,
    target_unit: Optional[str] = None,
    groups: Optional[list[str]] = None,
    print_tables: bool = True,
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

    Returns
    -------
    dict with keys: ``summary``, ``per_unit``, ``detail``, ``target_unit``,
    ``groups``, ``figs_dir``.
    """
    matplotlib.use("Agg")

    output_dir = Path(output_dir)
    figs_dir = output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    if target_unit is None:
        target_unit = "All treated states"

    all_groups = list(pd.unique(draws_df["group"]))
    if groups is None:
        groups = all_groups
    else:
        unknown = [g for g in groups if g not in all_groups]
        if unknown:
            raise ValueError(
                f"groups={unknown} not present in draws_df (have {all_groups})"
            )

    logger.info(
        f"Generating reports: target_unit={target_unit!r}, groups={groups}, "
        f"output_dir={output_dir}"
    )

    agg_draws = _aggregate_treated_draws(draws_df)
    draws_df_aug = (
        pd.concat([draws_df, agg_draws], ignore_index=True)
        if not agg_draws.empty
        else draws_df
    )

    quantiles_df = _compute_quantiles(draws_df_aug)
    quantiles_df["treated_unit"] = quantiles_df["unit"] == target_unit
    if not pd.api.types.is_datetime64_any_dtype(quantiles_df["time"]):
        quantiles_df["time"] = pd.to_datetime(quantiles_df["time"])

    target_slug = _slug(target_unit)

    # ---- Per-group unit-level figures ----------------------------------------
    # When only one group, write directly under figs/ (no extra nesting).
    for grp in groups:
        grp_figs_dir = figs_dir / grp if len(groups) > 1 else figs_dir
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
        draws_df_aug, group_var="unit", estimand="ratio", method="mu"
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
    make_all_ppc_plots(draws_df, output_dir=str(ppc_dir))

    # ---- Tables ---------------------------------------------------------------
    summary = make_summary_table(draws_df_aug, target_unit)
    summary.to_csv(figs_dir / "summary_table.csv", index=False)

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
    detail = detail[
        [
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
    ].sort_values(["unit", "time"])
    detail.to_csv(figs_dir / "expected_vs_observed.csv", index=False)

    post = detail[detail["treatment"] == 1].copy()
    per_unit = (
        post.groupby(["unit", "group"])
        .agg(
            n_periods=("time", "size"),
            observed=("observed", "sum"),
            expected_mean=("expected_mean", "sum"),
            expected_lower_95=("expected_lower_95", "sum"),
            expected_upper_95=("expected_upper_95", "sum"),
        )
        .reset_index()
    )
    per_unit["excess"] = per_unit["observed"] - per_unit["expected_mean"]
    per_unit["excess_pct"] = per_unit["excess"] / per_unit["expected_mean"] * 100
    per_unit = per_unit.sort_values("excess_pct", ascending=False)
    per_unit.to_csv(figs_dir / "post_treatment_summary.csv", index=False)

    if print_tables:
        _print_rich_tables(summary, per_unit, draws_df_aug, target_unit)

    return {
        "summary": summary,
        "per_unit": per_unit,
        "detail": detail,
        "target_unit": target_unit,
        "groups": groups,
        "figs_dir": figs_dir,
    }


# -----------------------------------------------------------------------------
# Terminal rendering
# -----------------------------------------------------------------------------


def _print_rich_tables(
    summary: pd.DataFrame,
    per_unit: pd.DataFrame,
    draws_df: pd.DataFrame,
    target_unit: str,
) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Headline summary
    t = Table(title=f"{target_unit} — Observed vs Expected", show_lines=False)
    for col in summary.columns:
        t.add_column(col, justify="right" if col != "Group" else "left")
    for _, row in summary.iterrows():
        t.add_row(*[str(v) for v in row.tolist()])
    console.print(t)

    # Per-unit post-treatment totals — "All treated states" pinned to top
    _AGG = "All treated states"
    agg_rows = per_unit[per_unit["unit"] == _AGG]
    other_rows = per_unit[per_unit["unit"] != _AGG]
    per_unit_ordered = pd.concat([agg_rows, other_rows], ignore_index=True)

    t = Table(
        title="Post-treatment totals by unit (ranked by % excess)",
        show_lines=False,
    )
    t.add_column("Unit", justify="left")
    t.add_column("Periods", justify="right")
    t.add_column("Observed", justify="right")
    t.add_column("Expected (95% CI)", justify="right")
    t.add_column("Excess", justify="right")
    t.add_column("Excess %", justify="right")
    for i, (_, r) in enumerate(per_unit_ordered.iterrows()):
        style = "bold green" if r["unit"] == target_unit else None
        exp_ci = (
            f"{r['expected_mean']:,.0f} "
            f"({r['expected_lower_95']:,.0f}, {r['expected_upper_95']:,.0f})"
        )
        t.add_row(
            str(r["unit"]),
            f"{int(r['n_periods'])}",
            f"{r['observed']:,.0f}",
            exp_ci,
            f"{r['excess']:+,.0f}",
            f"{r['excess_pct']:+.2f}%",
            style=style,
        )
        if i == len(agg_rows) - 1 and not agg_rows.empty:
            t.add_section()
    console.print(t)

    # Headline draws-based effect
    tx = draws_df[(draws_df["unit"] == target_unit) & (draws_df["treatment"] == 1)]
    by_draw = tx.groupby(".draw").agg(
        observed=("outcome", "sum"), expected=("ypred", "sum")
    )
    by_draw["excess"] = by_draw["observed"] - by_draw["expected"]
    by_draw["excess_pct"] = by_draw["excess"] / by_draw["expected"] * 100
    ex = by_draw["excess"]
    pc = by_draw["excess_pct"]
    t = Table(title=f"Headline Treatment Effect — {target_unit}", show_lines=False)
    t.add_column("Metric")
    t.add_column("Estimate (95% CI)", justify="right")
    t.add_row(
        "Excess births",
        f"{ex.mean():,.0f} ({np.quantile(ex, 0.025):,.0f}, "
        f"{np.quantile(ex, 0.975):,.0f})",
    )
    t.add_row(
        "Percent change",
        f"{pc.mean():.1f}% ({np.quantile(pc, 0.025):.1f}%, "
        f"{np.quantile(pc, 0.975):.1f}%)",
    )
    console.print(t)
