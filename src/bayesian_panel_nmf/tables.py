"""Table computation and terminal rendering for MCMC posterior draws.

No matplotlib here — figure orchestration lives in ``reports.py``, plotting
primitives live in ``plots.py``. This module holds the pure pandas/numpy
table math (``make_summary_table``, quantile aggregation, per-unit
post-treatment totals) and rich-terminal rendering (``_print_rich_tables``,
``print_run_summary_panel``). rich is imported lazily inside the render
functions, matching the pattern already used for terminal output elsewhere
in this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from loguru import logger

from .plots import _detect_outcome_column, _standardize_columns

# -----------------------------------------------------------------------------
# Report-level helpers (draws aggregation)
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


def _auto_detect_target(draws_df: pd.DataFrame) -> str | None:
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
# Headline summary table (observed vs expected, by group)
# -----------------------------------------------------------------------------


def make_summary_table(
    merged_df: pd.DataFrame,
    target_unit: str,
    outcome_col: str = "outcome",
    denom_col: str = "denominator",
    rate_normalizer: float = 1000.0,
) -> pd.DataFrame:
    """
    Generate summary table for observed vs expected outcomes.

    Returns a pandas DataFrame.

    Parameters
    ----------
    merged_df : pd.DataFrame
    target_unit : str
        Unit to summarize.
    outcome_col : str
    denom_col : str
    rate_normalizer : float
    """
    df = _standardize_columns(merged_df)

    if outcome_col not in df.columns and any(
        col in df.columns for col in ("outcome", "births", "count", "y")
    ):
        outcome_col = _detect_outcome_column(df)

    if denom_col not in df.columns and "population" in df.columns:
        denom_col = "population"

    df_tgt = df[(df["unit"] == target_unit) & (df["treatment"] == 1)].copy()

    if df_tgt.empty:
        return pd.DataFrame()

    if "start_date" in df.columns and "end_date" in df.columns:
        df_tgt["years"] = (
            pd.to_datetime(df_tgt["end_date"]) - pd.to_datetime(df_tgt["start_date"])
        ).dt.days / 365.25
    else:
        df_tgt["years"] = 1.0

    def compute_stats(g):
        stats = {
            "ypred": np.sum(g["ypred"]),
            "outcome": np.sum(g[outcome_col]),
            "treated": np.sum(np.exp(g["mu_treated"]))
            if "mu_treated" in g.columns
            else np.nan,
            "untreated": np.sum(np.exp(g["mu"])) if "mu" in g.columns else np.nan,
            "denom_val": np.sum(g[denom_col] * g["years"]),
        }
        return pd.Series(stats)

    draw_stats = (
        df_tgt.groupby(["group", ".draw"])
        .apply(compute_stats, include_groups=False)
        .reset_index()
    )

    draw_stats["treated_rate"] = (
        draw_stats["treated"] / draw_stats["denom_val"] * rate_normalizer
    )
    draw_stats["untreated_rate"] = (
        draw_stats["untreated"] / draw_stats["denom_val"] * rate_normalizer
    )
    draw_stats["outcome_rate"] = (
        draw_stats["outcome"] / draw_stats["denom_val"] * rate_normalizer
    )
    draw_stats["outcome_diff"] = draw_stats["treated"] - draw_stats["untreated"]

    summary = []
    for grp in draw_stats["group"].unique():
        gd = draw_stats[draw_stats["group"] == grp]

        outcome_mean = gd["outcome"].mean()
        denom_mean = gd["denom_val"].mean()
        outcome_rate = gd["outcome_rate"].mean()

        diff = gd["outcome_diff"]
        diff_mean, diff_l, diff_u = (
            diff.mean(),
            np.quantile(diff, 0.025),
            np.quantile(diff, 0.975),
        )

        rate_diff = gd["treated_rate"] - gd["untreated_rate"]
        rd_mean, rd_l, rd_u = (
            rate_diff.mean(),
            np.quantile(rate_diff, 0.025),
            np.quantile(rate_diff, 0.975),
        )

        ratio = gd["treated_rate"] / gd["untreated_rate"]
        pct = 100 * (ratio - 1)
        pct_mean, pct_l, pct_u = (
            pct.mean(),
            np.quantile(pct, 0.025),
            np.quantile(pct, 0.975),
        )

        # two-sided pval
        pval = 2 * min(
            np.mean(gd["untreated"] > gd["treated"]),
            np.mean(gd["untreated"] < gd["treated"]),
        )

        sig = "*" if pval < 0.05 else ""

        summary.append(
            {
                "Group": f"{grp}{sig}",
                "Person-Years": int(denom_mean),
                "Observed": int(outcome_mean),
                "Expected": int(outcome_mean - diff_mean),
                "Diff (95% CI)": f"{int(diff_mean)} ({int(diff_l)}, {int(diff_u)})",
                "Obs Rate": np.round(outcome_rate, 2),
                "Exp Rate": np.round(outcome_rate - rd_mean, 2),
                "Rate Diff CI": f"{rd_mean:.2f} ({rd_l:.2f}, {rd_u:.2f})",
                "Pct Change CI": f"{pct_mean:.1f}% ({pct_l:.1f}%, {pct_u:.1f}%)",
            }
        )

    return pd.DataFrame(summary)


# -----------------------------------------------------------------------------
# Per-unit post-treatment table (mu-based, matches reference make_fertility_table)
# -----------------------------------------------------------------------------


def _compute_per_unit_post_treatment(
    draws_df: pd.DataFrame, csv_path: Path
) -> pd.DataFrame:
    """Per-(unit, group) post-treatment totals matching reference R
    ``make_fertility_table`` and the JAMA supplement per-state tables.

    Estimands (identical to upstream R):
        expected (untreated) = sum(exp(mu))           # counterfactual
        treated              = sum(exp(mu_treated))    # model fit WITH treatment
        excess               = treated - untreated      # "Expected difference"
        excess_pct           = 100 * (treated/untreated - 1)   # percent change

    ``observed`` is retained for transparency only; the supplement's per-state
    excess estimand is the model-implied treatment effect (treated − untreated),
    not observed minus the counterfactual. CI is the draw-level distribution
    of sum(exp(mu)) / sum(exp(mu_treated)).
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
                "treated": float(np.sum(np.exp(g["mu_treated"].to_numpy(dtype=float)))),
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

    # Merge observed (transparency only) and compute the model-implied excess:
    # treated - untreated, matching the supplement's per-state estimand.
    draw_sums = draw_sums.merge(
        observed_totals[["unit", "group", "observed"]],
        on=["unit", "group"],
        how="left",
    )
    draw_sums["excess"] = draw_sums["treated"] - draw_sums["expected"]
    draw_sums["excess_pct"] = 100 * (draw_sums["treated"] / draw_sums["expected"] - 1)

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
    target_unit: str,
    print_target_table: bool = True,
    highlight_unit: str | None = None,
) -> None:
    """Print the headline + per-unit rich tables.

    ``target_unit`` anchors the headline table title. ``highlight_unit`` is the
    unit whose per-unit row is bold-green; pass None (e.g. when the target was
    auto-detected rather than explicitly set) for no highlight.
    """
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
        style = (
            "bold green"
            if highlight_unit is not None and row["unit"] == highlight_unit
            else None
        )
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


# -----------------------------------------------------------------------------
# Run-summary panel (additive terminal output — no file/data side effects)
# -----------------------------------------------------------------------------


_STATUS_STYLE = {
    "PASS": "bold green",
    "WARN": "yellow",
    "FAIL": "bold red",
    "fixed": "dim",
}


def render_diagnostics_table(rows: list[dict], title: str | None = None) -> bool:
    """Render per-parameter diagnostic rows as one Rich table.

    ``rows`` come from ``diagnostics.parameter_diagnostics``. Shows every
    parameter; the ``gate`` column marks whether it counts toward the run
    verdict (non-gated rows are dimmed and shown for information only). ESS is
    a single ``min(bulk, tail)`` column. Returns True iff no GATED row FAILs
    (fixed/constant and non-gated sites never fail the run). Display only.
    """
    from rich.console import Console
    from rich.table import Table

    t = Table(title=title)
    t.add_column("Parameter")
    t.add_column("max R-hat", justify="right")
    t.add_column("min ESS", justify="right")
    t.add_column("status")
    t.add_column("gate")
    for r in rows:
        fmt = lambda v, p: "\u2014" if v is None else f"{v:{p}}"  # noqa: E731
        gated = r.get("gated", True)
        # Non-gated rows are informational: dim them and blank the status
        # style so they don't read as run failures.
        style = _STATUS_STYLE.get(r["status"]) if gated else "dim"
        t.add_row(
            r["param"],
            fmt(r["rhat"], ".4f"),
            fmt(r.get("ess", r.get("ess_bulk")), ".0f"),
            r["status"],
            "\u2713" if gated else "",
            style=style,
        )
    Console().print(t)
    return not any(r["status"] == "FAIL" and r.get("gated", True) for r in rows)


def render_component_table(fits: list[dict], title: str | None = None) -> bool:
    """Render one row per cut Stage-2 component from manifest fit records.

    Each ``fit`` is a ``build_cut_convergence_manifest`` component record
    (``component``, ``stage1_chain``, ``rhat_max``, ``ess_bulk_min``,
    ``divergences``, ``converged``). Lists EVERY component, not just the
    worst. Returns True iff all components converged. Display only;
    diagnostics are never pooled across components.
    """
    from rich.console import Console
    from rich.table import Table

    t = Table(title=title)
    t.add_column("Component", justify="right")
    t.add_column("S1 chain", justify="right")
    t.add_column("max R-hat", justify="right")
    t.add_column("min bulk ESS", justify="right")
    t.add_column("div", justify="right")
    t.add_column("status")
    all_ok = True
    for f in sorted(fits, key=lambda r: int(r["component"])):
        ok = bool(f.get("converged", False))
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        t.add_row(
            str(f["component"]),
            str(f.get("stage1_chain", "\u2014")),
            f"{f['rhat_max']:.4f}" if "rhat_max" in f else "\u2014",
            f"{f['ess_bulk_min']:.0f}" if "ess_bulk_min" in f else "\u2014",
            str(f.get("divergences", "\u2014")),
            status,
            style=_STATUS_STYLE.get(status),
        )
    Console().print(t)
    return all_ok


def print_run_summary_panel(
    model_type: str,
    rank: int,
    num_chains: int,
    chain_method: str,
    outcome_distribution: str,
    convergence: dict,
    figures: list[str],
    artifact_paths: list[str] | str | Path,
    diagnostic_rows: list[dict] | None = None,
    component_fits: list[dict] | None = None,
) -> None:
    """Render a rich Panel summarizing one completed rank run.

    Purely additive terminal output: no file or data side effects, so it
    cannot affect any golden-checked artifact. Intended to be called right
    after a rank's convergence gate and figures have been written, from
    both the joint (``_run_single_rank``) and cut (``_run_cut_rank``) paths.

    Parameters
    ----------
    model_type : str
    rank : int
    num_chains : int
    chain_method : str
    outcome_distribution : str
    convergence : dict
        The convergence gate dict (``diagnostics.convergence_summary()`` /
        ``cut.summarize_mcmc()`` / ``results.build_cut_convergence_manifest()``
        shape): at minimum a ``"converged"`` bool, plus any of
        ``rhat_max``, ``ess_bulk_min``, ``ess_tail_min``, ``divergences``.
    figures : list of str
        Figure names selected for this run (``PLOT_REGISTRY`` subset).
    artifact_paths : list of str, or str/Path
        Artifact directory or list of written artifact paths to display.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    converged = bool(convergence.get("converged", False))
    status = (
        "[bold green]PASS[/bold green]" if converged else "[bold red]FAIL[/bold red]"
    )
    div = convergence.get("divergences", 0)

    # Per-parameter (joint) or per-component (cut) diagnostics table, worst
    # first, printed before the panel. Constant sites show as 'fixed', never
    # FAIL. The panel then carries the one-line verdict + divergences (the
    # gate itself is convergence_summary; this is display only).
    if diagnostic_rows is not None:
        render_diagnostics_table(
            diagnostic_rows, title=f"Convergence — {model_type} rank {rank}"
        )
    elif component_fits is not None:
        render_component_table(
            component_fits,
            title=f"Convergence by component — {model_type} rank {rank}",
        )

    lines = [
        f"[bold]Model type:[/bold] {model_type}   [bold]Rank:[/bold] {rank}",
        f"[bold]Chains:[/bold] {num_chains} ({chain_method})   "
        f"[bold]Distribution:[/bold] {outcome_distribution}",
        "",
        f"[bold]Convergence:[/bold] {status}   [bold]divergences:[/bold] {div}",
    ]

    lines.append("")
    lines.append(f"[bold]Figures:[/bold] {', '.join(figures) if figures else '(none)'}")

    paths = (
        [str(artifact_paths)]
        if isinstance(artifact_paths, (str, Path))
        else [str(p) for p in artifact_paths]
    )
    lines.append(f"[bold]Artifacts:[/bold] {', '.join(paths) if paths else '(none)'}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"Run summary — {model_type} rank {rank}",
            border_style="green" if converged else "red",
        )
    )
    logger.debug(
        f"{model_type} rank {rank}: run-summary panel printed (converged={converged})"
    )
