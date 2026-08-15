"""Posterior-predictive checks and descriptive panel plots.

Accepts both standardized (unit/group/denominator/treatment) and legacy
(state/category/population/exposure_code) column names. Dimensions: K=groups,
D=units, N=time periods.
"""

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Legacy → standardized column name map
_COLUMN_MAPPING = {
    "state": "unit",
    "category": "group",
    "population": "denominator",
    "exposure_code": "treatment",
    "banned_state": "treated_unit",  # derived column
}

# Reverse mapping for detecting legacy columns
_LEGACY_COLUMNS = set(_COLUMN_MAPPING.keys())


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy columns (state/category/population/exposure_code) to
    standardized names (unit/group/denominator/treatment). Returns a copy.
    """
    df = df.copy()

    rename_map = {}
    for legacy, standard in _COLUMN_MAPPING.items():
        if legacy in df.columns and standard not in df.columns:
            rename_map[legacy] = standard

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _detect_outcome_column(df: pd.DataFrame) -> str:
    """Return the outcome column: standardized name first, then legacy fallbacks."""
    for col in ("outcome", "births", "count", "y"):
        if col in df.columns:
            return col
    raise ValueError(
        f"No outcome column found (looked for outcome/births/count/y); have: {list(df.columns)}"
    )


def _identify_treated_units(df: pd.DataFrame) -> list[str]:
    """Units treated at any time point (upstream R's 'banned_state' concept)."""
    if "treated_unit" in df.columns:
        treated_mask = cast(pd.Series, df["treated_unit"]) == 1
        units = cast(pd.Series, df.loc[treated_mask, "unit"])
        return units.drop_duplicates().tolist()
    elif "treatment" in df.columns:
        treated_mask = cast(pd.Series, df["treatment"]) == 1
        units = cast(pd.Series, df.loc[treated_mask, "unit"])
        return units.drop_duplicates().tolist()
    else:
        units = cast(pd.Series, df["unit"])
        return units.drop_duplicates().tolist()


def _filter_ppc_units(
    df: pd.DataFrame,
    treated_units: list[str],
    ppc_units: list[str] | None = None,
    ppc_exclude_units: list[str] | None = None,
) -> pd.DataFrame:
    """Filter DataFrame to units selected for PPC."""
    unit_values = set(cast(pd.Series, df["unit"]).drop_duplicates().tolist())
    if ppc_units is not None:
        if len(ppc_units) == 0:
            warnings.warn(
                "ppc_units is an empty list; no units will be included in PPC.",
                stacklevel=3,
            )
        selected = [u for u in ppc_units if u in unit_values]
        missing = [u for u in ppc_units if u not in unit_values]
        if missing:
            warnings.warn(
                f"ppc_units missing from draws: {missing}",
                stacklevel=3,
            )
        mask = cast(pd.Series, df["unit"]).isin(selected)
        df = cast(pd.DataFrame, df.loc[mask].copy())
    else:
        mask = cast(pd.Series, df["unit"]).isin(treated_units)
        df = cast(pd.DataFrame, df.loc[mask].copy())

    if ppc_exclude_units is not None:
        mask = ~cast(pd.Series, df["unit"]).isin(ppc_exclude_units)
        df = cast(pd.DataFrame, df.loc[mask].copy())

    return df


def _prepare_ppc_residuals(
    draws_df: pd.DataFrame,
    outcome_col: str,
    categories: list[str] | None,
    ppc_units: list[str] | None,
    ppc_exclude_units: list[str] | None,
    *,
    sort_by_time: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Shared setup for the residual PPC checks (abs residual, ACF, RMSE):
    standardize columns, resolve the outcome column and categories,
    restrict to the control period and selected units, and compute
    observed/predicted residuals against the counterfactual rate.

    ``sort_by_time=True`` sorts within each (unit, group, draw) group —
    required for ACF, which needs chronological order to lag correctly.

    Returns
    -------
    df_control : filtered rows with ``pred_diff``/``obs_diff`` added.
    resolved_categories : ``categories``, or every group in ``draws_df``.
    selected_units : units left in ``df_control`` after filtering.

    Callers use the last two to filter their own stats/pvals frames the same way.
    """
    df = _standardize_columns(draws_df)

    if outcome_col not in df.columns:
        outcome_col = _detect_outcome_column(df)

    resolved_categories = (
        categories if categories is not None else df["group"].unique().tolist()
    )
    df = df[df["group"].isin(resolved_categories)]

    treated_units = _identify_treated_units(df)

    df_control = df[df["treatment"] == 0].copy()
    df_control = _filter_ppc_units(
        df_control,
        treated_units,
        ppc_units=ppc_units,
        ppc_exclude_units=ppc_exclude_units,
    )

    df_control["pred_diff"] = df_control["ypred"] - np.exp(df_control["mu"])
    df_control["obs_diff"] = df_control[outcome_col] - np.exp(df_control["mu"])

    if sort_by_time and "time" in df_control.columns:
        df_control = df_control.sort_values(["unit", "group", ".draw", "time"])

    selected_units = df_control["unit"].unique().tolist()

    return df_control, resolved_categories, selected_units


def _compute_autocorrelation(x: np.ndarray, lag: int) -> float:
    """Autocorrelation of 1D array ``x`` at ``lag``. NaN if too short or zero-variance."""
    x = np.asarray(x)
    n = len(x)

    if n <= lag:
        return np.nan

    x_centered = x - np.nanmean(x)
    var = np.nansum(x_centered**2)
    if var == 0:
        return np.nan

    acf = np.nansum(x_centered[:-lag] * x_centered[lag:]) / var

    return acf


def _setup_plot_style():
    """Set up matplotlib style similar to ggplot2 theme_bw()."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "grid.color": "lightgray",
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def _new_fig(figsize: tuple[int, int], **kwargs) -> tuple[Figure, Axes]:
    """Wrapper around ``plt.subplots`` for a single axes."""
    return plt.subplots(figsize=figsize, **kwargs)


def _new_grid_fig(
    figsize: tuple[int, int], nrows: int, ncols: int, **kwargs
) -> tuple[Figure, np.ndarray]:
    """Wrapper around ``plt.subplots`` for a faceted grid; always returns
    a 2D Axes array."""
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def _empty_placeholder_fig(
    figsize: tuple[int, int], message: str, title: str | None = None
) -> tuple[Figure, Axes]:
    """Blank figure with a centered message — used by make_* functions when there's no data."""
    fig, ax = _new_fig(figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    if title is not None:
        ax.set_title(title)
    return fig, ax


def _finalize(
    fig: Figure,
    axes: Axes | list[Axes] | np.ndarray | None = None,
    *,
    rotate_xticks: bool = False,
) -> Figure:
    """Optional x-tick rotation, then ``tight_layout``."""
    if rotate_xticks and axes is not None:
        ax_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        for ax in ax_list:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


def _create_faceted_histograms(
    stats_df: pd.DataFrame,
    pvals_df: pd.DataFrame,
    x_col: str,
    title: str,
    xlabel: str,
    facet_cols: list[str],
    figsize: tuple[int, int],
    ncol: int = 3,
) -> Figure:
    """Grid of histograms, one per combination of ``facet_cols``, each
    annotated with its p-value from ``pvals_df``."""
    _setup_plot_style()

    if len(facet_cols) == 1:
        facet_keys = stats_df[facet_cols[0]].unique()
        facet_labels = [str(k) for k in facet_keys]
    else:
        facet_keys = stats_df[facet_cols].drop_duplicates().values.tolist()
        facet_labels = [" + ".join(str(v) for v in k) for k in facet_keys]

    n_facets = len(facet_keys)
    if n_facets == 0:
        fig, _ = _empty_placeholder_fig(figsize, "No data available", title=title)
        return fig

    nrow = int(np.ceil(n_facets / ncol))

    fig, axes = _new_grid_fig(figsize, nrow, ncol, squeeze=False)
    axes = axes.flatten()

    for i, (facet_key, facet_label) in enumerate(
        zip(facet_keys, facet_labels, strict=True)
    ):
        ax = axes[i]

        if len(facet_cols) == 1:
            mask = stats_df[facet_cols[0]] == facet_key
            pval_mask = pvals_df[facet_cols[0]] == facet_key
        else:
            mask = np.all(
                [
                    stats_df[col] == val
                    for col, val in zip(facet_cols, facet_key, strict=True)
                ],
                axis=0,
            )
            pval_mask = np.all(
                [
                    pvals_df[col] == val
                    for col, val in zip(facet_cols, facet_key, strict=True)
                ],
                axis=0,
            )

        facet_data = stats_df.loc[mask, x_col].dropna()

        if len(facet_data) == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(facet_label)
            continue

        ax.hist(facet_data, bins=30, alpha=0.5, color="steelblue", edgecolor="white")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)

        pval_row = pvals_df.loc[pval_mask]
        if len(pval_row) > 0:
            pval = pval_row["pval"].values[0]
            ax.text(
                0.95,
                0.95,
                f"{pval:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="red",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(facet_label, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for j in range(n_facets, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.supxlabel(xlabel, fontsize=12)
    fig.supylabel("Count", fontsize=12)

    return _finalize(fig)


def make_abs_ppc_plot(
    draws_df: pd.DataFrame,
    outcome_col: str = "outcome",
    categories: list[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    ppc_units: list[str] | None = None,
    ppc_exclude_units: list[str] | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """PPC: max absolute residual per unit/group/draw, observed vs predicted.
    Tests whether the model captures extreme deviations.

    obs_diff = outcome - exp(mu); pred_diff = ypred - exp(mu)
    Statistic: max(|obs_diff|) vs max(|pred_diff|); p-value = P(observed max < predicted max).

    ``draws_df`` needs: .draw, unit, group, treatment, outcome_col, ypred, mu.

    Returns
    -------
    fig : faceted histogram, one panel per unit/group.
    pvals_df : columns unit, group, pval.
    """
    df_control, categories, selected_units = _prepare_ppc_residuals(
        draws_df, outcome_col, categories, ppc_units, ppc_exclude_units
    )

    max_stats = (
        df_control.groupby(["unit", "group", ".draw"])
        .agg(
            max_pred_diff=("pred_diff", lambda x: np.nanmax(np.abs(x))),
            max_obs_diff=("obs_diff", lambda x: np.nanmax(np.abs(x))),
        )
        .reset_index()
    )

    max_stats["diff_in_diff"] = max_stats["max_obs_diff"] - max_stats["max_pred_diff"]

    pvals_df = (
        max_stats.groupby(["unit", "group"])
        .agg(pval=("diff_in_diff", lambda x: np.mean(x < 0)))
        .reset_index()
    )

    pvals_df = pvals_df[pvals_df["group"].isin(categories)]
    pvals_df = pvals_df[pvals_df["unit"].isin(selected_units)]

    fig = _create_faceted_histograms(
        stats_df=max_stats[max_stats["unit"].isin(selected_units)],
        pvals_df=pvals_df,
        x_col="diff_in_diff",
        title="Difference in Maximum Absolute Predicted Residual",
        xlabel="Observed - Predicted Max Residual",
        facet_cols=["unit", "group"],
        figsize=figsize,
    )

    return fig, pvals_df


def make_acf_ppc_plot(
    draws_df: pd.DataFrame,
    lag: int = 6,
    outcome_col: str = "outcome",
    categories: list[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    ppc_units: list[str] | None = None,
    ppc_exclude_units: list[str] | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """PPC: residual autocorrelation at ``lag``, per unit/group/draw, observed vs predicted.

    obs_diff = outcome - exp(mu); pred_diff = ypred - exp(mu)
    Statistic: acf(obs_diff, lag) vs acf(pred_diff, lag); p-value = P(obs_acf - pred_acf < 0).

    Same required columns as ``make_abs_ppc_plot``.

    Returns
    -------
    fig : faceted histogram, one panel per unit/group.
    pvals_df : columns unit, group, pval.
    """
    df_control, categories, selected_units = _prepare_ppc_residuals(
        draws_df,
        outcome_col,
        categories,
        ppc_units,
        ppc_exclude_units,
        sort_by_time=True,
    )

    def compute_acf_stats(group_df):
        obs_vals = group_df["obs_diff"].values
        pred_vals = group_df["pred_diff"].values

        obs_acf = _compute_autocorrelation(obs_vals, lag)
        pred_acf = _compute_autocorrelation(pred_vals, lag)

        return pd.Series(
            {"obs_ac": obs_acf, "pred_ac": pred_acf, "diff_in_ac": obs_acf - pred_acf}
        )

    acf_stats = (
        df_control.groupby(["unit", "group", ".draw"])
        .apply(compute_acf_stats, include_groups=False)
        .reset_index()
    )
    acf_stats = acf_stats.dropna(subset=["diff_in_ac"])

    pvals_df = (
        acf_stats.groupby(["unit", "group"])
        .agg(pval=("diff_in_ac", lambda x: np.mean(x < 0)))
        .reset_index()
    )

    pvals_df = pvals_df[pvals_df["group"].isin(categories)]
    pvals_df = pvals_df[pvals_df["unit"].isin(selected_units)]

    fig = _create_faceted_histograms(
        stats_df=acf_stats[acf_stats["unit"].isin(selected_units)],
        pvals_df=pvals_df,
        x_col="diff_in_ac",
        title=f"Difference in Residual Autocorrelation (Lag {lag})",
        xlabel="Observed - Predicted Autocorrelation",
        facet_cols=["unit", "group"],
        figsize=figsize,
    )

    return fig, pvals_df


def make_rmse_ppc_plot(
    draws_df: pd.DataFrame,
    outcome_col: str = "outcome",
    categories: list[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    ppc_units: list[str] | None = None,
    ppc_exclude_units: list[str] | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """PPC: RMSE of residuals per unit/group/draw, observed vs predicted.

    obs_diff = outcome - exp(mu); pred_diff = ypred - exp(mu)
    Statistic: sqrt(mean(obs_diff^2)) vs sqrt(mean(pred_diff^2)); p-value = P(observed RMSE < predicted RMSE).

    Same required columns as ``make_abs_ppc_plot``.

    Returns
    -------
    fig : faceted histogram, one panel per unit/group.
    pvals_df : columns unit, group, pval.
    """
    df_control, categories, selected_units = _prepare_ppc_residuals(
        draws_df, outcome_col, categories, ppc_units, ppc_exclude_units
    )

    rmse_stats = (
        df_control.groupby(["unit", "group", ".draw"])
        .agg(
            rmse_pred_diff=("pred_diff", lambda x: np.sqrt(np.nanmean(x**2))),
            rmse_obs_diff=("obs_diff", lambda x: np.sqrt(np.nanmean(x**2))),
        )
        .reset_index()
    )

    rmse_stats["diff_in_diff"] = (
        rmse_stats["rmse_obs_diff"] - rmse_stats["rmse_pred_diff"]
    )

    pvals_df = (
        rmse_stats.groupby(["unit", "group"])
        .agg(pval=("diff_in_diff", lambda x: np.mean(x < 0)))
        .reset_index()
    )

    pvals_df = pvals_df[pvals_df["group"].isin(categories)]
    pvals_df = pvals_df[pvals_df["unit"].isin(selected_units)]

    rmse_stats_plot = rmse_stats[rmse_stats["unit"].isin(selected_units)]
    rmse_stats_plot = rmse_stats_plot[rmse_stats_plot["group"].isin(categories)]

    fig = _create_faceted_histograms(
        stats_df=rmse_stats_plot,
        pvals_df=pvals_df,
        x_col="diff_in_diff",
        title="Difference in RMSE",
        xlabel="Observed - Predicted RMSE",
        facet_cols=["unit", "group"],
        figsize=figsize,
    )

    return fig, pvals_df


def make_unit_corr_ppc_plot(
    draws_df: pd.DataFrame,
    max_treat_date: str | None = None,
    outcome_col: str = "outcome",
    categories: list[str] | None = None,
    ndraws: int = 1000,
    figsize: tuple[int, int] = (10, 6),
    ppc_units: list[str] | None = None,
    ppc_exclude_units: list[str] | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """PPC: spectral norm (largest eigenvalue) of the cross-unit residual
    correlation matrix, per time point, observed vs predicted. Tests
    whether the model captures cross-sectional dependence.

    obs_diff = outcome - exp(mu); pred_diff = ypred - exp(mu)
    Statistic: sqrt(max eigenvalue of correlation matrix); p-value = P(observed < predicted).

    Same required columns as ``make_abs_ppc_plot``. ``max_treat_date``
    (format YYYY-MM-DD) restricts to time < that date; otherwise uses the
    control period only. ``ndraws`` caps draws used, for speed.

    Returns
    -------
    fig : faceted histogram, one panel per category.
    pvals_df : columns group, pval.
    """
    df = _standardize_columns(draws_df)

    if outcome_col not in df.columns:
        outcome_col = _detect_outcome_column(df)

    if categories is None:
        categories = df["group"].unique().tolist()

    df = df[df["group"].isin(categories)]

    if ppc_units is not None or ppc_exclude_units is not None:
        all_units = cast(pd.Series, df["unit"]).drop_duplicates().tolist()
        df = _filter_ppc_units(
            df,
            treated_units=all_units,
            ppc_units=ppc_units,
            ppc_exclude_units=ppc_exclude_units,
        )

    if max_treat_date is not None and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df[df["time"] < pd.to_datetime(max_treat_date)]
    elif "treatment" in df.columns:
        df = df[df["treatment"] == 0]  # control period only

    if ".draw" in df.columns:
        unique_draws = df[".draw"].unique()
        if len(unique_draws) > ndraws:
            selected_draws = unique_draws[:ndraws]
            df = df[df[".draw"].isin(selected_draws)]

    df["obs_residual"] = df[outcome_col] - np.exp(df["mu"])
    df["pred_residual"] = df["ypred"] - np.exp(df["mu"])

    # drop units with >25% missing outcome
    na_frac = df.groupby(["unit", "group"])[outcome_col].apply(
        lambda x: x.isna().mean()
    )
    na_frac = na_frac.reset_index(name="na_frac")
    valid_units = na_frac[na_frac["na_frac"] < 0.25][
        ["unit", "group"]
    ].drop_duplicates()
    df = df.merge(valid_units, on=["unit", "group"], how="inner")

    def compute_spectral_norm(residuals_matrix: np.ndarray) -> float:
        """Largest eigenvalue of the correlation matrix, sqrt'd."""
        valid_cols = ~np.all(np.isnan(residuals_matrix), axis=0)
        residuals_matrix = residuals_matrix[:, valid_cols]

        if residuals_matrix.shape[1] < 2:
            return np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_matrix = np.corrcoef(residuals_matrix.T)

        if np.any(np.isnan(corr_matrix)):
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 1.0)

        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        max_eigenvalue = np.max(eigenvalues)

        return np.sqrt(max(max_eigenvalue, 0))

    eval_results = []

    for group in categories:
        group_df = df[df["group"] == group]

        if len(group_df) == 0:
            continue

        unique_draws = group_df[".draw"].unique()
        units = group_df["unit"].unique()
        times = group_df["time"].unique() if "time" in group_df.columns else [0]

        for draw in unique_draws:
            draw_df = group_df[group_df[".draw"] == draw]

            # time x unit residual matrices
            obs_matrix = np.full((len(times), len(units)), np.nan)
            pred_matrix = np.full((len(times), len(units)), np.nan)

            time_to_idx = {t: i for i, t in enumerate(times)}
            unit_to_idx = {u: i for i, u in enumerate(units)}

            for _, row in draw_df.iterrows():
                t_idx = time_to_idx.get(row.get("time", 0), 0)
                u_idx = unit_to_idx.get(row["unit"])
                if u_idx is not None:
                    obs_matrix[t_idx, u_idx] = row["obs_residual"]
                    # Mask pred_residual where obs_residual is NaN
                    if not np.isnan(row["obs_residual"]):
                        pred_matrix[t_idx, u_idx] = row["pred_residual"]

            obs_sval = compute_spectral_norm(obs_matrix)
            pred_sval = compute_spectral_norm(pred_matrix)

            eval_results.append(
                {
                    "group": group,
                    ".draw": draw,
                    "obs_sval": obs_sval,
                    "pred_sval": pred_sval,
                    "eval_diff": obs_sval - pred_sval,
                }
            )

    eval_stats = pd.DataFrame(eval_results)
    eval_stats = eval_stats.dropna(subset=["eval_diff"])

    if len(eval_stats) == 0:
        fig, _ = _empty_placeholder_fig(
            figsize,
            "Insufficient data for spectral norm computation",
            title="Difference in Unit Correlations",
        )
        return fig, pd.DataFrame(columns=["group", "pval"])

    pvals_df = (
        eval_stats.groupby("group")
        .agg(pval=("eval_diff", lambda x: np.mean(x < 0)))
        .reset_index()
    )

    pvals_df = pvals_df[pvals_df["group"].isin(categories)]

    fig = _create_faceted_histograms(
        stats_df=eval_stats,
        pvals_df=pvals_df,
        x_col="eval_diff",
        title="Difference in Unit Correlations",
        xlabel="Observed - Predicted Spectral Norm",
        facet_cols=["group"],
        figsize=figsize,
        ncol=2,
    )

    return fig, pvals_df


def make_raw_rate_plot(
    df: pd.DataFrame,
    group: str | None = None,
    unit_col: str = "unit",
    rate_multiplier: float = 1000,
    treatment_dates: dict[str, str] | None = None,
    separate_unit: str | None = None,
    smooth_window: int | None = None,
    plot_type: str = "rate",  # 'rate' or 'count'
    figsize: tuple[int, int] = (10, 6),
) -> tuple[Figure, Axes]:
    """Time series of rates (or counts) for Treated vs Control units, with
    optional smoothing, treatment-date markers, and one unit split out.

    Parameters
    ----------
    df : columns unit, time, group, outcome, denominator, treatment (or
        legacy names). ``unit`` is a panel entity (state, hospital, etc.).
    group : filter to this group/category; None aggregates all groups.
    unit_col : column holding unit identifiers, if not already ``unit``.
    rate_multiplier : e.g. 1000 = rate per 1,000.
    treatment_dates : {label: date} for vertical marker lines,
        e.g. {'Policy Change': '2022-06-24'}.
    separate_unit : plot this unit separately from other treated units.
    smooth_window : rolling-mean window; None = no smoothing.
    plot_type : 'rate' (outcome/denominator * rate_multiplier) or 'count'.
    """
    _setup_plot_style()

    df = _standardize_columns(df)

    if unit_col != "unit" and unit_col in df.columns:
        df = df.rename(columns={unit_col: "unit"})

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    if group is not None and "group" in df.columns:
        df = df[df["group"] == group].copy()

    treated_units = _identify_treated_units(df)

    def assign_treatment_group(row):
        if separate_unit is not None and row["unit"] == separate_unit:
            return separate_unit
        elif row["unit"] in treated_units:
            return "Treated"
        else:
            return "Control"

    df["treatment_group"] = df.apply(assign_treatment_group, axis=1)

    agg_df = (
        df.groupby(["treatment_group", "time"])
        .agg(outcome=("outcome", "sum"), denominator=("denominator", "sum"))
        .reset_index()
    )

    if plot_type == "count":
        agg_df["y_value"] = agg_df["outcome"]
        ylabel = "Count"
    else:  # 'rate' (default)
        agg_df["y_value"] = (
            agg_df["outcome"] / agg_df["denominator"]
        ) * rate_multiplier
        ylabel = f"Rate per {rate_multiplier:,.0f}"

    if smooth_window is not None and smooth_window > 1:
        agg_df = agg_df.sort_values(["treatment_group", "time"])
        agg_df["y_smooth"] = agg_df.groupby("treatment_group")["y_value"].transform(
            lambda x: x.rolling(window=smooth_window, center=True, min_periods=1).mean()
        )
    else:
        agg_df["y_smooth"] = agg_df["y_value"]

    agg_df = agg_df.sort_values("time")

    colors = {
        "Treated": "#E41A1C",  # Red
        "Control": "#999999",  # Gray
    }
    if separate_unit is not None:
        colors[separate_unit] = "#FF7F00"  # Orange

    fig, ax = _new_fig(figsize)

    # Control drawn first so it sits behind Treated/separate_unit
    plot_order = ["Control", "Treated"]
    if separate_unit is not None:
        plot_order.append(separate_unit)

    for tgroup in plot_order:
        group_data = agg_df[agg_df["treatment_group"] == tgroup]
        if len(group_data) == 0:
            continue

        color = colors.get(tgroup, "#333333")
        linewidth = 2.0 if tgroup != "Control" else 1.5
        alpha = 1.0 if tgroup != "Control" else 0.7

        ax.plot(
            group_data["time"],
            group_data["y_smooth"],
            label=tgroup,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

        # smoothed line + faint raw points
        if smooth_window is not None and smooth_window > 1:
            ax.scatter(
                group_data["time"],
                group_data["y_value"],
                color=color,
                alpha=0.2,
                s=10,
            )

    if treatment_dates is not None:
        marker_colors = [
            "#FF8C00",
            "#DC143C",
            "#9400D3",
            "#228B22",
        ]  # Orange, Crimson, Violet, Green
        for i, (label, date_str) in enumerate(treatment_dates.items()):
            date = pd.to_datetime(date_str)
            color = marker_colors[i % len(marker_colors)]
            ax.axvline(x=date, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(
                date,
                y_pos,
                f" {label}",
                ha="left",
                va="top",
                fontsize=9,
                rotation=90,
                color=color,
                alpha=0.9,
                fontweight="bold",
            )

    group_label = f" ({group})" if group else ""
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel(f"{ylabel}{group_label}", fontsize=11)
    title_type = "Rate" if plot_type == "rate" else "Count"
    ax.set_title(
        f"{title_type} by Treatment Group{group_label}", fontsize=13, fontweight="bold"
    )
    ax.legend(loc="best", frameon=True, fancybox=True)

    _finalize(fig, ax, rotate_xticks=True)
    return fig, ax


def make_group_comparison_plot(
    df: pd.DataFrame,
    groups: list[str] | None = None,
    rate_multiplier: float = 1000,
    treatment_dates: dict[str, str] | None = None,
    plot_type: str = "rate",  # 'rate' or 'count'
    figsize: tuple[int, int] = (12, 8),
) -> tuple[Figure, np.ndarray]:
    """Faceted plot: one subplot per group, each showing Treated vs Control
    time series.

    Parameters
    ----------
    df : columns unit, time, group, outcome, denominator, treatment (or legacy names).
    groups : groups to plot; None uses every group present.
    rate_multiplier : e.g. 1000 = rate per 1,000.
    treatment_dates : {label: date} for vertical marker lines.
    plot_type : 'rate' (outcome/denominator * multiplier) or 'count'.
    """
    _setup_plot_style()

    df = _standardize_columns(df)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    if groups is None:
        groups = df["group"].unique().tolist()

    n_groups = len(groups)
    if n_groups == 0:
        fig, ax = _empty_placeholder_fig(figsize, "No groups to plot")
        return fig, np.array([ax])

    ncols = min(2, n_groups)
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = _new_grid_fig(figsize, nrows, ncols, squeeze=False, sharex=True)
    axes_flat = axes.flatten()

    treated_units = _identify_treated_units(df)

    colors = {
        "Treated": "#E41A1C",
        "Control": "#999999",
    }

    for idx, group in enumerate(groups):
        ax = axes_flat[idx]

        group_df = df[df["group"] == group].copy()

        if len(group_df) == 0:
            ax.text(
                0.5,
                0.5,
                f"No data for {group}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(group, fontsize=11, fontweight="bold")
            continue

        group_df["treatment_group"] = group_df["unit"].apply(
            lambda u: "Treated" if u in treated_units else "Control"
        )

        agg_df = (
            group_df.groupby(["treatment_group", "time"])
            .agg(outcome=("outcome", "sum"), denominator=("denominator", "sum"))
            .reset_index()
        )

        if plot_type == "count":
            agg_df["y_value"] = agg_df["outcome"]
            ylabel = "Count"
        else:  # 'rate'
            agg_df["y_value"] = (
                agg_df["outcome"] / agg_df["denominator"]
            ) * rate_multiplier
            ylabel = f"Rate per {rate_multiplier:,.0f}"
        agg_df = agg_df.sort_values("time")

        for tgroup in ["Control", "Treated"]:
            tg_data = agg_df[agg_df["treatment_group"] == tgroup]
            if len(tg_data) == 0:
                continue

            color = colors.get(tgroup, "#333333")
            linewidth = 2.0 if tgroup == "Treated" else 1.5
            alpha = 1.0 if tgroup == "Treated" else 0.7

            ax.plot(
                tg_data["time"],
                tg_data["y_value"],
                label=tgroup,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )

        if treatment_dates is not None:
            marker_colors = ["#FF8C00", "#DC143C", "#9400D3", "#228B22"]
            for i, (_label, date_str) in enumerate(treatment_dates.items()):
                date = pd.to_datetime(date_str)
                color = marker_colors[i % len(marker_colors)]
                ax.axvline(
                    x=date, color=color, linestyle="--", linewidth=1.0, alpha=0.8
                )

        ax.set_title(group, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)

        if idx == 0:  # legend on first subplot only
            ax.legend(loc="best", frameon=True, fontsize=8)

    for idx in range(len(groups), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.supxlabel("Time", fontsize=11)
    title_type = "Rate" if plot_type == "rate" else "Count"
    fig.suptitle(
        f"{title_type} Comparison by Group", fontsize=13, fontweight="bold", y=1.02
    )

    _finalize(fig, axes_flat[: len(groups)], rotate_xticks=True)
    return fig, axes


def make_unit_fit_plot(
    quantiles_df: pd.DataFrame,
    unit_name: str,
    group: str = "total",
    outcome_col: str = "outcome",
    figsize: tuple[int, int] = (10, 6),
) -> tuple[Figure, Axes]:
    """Observed vs predicted (with 95% CI) over time for one unit/group.

    ``quantiles_df`` needs ypred_mean, ypred_lower, ypred_upper columns.
    """
    _setup_plot_style()
    sns.set_palette("husl")
    df = _standardize_columns(quantiles_df)

    if outcome_col not in df.columns and any(
        col in df.columns for col in ("outcome", "births", "count", "y")
    ):
        outcome_col = _detect_outcome_column(df)

    df_plot = df[(df["group"] == group) & (df["unit"] == unit_name)].copy()
    if df_plot.empty:
        fig, ax = _empty_placeholder_fig(
            figsize,
            f"No data for {unit_name} / {group}",
            title=f"{unit_name} ({group})",
        )
        return fig, ax

    if "time" in df_plot.columns and not pd.api.types.is_datetime64_any_dtype(
        df_plot["time"]
    ):
        df_plot["time"] = pd.to_datetime(df_plot["time"])

    df_plot = df_plot.sort_values("time")

    # drop NaN rows separately per series so gaps don't draw spurious lines
    df_pred = df_plot.dropna(subset=["ypred_mean", "ypred_lower", "ypred_upper"])
    df_obs = df_plot.dropna(subset=[outcome_col])

    fig, ax = _new_fig(figsize)

    if not df_pred.empty:
        ax.fill_between(
            df_pred["time"],
            df_pred["ypred_lower"],
            df_pred["ypred_upper"],
            alpha=0.3,
            color=sns.color_palette()[0],
            label="95% CI",
        )

        sns.lineplot(
            data=df_pred,
            x="time",
            y="ypred_mean",
            color=sns.color_palette()[3],
            linewidth=2,
            label="Predicted Mean",
            ax=ax,
        )

    if not df_obs.empty:
        sns.scatterplot(
            data=df_obs,
            x="time",
            y=outcome_col,
            color=".2",
            s=40,
            alpha=0.8,
            label="Observed",
            zorder=3,
            ax=ax,
        )

    if (
        "treated_unit" in df_plot.columns
        and df_plot["treated_unit"].iloc[0]
        and "treatment" in df_plot.columns
    ):
        treated_times = df_plot[df_plot["treatment"] == 1]["time"]
        if not treated_times.empty:
            t_date = treated_times.iloc[0]
            ax.axvline(
                x=t_date,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Treatment start",
            )

    ax.set_title(f"Model Fit: {unit_name} ({group})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(outcome_col.capitalize(), fontsize=12)
    ax.legend(loc="best", frameon=True)

    _finalize(fig, ax, rotate_xticks=True)
    return fig, ax


def make_unit_gap_plot(
    quantiles_df: pd.DataFrame,
    unit_name: str,
    group: str = "total",
    outcome_col: str = "outcome",
    figsize: tuple[int, int] = (10, 6),
) -> tuple[Figure, Axes]:
    """Relative gap (observed/predicted - 1) over time for one unit/group.

    ``quantiles_df`` needs ypred_mean, ypred_lower, ypred_upper columns.
    """
    _setup_plot_style()
    df = _standardize_columns(quantiles_df)

    if outcome_col not in df.columns and any(
        col in df.columns for col in ("outcome", "births", "count", "y")
    ):
        outcome_col = _detect_outcome_column(df)

    df_plot = df[(df["group"] == group) & (df["unit"] == unit_name)].copy()
    if df_plot.empty:
        fig, ax = _empty_placeholder_fig(
            figsize,
            f"No data for {unit_name} / {group}",
            title=f"Gap: {unit_name} ({group})",
        )
        return fig, ax

    if "time" in df_plot.columns and not pd.api.types.is_datetime64_any_dtype(
        df_plot["time"]
    ):
        df_plot["time"] = pd.to_datetime(df_plot["time"])

    df_plot = df_plot.sort_values("time")

    # ratio undefined for NaN/non-positive predictions
    df_plot = df_plot.dropna(
        subset=[outcome_col, "ypred_mean", "ypred_lower", "ypred_upper"]
    )
    df_plot = df_plot[
        (df_plot["ypred_mean"] > 0)
        & (df_plot["ypred_lower"] > 0)
        & (df_plot["ypred_upper"] > 0)
    ]

    if df_plot.empty:
        fig, ax = _empty_placeholder_fig(figsize, "No valid data points")
        return fig, ax

    t_date = None
    if (
        "treated_unit" in df_plot.columns
        and df_plot["treated_unit"].iloc[0]
        and "treatment" in df_plot.columns
    ):
        treated_times = df_plot[df_plot["treatment"] == 1]["time"]
        if not treated_times.empty:
            t_date = treated_times.iloc[0]

    fig, ax = _new_fig(figsize)

    gap_mean = df_plot[outcome_col] / df_plot["ypred_mean"] - 1
    gap_lower = df_plot[outcome_col] / df_plot["ypred_upper"] - 1
    gap_upper = df_plot[outcome_col] / df_plot["ypred_lower"] - 1

    color_ribbon = sns.color_palette("muted")[0]
    color_line = sns.color_palette("dark")[3]

    if t_date is not None:
        # pre/post split so the ribbon/line don't connect across treatment
        pre = df_plot["time"] < t_date
        post = df_plot["time"] >= t_date

        for mask in [pre, post]:
            if not mask.any():
                continue
            ax.fill_between(
                df_plot.loc[mask, "time"],
                gap_lower[mask],
                gap_upper[mask],
                alpha=0.25,
                color=color_ribbon,
            )
            ax.plot(
                df_plot.loc[mask, "time"],
                gap_mean[mask],
                color=color_line,
                linewidth=2,
            )
        ax.axvline(x=t_date, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    else:
        ax.fill_between(
            df_plot["time"], gap_lower, gap_upper, alpha=0.25, color=color_ribbon
        )
        ax.plot(df_plot["time"], gap_mean, color=color_line, linewidth=2)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.75)

    ax.set_title(
        f"Prediction Gap: {unit_name} ({group})", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Observed / Predicted - 1", fontsize=12)

    _finalize(fig, ax, rotate_xticks=True)
    return fig, ax


def make_interval_plot(
    merged_df: pd.DataFrame,
    units: list[str] | None = None,
    group_var: str = "unit",
    categories: list[str] | None = None,
    outcome_col: str = "outcome",
    denom_col: str = "denominator",
    rate_normalizer: float = 1000.0,
    estimand: str = "diff",
    method: str = "mu",
    x_var: str = "unit",
    color_group: str | None = None,
    figsize: tuple[int, int] = (12, 10),
) -> tuple[Figure, Axes]:
    """Effect estimate with 67%/95% credible intervals, one row per
    ``x_var`` value (default: unit).

    Parameters
    ----------
    merged_df : draws with .draw, time, unit, group, mu, mu_treated, etc.
    units : units to include; None = all treated units.
    group_var : variable to aggregate the effect by (default 'unit').
    categories : categories to include; None = all.
    estimand : 'diff' (rate difference) or 'ratio' (percent change).
    method : 'mu' (model log-rate) or 'pred' (posterior predictive counts).
    x_var : variable plotted on the y-axis rows (default 'unit').
    color_group : variable used to color/dodge points within a row.
    """
    _setup_plot_style()
    df = _standardize_columns(merged_df)

    if color_group is None:
        color_group = "group" if group_var == "unit" else "unit"

    if outcome_col not in df.columns and any(
        col in df.columns for col in ("outcome", "births", "count", "y")
    ):
        outcome_col = _detect_outcome_column(df)

    if denom_col not in df.columns and "population" in df.columns:
        denom_col = "population"

    if categories is not None:
        df = df[df["group"].isin(categories)]

    if "treatment" in df.columns:
        df = df[df["treatment"] == 1]

    if units is None and "treated_unit" in df.columns:
        units = df[df["treated_unit"] == 1]["unit"].unique().tolist()

    if units is not None:
        df = df[df["unit"].isin(units)]

    agg_cols = list(
        set(
            [".draw", group_var]
            + (
                [color_group]
                if color_group in df.columns and color_group != group_var
                else []
            )
        )
    )

    if "start_date" in df.columns and "end_date" in df.columns:
        df["years"] = (
            pd.to_datetime(df["end_date"]) - pd.to_datetime(df["start_date"])
        ).dt.days / 365.25
    else:
        df["years"] = 1.0

    def compute_draw_effect(grp):
        if method == "mu":
            treated = np.sum(np.exp(grp["mu_treated"]))
            untreated = np.sum(np.exp(grp["mu"]))
            # Summed person-time (pop_t * years_t), matching upstream
            # make_interval_plot and make_summary_table. A mean here would
            # divide summed counts by per-period person-time (off by T).
            denom_val = np.sum(grp[denom_col] * grp["years"]) if len(grp) > 0 else 1

            treated_rate = treated / denom_val * rate_normalizer
            untreated_rate = untreated / denom_val * rate_normalizer

            if estimand == "diff":
                return treated_rate - untreated_rate
            else:
                return (
                    100 * (treated_rate / untreated_rate - 1)
                    if untreated_rate > 0
                    else 0
                )
        else:
            outcome_val = np.sum(grp[outcome_col])
            ypred_val = np.sum(grp["ypred"])
            denom_val = np.mean(grp[denom_col])
            years = np.mean(grp["years"])

            outcome_rate = (outcome_val / years) / (denom_val / rate_normalizer)
            ypred_rate = (ypred_val / years) / (denom_val / rate_normalizer)

            if estimand == "diff":
                return outcome_rate - ypred_rate
            else:
                return outcome_rate / ypred_rate if ypred_rate > 0 else 1.0

    effect_df = (
        df.groupby(agg_cols)
        .apply(compute_draw_effect, include_groups=False)
        .reset_index(name="causal_effect")
    )

    groupby_cols = [x_var] + (
        [color_group]
        if color_group in effect_df.columns and color_group != x_var
        else []
    )

    plot_df = (
        effect_df.groupby(groupby_cols)
        .agg(
            median=("causal_effect", "median"),
            lower_95=("causal_effect", lambda x: np.quantile(x, 0.025)),
            upper_95=("causal_effect", lambda x: np.quantile(x, 0.975)),
            lower_67=("causal_effect", lambda x: np.quantile(x, 0.165)),
            upper_67=("causal_effect", lambda x: np.quantile(x, 0.835)),
        )
        .reset_index()
    )

    sort_order = plot_df.groupby(x_var)["median"].median().sort_values().index
    plot_df[x_var] = pd.Categorical(plot_df[x_var], categories=sort_order, ordered=True)
    plot_df = plot_df.sort_values(x_var)

    fig, ax = _new_fig(figsize)
    y_positions = np.arange(len(sort_order))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sort_order)

    if color_group in effect_df.columns and color_group != x_var:
        color_cats = plot_df[color_group].unique()
        palette = sns.color_palette("deep", n_colors=len(color_cats))
        offsets = np.linspace(-0.2, 0.2, len(color_cats))

        for i, c in enumerate(color_cats):
            c_data = plot_df[plot_df[color_group] == c]
            y_locs = [sort_order.get_loc(v) for v in c_data[x_var]]
            dodged_y = np.array(y_locs) + offsets[i]

            ax.hlines(
                dodged_y,
                c_data["lower_95"],
                c_data["upper_95"],
                color=palette[i],
                alpha=0.4,
                linewidth=2,
            )
            ax.hlines(
                dodged_y,
                c_data["lower_67"],
                c_data["upper_67"],
                color=palette[i],
                alpha=0.9,
                linewidth=4,
            )
            ax.plot(
                c_data["median"],
                dodged_y,
                "o",
                color="white",
                markersize=6,
                markeredgecolor=palette[i],
                markeredgewidth=2,
                label=c,
            )
        ax.legend(title=color_group, loc="best")
    else:
        color = sns.color_palette("deep")[0]
        ax.hlines(
            y_positions,
            plot_df["lower_95"],
            plot_df["upper_95"],
            color=color,
            alpha=0.4,
            linewidth=2,
        )
        ax.hlines(
            y_positions,
            plot_df["lower_67"],
            plot_df["upper_67"],
            color=color,
            alpha=0.9,
            linewidth=4,
        )
        ax.plot(
            plot_df["median"],
            y_positions,
            "o",
            color="white",
            markersize=6,
            markeredgecolor=color,
            markeredgewidth=2,
        )

    ax.axvline(
        x=0 if estimand == "diff" else (0 if method == "mu" else 1),
        color="black",
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel(
        "Expected Percent Change"
        if (estimand == "ratio" and method == "mu")
        else "Causal Effect",
        fontsize=12,
    )
    ax.set_ylabel("")
    ax.set_title("Causal Intervals", fontsize=14, fontweight="bold")

    return _finalize(fig), ax


def make_all_ppc_plots(
    draws_df: pd.DataFrame,
    output_dir: str | None = None,
    outcome_col: str = "outcome",
    categories: list[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    acf_lag: int = 6,
    acf_lags: list[int] | None = None,
    max_treat_date: str | None = None,
    ndraws: int = 1000,
    ppc_units: list[str] | None = None,
    ppc_exclude_units: list[str] | None = None,
) -> dict[str, dict]:
    """Run all four PPC checks (abs residual, ACF, RMSE, unit correlation);
    save PNGs + a combined p-values CSV if ``output_dir`` is given.

    Returns a dict keyed by 'abs', 'acf', 'rmse', 'unit_corr', each
    ``{'fig': ..., 'pvals': ...}``.
    """
    results = {}

    logger.info("Generating PPC plots...")

    logger.info("  - Maximum absolute residual plot...")
    fig_abs, pvals_abs = make_abs_ppc_plot(
        draws_df,
        outcome_col=outcome_col,
        categories=categories,
        figsize=figsize,
        ppc_units=ppc_units,
        ppc_exclude_units=ppc_exclude_units,
    )
    results["abs"] = {"fig": fig_abs, "pvals": pvals_abs}
    logger.debug(f"    Generated {len(pvals_abs)} p-values for abs residual check")

    resolved_acf_lags = acf_lags if acf_lags is not None else [acf_lag]
    acf_results = {}
    for lag in resolved_acf_lags:
        logger.info(f"  - ACF plot (lag={lag})...")
        fig_acf, pvals_acf = make_acf_ppc_plot(
            draws_df,
            lag=lag,
            outcome_col=outcome_col,
            categories=categories,
            figsize=figsize,
            ppc_units=ppc_units,
            ppc_exclude_units=ppc_exclude_units,
        )
        acf_key = f"acf_lag{lag}"
        results[acf_key] = {"fig": fig_acf, "pvals": pvals_acf, "lag": lag}
        acf_results[acf_key] = results[acf_key]
        logger.debug(f"    Generated {len(pvals_acf)} p-values for ACF check")
    if len(resolved_acf_lags) == 1:
        results["acf"] = next(iter(acf_results.values()))

    logger.info("  - RMSE plot...")
    fig_rmse, pvals_rmse = make_rmse_ppc_plot(
        draws_df,
        outcome_col=outcome_col,
        categories=categories,
        figsize=figsize,
        ppc_units=ppc_units,
        ppc_exclude_units=ppc_exclude_units,
    )
    results["rmse"] = {"fig": fig_rmse, "pvals": pvals_rmse}
    logger.debug(f"    Generated {len(pvals_rmse)} p-values for RMSE check")

    # Not filtered by ppc_units: spectral norm needs >=2 units, so this
    # check always runs on every unit in the frame (matches upstream).
    logger.info("  - Unit correlation plot...")
    fig_corr, pvals_corr = make_unit_corr_ppc_plot(
        draws_df,
        max_treat_date=max_treat_date,
        outcome_col=outcome_col,
        categories=categories,
        ndraws=ndraws,
        figsize=(10, 6),
    )
    results["unit_corr"] = {"fig": fig_corr, "pvals": pvals_corr}
    logger.debug(f"    Generated {len(pvals_corr)} p-values for unit correlation check")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving plots to {output_dir}...")
        fig_abs.savefig(
            output_path / "ppc_abs_residual.png", dpi=150, bbox_inches="tight"
        )
        for lag in resolved_acf_lags:
            acf_key = f"acf_lag{lag}"
            results[acf_key]["fig"].savefig(
                output_path / f"ppc_acf_lag{lag}.png", dpi=150, bbox_inches="tight"
            )
        fig_rmse.savefig(output_path / "ppc_rmse.png", dpi=150, bbox_inches="tight")
        fig_corr.savefig(
            output_path / "ppc_unit_corr.png", dpi=150, bbox_inches="tight"
        )
        logger.debug("  Saved all PPC plot images")

        all_pvals = []
        for name, data in results.items():
            if name == "acf":
                continue
            pvals = data["pvals"].copy()
            pvals["check_type"] = name
            all_pvals.append(pvals)

        if all_pvals:
            combined_pvals = pd.concat(all_pvals, ignore_index=True)
            combined_pvals.to_csv(output_path / "ppc_pvalues.csv", index=False)
            logger.info(f"  Saved p-values to {output_path / 'ppc_pvalues.csv'}")

    logger.info("PPC plots completed.")
    return results


# -----------------------------------------------------------------------------
# Figure selection registry
# -----------------------------------------------------------------------------

# Figure name -> plotting function. output.figures selects which of these
# reports.generate_reports() renders; config.py imports this dict directly,
# so there's no separate key list to keep in sync. summary_table is a table,
# not a figure, and always renders, so it's not listed here.
PLOT_REGISTRY: dict[str, Callable] = {
    "unit_fit": make_unit_fit_plot,
    "unit_gap": make_unit_gap_plot,
    "raw_rate": make_raw_rate_plot,
    "interval": make_interval_plot,
    "group_comparison": make_group_comparison_plot,
    "ppc": make_all_ppc_plots,
}
