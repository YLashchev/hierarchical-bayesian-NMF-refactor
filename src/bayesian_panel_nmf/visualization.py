"""Posterior-predictive checks and descriptive panel plots.

Accepts both standardized (unit/group/denominator/treatment) and legacy
(state/category/population/exposure_code) column names. Dimensions: K=groups,
D=units, N=time periods.
"""

import warnings
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
    """
    Normalize column names to standardized format.

    Handles both standardized names (unit, group, denominator, treatment)
    and legacy names (state, category, population, exposure_code).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with either standardized or legacy column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names. Original DataFrame is not modified.
    """
    df = df.copy()

    # Rename legacy columns to standardized names
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
    """
    Identify units that have treatment at any time point.

    In the original R code, this was the 'banned_state' concept - units
    that have treatment == 1 at some point.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with standardized columns including 'unit' and 'treatment'.

    Returns
    -------
    list of str
        List of unit names that have treatment == 1 at any time point.
    """
    if "treated_unit" in df.columns:
        # Legacy column was renamed
        treated_mask = cast(pd.Series, df["treated_unit"]) == 1
        units = cast(pd.Series, df.loc[treated_mask, "unit"])
        return units.drop_duplicates().tolist()
    elif "treatment" in df.columns:
        # Infer from treatment column
        treated_mask = cast(pd.Series, df["treatment"]) == 1
        units = cast(pd.Series, df.loc[treated_mask, "unit"])
        return units.drop_duplicates().tolist()
    else:
        # No treatment column - return all units
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


def _compute_autocorrelation(x: np.ndarray, lag: int) -> float:
    """
    Compute autocorrelation at a specific lag.

    Parameters
    ----------
    x : np.ndarray
        1D array of values.
    lag : int
        Lag for autocorrelation.

    Returns
    -------
    float
        Autocorrelation at the specified lag.
    """
    x = np.asarray(x)
    n = len(x)

    if n <= lag:
        return np.nan

    # Remove mean
    x_centered = x - np.nanmean(x)

    # Compute autocorrelation
    var = np.nansum(x_centered**2)
    if var == 0:
        return np.nan

    # Autocorrelation at lag
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
    """
    Create faceted histogram plot with p-value annotations.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with statistics for histograms.
    pvals_df : pd.DataFrame
        DataFrame with p-values for each facet.
    x_col : str
        Column name for x-axis values.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    facet_cols : list of str
        Columns to use for faceting (e.g., ['unit', 'group']).
    figsize : tuple
        Figure size (width, height).
    ncol : int
        Number of columns in facet grid.

    Returns
    -------
    matplotlib.Figure
        The figure object.
    """
    _setup_plot_style()

    # Get unique facet combinations
    if len(facet_cols) == 1:
        facet_keys = stats_df[facet_cols[0]].unique()
        facet_labels = [str(k) for k in facet_keys]
    else:
        facet_keys = stats_df[facet_cols].drop_duplicates().values.tolist()
        facet_labels = [" + ".join(str(v) for v in k) for k in facet_keys]

    n_facets = len(facet_keys)
    if n_facets == 0:
        # No data to plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return fig

    nrow = int(np.ceil(n_facets / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, (facet_key, facet_label) in enumerate(
        zip(facet_keys, facet_labels, strict=True)
    ):
        ax = axes[i]

        # Filter data for this facet
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

        # Draw histogram
        ax.hist(facet_data, bins=30, alpha=0.5, color="steelblue", edgecolor="white")

        # Draw vertical line at x=0
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)

        # Add p-value annotation
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

    # Hide unused axes
    for j in range(n_facets, len(axes)):
        axes[j].set_visible(False)

    # Set common labels
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.supxlabel(xlabel, fontsize=12)
    fig.supylabel("Count", fontsize=12)

    plt.tight_layout()

    return fig


def make_abs_ppc_plot(
    draws_df: pd.DataFrame,
    outcome_col: str = "outcome",
    categories: list[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    ppc_units: list[str] | None = None,
    ppc_exclude_units: list[str] | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """
    Posterior Predictive Check: Maximum Absolute Residual.

    Computes the maximum absolute residual for each unit/group/draw and compares
    observed vs predicted. Tests whether the model captures extreme deviations.

    Residuals:
    - obs_diff = outcome - exp(mu)
    - pred_diff = ypred - exp(mu)

    Statistic: max(|obs_diff|) vs max(|pred_diff|)
    P-value: proportion where observed max < predicted max

    Parameters
    ----------
    draws_df : pd.DataFrame
        DataFrame with MCMC draws. Must contain columns:
        - .draw: draw identifier
        - unit: panel entity (D dimension)
        - group: outcome category (K dimension)
        - treatment: binary treatment indicator
        - outcome (or the specified outcome_col): observed outcome
        - ypred: posterior predictive value
        - mu: counterfactual log-rate
    outcome_col : str, default='outcome'
        Name of the outcome column.
    categories : list of str, optional
        Categories (K dimension) to include. If None, uses all categories.
    figsize : tuple, default=(12, 8)
        Figure size (width, height).

    Returns
    -------
    fig : matplotlib.Figure
        Faceted histogram plot.
    pvals_df : pd.DataFrame
        DataFrame with columns: unit, group, pval
    """
    # Standardize column names
    df = _standardize_columns(draws_df)

    # Handle outcome column (might have custom name)
    if outcome_col not in df.columns:
        outcome_col = _detect_outcome_column(df)

    # Get categories
    if categories is None:
        categories = df["group"].unique().tolist()

    # Filter to categories
    df = df[df["group"].isin(categories)]

    # Identify treated units
    treated_units = _identify_treated_units(df)

    # Filter to control period only
    df_control = df[df["treatment"] == 0].copy()

    # Filter to configured units or default treated units.
    df_control = _filter_ppc_units(
        df_control,
        treated_units,
        ppc_units=ppc_units,
        ppc_exclude_units=ppc_exclude_units,
    )

    # Compute residuals
    df_control["pred_diff"] = df_control["ypred"] - np.exp(df_control["mu"])
    df_control["obs_diff"] = df_control[outcome_col] - np.exp(df_control["mu"])

    # Compute max absolute residual per unit/group/draw
    max_stats = (
        df_control.groupby(["unit", "group", ".draw"])
        .agg(
            max_pred_diff=("pred_diff", lambda x: np.nanmax(np.abs(x))),
            max_obs_diff=("obs_diff", lambda x: np.nanmax(np.abs(x))),
        )
        .reset_index()
    )

    max_stats["diff_in_diff"] = max_stats["max_obs_diff"] - max_stats["max_pred_diff"]

    # Compute p-values
    pvals_df = (
        max_stats.groupby(["unit", "group"])
        .agg(pval=("diff_in_diff", lambda x: np.mean(x < 0)))
        .reset_index()
    )

    # Filter pvals to categories and selected units
    selected_units = df_control["unit"].unique().tolist()
    pvals_df = pvals_df[pvals_df["group"].isin(categories)]
    pvals_df = pvals_df[pvals_df["unit"].isin(selected_units)]

    # Create faceted plot
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
    """
    Posterior Predictive Check: Autocorrelation of Residuals.

    Computes autocorrelation of residuals at the specified lag for each
    unit/group/draw and compares observed vs predicted.

    Residuals:
    - obs_diff = outcome - exp(mu)
    - pred_diff = ypred - exp(mu)

    Statistic: acf(obs_diff, lag) vs acf(pred_diff, lag)
    P-value: proportion where (obs_acf - pred_acf) < 0

    Parameters
    ----------
    draws_df : pd.DataFrame
        DataFrame with MCMC draws. See make_abs_ppc_plot for required columns.
    lag : int, default=6
        Lag for autocorrelation computation.
    outcome_col : str, default='outcome'
        Name of the outcome column.
    categories : list of str, optional
        Categories (K dimension) to include. If None, uses all categories.
    figsize : tuple, default=(12, 8)
        Figure size (width, height).

    Returns
    -------
    fig : matplotlib.Figure
        Faceted histogram plot.
    pvals_df : pd.DataFrame
        DataFrame with columns: unit, group, pval
    """
    # Standardize column names
    df = _standardize_columns(draws_df)

    # Handle outcome column
    if outcome_col not in df.columns:
        outcome_col = _detect_outcome_column(df)

    # Get categories
    if categories is None:
        categories = df["group"].unique().tolist()

    # Filter to categories
    df = df[df["group"].isin(categories)]

    # Identify treated units
    treated_units = _identify_treated_units(df)

    # Filter to control period and configured/default units
    df_control = df[df["treatment"] == 0].copy()
    df_control = _filter_ppc_units(
        df_control,
        treated_units,
        ppc_units=ppc_units,
        ppc_exclude_units=ppc_exclude_units,
    )

    # Compute residuals
    df_control["pred_diff"] = df_control["ypred"] - np.exp(df_control["mu"])
    df_control["obs_diff"] = df_control[outcome_col] - np.exp(df_control["mu"])

    # Sort by time within each group for proper ACF computation
    if "time" in df_control.columns:
        df_control = df_control.sort_values(["unit", "group", ".draw", "time"])

    # Compute ACF per unit/group/draw
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

    # Remove NaN values
    acf_stats = acf_stats.dropna(subset=["diff_in_ac"])

    # Compute p-values
    pvals_df = (
        acf_stats.groupby(["unit", "group"])
        .agg(pval=("diff_in_ac", lambda x: np.mean(x < 0)))
        .reset_index()
    )

    selected_units = df_control["unit"].unique().tolist()
    pvals_df = pvals_df[pvals_df["group"].isin(categories)]
    pvals_df = pvals_df[pvals_df["unit"].isin(selected_units)]

    # Create faceted plot
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
    """
    Posterior Predictive Check: RMSE of Residuals.

    Computes RMSE of residuals for each unit/group/draw and compares
    observed vs predicted.

    Residuals:
    - obs_diff = outcome - exp(mu)
    - pred_diff = ypred - exp(mu)

    Statistic: sqrt(mean(obs_diff^2)) vs sqrt(mean(pred_diff^2))
    P-value: proportion where observed RMSE < predicted RMSE

    Parameters
    ----------
    draws_df : pd.DataFrame
        DataFrame with MCMC draws. See make_abs_ppc_plot for required columns.
    outcome_col : str, default='outcome'
        Name of the outcome column.
    categories : list of str, optional
        Categories (K dimension) to include. If None, uses all categories.
    figsize : tuple, default=(12, 8)
        Figure size (width, height).

    Returns
    -------
    fig : matplotlib.Figure
        Faceted histogram plot.
    pvals_df : pd.DataFrame
        DataFrame with columns: unit, group, pval
    """
    # Standardize column names
    df = _standardize_columns(draws_df)

    # Handle outcome column
    if outcome_col not in df.columns:
        outcome_col = _detect_outcome_column(df)

    # Get categories
    if categories is None:
        categories = df["group"].unique().tolist()

    # Filter to categories
    df = df[df["group"].isin(categories)]

    # Identify treated units
    treated_units = _identify_treated_units(df)

    # Filter to control period and configured/default units.
    df_control = df[df["treatment"] == 0].copy()
    df_control = _filter_ppc_units(
        df_control,
        treated_units,
        ppc_units=ppc_units,
        ppc_exclude_units=ppc_exclude_units,
    )

    # Compute residuals
    df_control["pred_diff"] = df_control["ypred"] - np.exp(df_control["mu"])
    df_control["obs_diff"] = df_control[outcome_col] - np.exp(df_control["mu"])

    # Compute RMSE per unit/group/draw
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

    # Compute p-values (filter to treated units for display)
    pvals_df = (
        rmse_stats.groupby(["unit", "group"])
        .agg(pval=("diff_in_diff", lambda x: np.mean(x < 0)))
        .reset_index()
    )

    selected_units = df_control["unit"].unique().tolist()
    pvals_df = pvals_df[pvals_df["group"].isin(categories)]
    pvals_df = pvals_df[pvals_df["unit"].isin(selected_units)]

    # Filter stats to selected units for plotting
    rmse_stats_plot = rmse_stats[rmse_stats["unit"].isin(selected_units)]
    rmse_stats_plot = rmse_stats_plot[rmse_stats_plot["group"].isin(categories)]

    # Create faceted plot
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
    """
    Posterior Predictive Check: Cross-Unit Correlation (Spectral Norm).

    Computes the spectral norm (largest eigenvalue) of the correlation matrix
    of residuals across units (D dimension) for each time point. Tests whether
    the model captures cross-sectional dependence.

    Residuals:
    - obs_diff = outcome - exp(mu)
    - pred_diff = ypred - exp(mu)

    Statistic: sqrt(max eigenvalue of correlation matrix)
    P-value: proportion where observed spectral norm < predicted spectral norm

    Parameters
    ----------
    draws_df : pd.DataFrame
        DataFrame with MCMC draws. See make_abs_ppc_plot for required columns.
    max_treat_date : str, optional
        Maximum date to include (filters time < max_treat_date).
        Format: 'YYYY-MM-DD'. If None, uses control period only.
    outcome_col : str, default='outcome'
        Name of the outcome column.
    categories : list of str, optional
        Categories (K dimension) to include. If None, uses all categories.
    ndraws : int, default=1000
        Maximum number of draws to use (for computational efficiency).
    figsize : tuple, default=(10, 6)
        Figure size (width, height).

    Returns
    -------
    fig : matplotlib.Figure
        Faceted histogram plot (by category only).
    pvals_df : pd.DataFrame
        DataFrame with columns: group, pval
    """
    # Standardize column names
    df = _standardize_columns(draws_df)

    # Handle outcome column
    if outcome_col not in df.columns:
        outcome_col = _detect_outcome_column(df)

    # Get categories
    if categories is None:
        categories = df["group"].unique().tolist()

    # Filter to categories
    df = df[df["group"].isin(categories)]

    if ppc_units is not None or ppc_exclude_units is not None:
        all_units = cast(pd.Series, df["unit"]).drop_duplicates().tolist()
        df = _filter_ppc_units(
            df,
            treated_units=all_units,
            ppc_units=ppc_units,
            ppc_exclude_units=ppc_exclude_units,
        )

    # Filter by max_treat_date if provided
    if max_treat_date is not None and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df[df["time"] < pd.to_datetime(max_treat_date)]
    elif "treatment" in df.columns:
        # Use control period only
        df = df[df["treatment"] == 0]

    # Limit draws for computational efficiency
    if ".draw" in df.columns:
        unique_draws = df[".draw"].unique()
        if len(unique_draws) > ndraws:
            selected_draws = unique_draws[:ndraws]
            df = df[df[".draw"].isin(selected_draws)]

    # Compute residuals
    df["obs_residual"] = df[outcome_col] - np.exp(df["mu"])
    df["pred_residual"] = df["ypred"] - np.exp(df["mu"])

    # Filter out units with too much missing data (>25%)
    na_frac = df.groupby(["unit", "group"])[outcome_col].apply(
        lambda x: x.isna().mean()
    )
    na_frac = na_frac.reset_index(name="na_frac")
    valid_units = na_frac[na_frac["na_frac"] < 0.25][
        ["unit", "group"]
    ].drop_duplicates()
    df = df.merge(valid_units, on=["unit", "group"], how="inner")

    def compute_spectral_norm(residuals_matrix: np.ndarray) -> float:
        """Compute spectral norm from correlation matrix."""
        # Remove columns with all NaN
        valid_cols = ~np.all(np.isnan(residuals_matrix), axis=0)
        residuals_matrix = residuals_matrix[:, valid_cols]

        if residuals_matrix.shape[1] < 2:
            return np.nan

        # Compute correlation matrix with pairwise complete observations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_matrix = np.corrcoef(residuals_matrix.T)

        if np.any(np.isnan(corr_matrix)):
            # Handle NaN in correlation matrix
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 1.0)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        max_eigenvalue = np.max(eigenvalues)

        return np.sqrt(max(max_eigenvalue, 0))

    # Compute spectral norm per category/draw
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

            # Create residual matrices (time x unit)
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
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "Insufficient data for spectral norm computation",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Difference in Unit Correlations")
        return fig, pd.DataFrame(columns=["group", "pval"])

    # Compute p-values
    pvals_df = (
        eval_stats.groupby("group")
        .agg(pval=("eval_diff", lambda x: np.mean(x < 0)))
        .reset_index()
    )

    pvals_df = pvals_df[pvals_df["group"].isin(categories)]

    # Create faceted plot (by category only)
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
    """
    Create a time series plot showing rates by treatment group.

    Creates a line plot with different colors for Treated, Control, and optionally
    a specific unit as a separate group. Supports smoothing and treatment date markers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: unit, time, group, outcome, denominator, treatment
        (or legacy column names: state, category, population, exposure_code).
        The 'unit' column represents the D dimension (panel entities such as
        states, hospitals, counties, firms, etc.).
    group : str, optional
        Group/category (K dimension) to filter to (e.g., 'total', 'usborn').
        If None, aggregates all groups together.
    unit_col : str, default='unit'
        Name of the column containing unit identifiers (D dimension).
        Use this if your data has a different column name for the panel entity.
    rate_multiplier : float, default=1000
        Multiplier for rate calculation (e.g., 1000 = per 1,000 population).
    treatment_dates : dict, optional
        Dictionary of {label: date} for vertical marker lines.
        Example: {'Policy Change': '2022-06-24', 'Implementation': '2022-09-01'}
    separate_unit : str, optional
        If provided, show this specific unit as a separate group from other
        treated units. Useful for highlighting a particular unit of interest.
    smooth_window : int, optional
        Rolling window size for smoothing. If None, no smoothing applied.
    plot_type : str, default='rate'
        What to plot on y-axis: 'rate' (outcome/denominator * rate_multiplier)
        or 'count' (raw outcome counts).
    figsize : tuple, default=(10, 6)
        Figure size (width, height).

    Returns
    -------
    fig : matplotlib.Figure
        The figure object.
    ax : matplotlib.Axes
        The axes object.

    Examples
    --------
    >>> fig, ax = make_raw_rate_plot(df, group="total", rate_multiplier=1000)
    >>> fig.savefig("rate_plot.png")

    >>> # Separate a specific unit
    >>> fig, ax = make_raw_rate_plot(
    ...     df,
    ...     group="total",
    ...     treatment_dates={"Policy": "2022-06-24"},
    ...     separate_unit="TX",
    ...     smooth_window=3,
    ... )
    """
    _setup_plot_style()

    # Standardize column names
    df = _standardize_columns(df)

    # Handle custom unit column name
    if unit_col != "unit" and unit_col in df.columns:
        df = df.rename(columns={unit_col: "unit"})

    # Ensure time is datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # Filter to specific group if provided
    if group is not None and "group" in df.columns:
        df = df[df["group"] == group].copy()

    # Identify treated units
    treated_units = _identify_treated_units(df)

    # Assign treatment group labels
    def assign_treatment_group(row):
        if separate_unit is not None and row["unit"] == separate_unit:
            return separate_unit
        elif row["unit"] in treated_units:
            return "Treated"
        else:
            return "Control"

    df["treatment_group"] = df.apply(assign_treatment_group, axis=1)

    # Aggregate by treatment_group and time
    agg_df = (
        df.groupby(["treatment_group", "time"])
        .agg(outcome=("outcome", "sum"), denominator=("denominator", "sum"))
        .reset_index()
    )

    # Compute y-value based on plot_type
    if plot_type == "count":
        agg_df["y_value"] = agg_df["outcome"]
        ylabel = "Count"
    else:  # 'rate' (default)
        agg_df["y_value"] = (
            agg_df["outcome"] / agg_df["denominator"]
        ) * rate_multiplier
        ylabel = f"Rate per {rate_multiplier:,.0f}"

    # Apply smoothing if requested
    if smooth_window is not None and smooth_window > 1:
        agg_df = agg_df.sort_values(["treatment_group", "time"])
        agg_df["y_smooth"] = agg_df.groupby("treatment_group")["y_value"].transform(
            lambda x: x.rolling(window=smooth_window, center=True, min_periods=1).mean()
        )
    else:
        agg_df["y_smooth"] = agg_df["y_value"]

    # Sort by time
    agg_df = agg_df.sort_values("time")

    # Define colors
    colors = {
        "Treated": "#E41A1C",  # Red
        "Control": "#999999",  # Gray
    }
    # Add color for separate_unit if specified
    if separate_unit is not None:
        colors[separate_unit] = "#FF7F00"  # Orange

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot order: Control first (in back), then Treated, then separate_unit
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

        # If smoothed, also show raw data as faint points
        if smooth_window is not None and smooth_window > 1:
            ax.scatter(
                group_data["time"],
                group_data["y_value"],
                color=color,
                alpha=0.2,
                s=10,
            )

    # Add treatment date markers with rotated labels
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
            # Add rotated label near top of plot
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

    # Styling
    group_label = f" ({group})" if group else ""
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel(f"{ylabel}{group_label}", fontsize=11)
    title_type = "Rate" if plot_type == "rate" else "Count"
    ax.set_title(
        f"{title_type} by Treatment Group{group_label}", fontsize=13, fontweight="bold"
    )
    ax.legend(loc="best", frameon=True, fancybox=True)

    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    return fig, ax


def make_group_comparison_plot(
    df: pd.DataFrame,
    groups: list[str] | None = None,
    rate_multiplier: float = 1000,
    treatment_dates: dict[str, str] | None = None,
    plot_type: str = "rate",  # 'rate' or 'count'
    figsize: tuple[int, int] = (12, 8),
) -> tuple[Figure, np.ndarray]:
    """
    Create a faceted plot comparing rates across different outcome groups.

    Each subplot shows treated vs control time series for a different group/category
    (K dimension).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: unit, time, group, outcome, denominator, treatment
        (or legacy column names).
    groups : list of str, optional
        List of groups (K dimension) to plot. If None, uses all unique groups.
    rate_multiplier : float, default=1000
        Multiplier for rate calculation.
    treatment_dates : dict, optional
        Dictionary of {label: date} for vertical marker lines.
    plot_type : str, default='rate'
        What to plot: 'rate' (outcome/denominator * multiplier) or 'count' (raw counts).
    figsize : tuple, default=(12, 8)
        Figure size (width, height).

    Returns
    -------
    fig : matplotlib.Figure
        The figure object.
    axes : np.ndarray
        Array of axes objects.

    Examples
    --------
    >>> fig, axes = make_group_comparison_plot(
    ...     df, groups=["usborn", "foreign"], treatment_dates={"Policy": "2022-06-24"}
    ... )
    >>> fig.savefig("group_comparison.png")
    """
    _setup_plot_style()

    # Standardize column names
    df = _standardize_columns(df)

    # Ensure time is datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # Get groups to plot
    if groups is None:
        groups = df["group"].unique().tolist()

    n_groups = len(groups)
    if n_groups == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No groups to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig, np.array([ax])

    # Determine subplot layout
    ncols = min(2, n_groups)
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True)
    axes_flat = axes.flatten()

    # Identify treated units
    treated_units = _identify_treated_units(df)

    # Colors
    colors = {
        "Treated": "#E41A1C",
        "Control": "#999999",
    }

    for idx, group in enumerate(groups):
        ax = axes_flat[idx]

        # Filter to this group
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

        # Assign treatment group
        group_df["treatment_group"] = group_df["unit"].apply(
            lambda u: "Treated" if u in treated_units else "Control"
        )

        # Aggregate by treatment_group and time
        agg_df = (
            group_df.groupby(["treatment_group", "time"])
            .agg(outcome=("outcome", "sum"), denominator=("denominator", "sum"))
            .reset_index()
        )

        # Compute y-value based on plot_type
        if plot_type == "count":
            agg_df["y_value"] = agg_df["outcome"]
            ylabel = "Count"
        else:  # 'rate'
            agg_df["y_value"] = (
                agg_df["outcome"] / agg_df["denominator"]
            ) * rate_multiplier
            ylabel = f"Rate per {rate_multiplier:,.0f}"
        agg_df = agg_df.sort_values("time")

        # Plot each treatment group
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

        # Add treatment date markers with colors
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

        # Only add legend to first subplot
        if idx == 0:
            ax.legend(loc="best", frameon=True, fontsize=8)

    # Hide unused axes
    for idx in range(len(groups), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Set common x-label
    fig.supxlabel("Time", fontsize=11)
    title_type = "Rate" if plot_type == "rate" else "Count"
    fig.suptitle(
        f"{title_type} Comparison by Group", fontsize=13, fontweight="bold", y=1.02
    )

    # Rotate x-axis labels
    for ax in axes_flat[: len(groups)]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    return fig, axes


def make_unit_fit_plot(
    quantiles_df: pd.DataFrame,
    unit_name: str,
    group: str = "total",
    outcome_col: str = "outcome",
    figsize: tuple[int, int] = (10, 6),
) -> tuple[Figure, Axes]:
    """
    Make fit plot showing observed vs predicted over time for a specific unit.

    Parameters
    ----------
    quantiles_df : pd.DataFrame
        DataFrame with quantiles (ypred_mean, ypred_lower, ypred_upper).
    unit_name : str
        Name of the unit to plot.
    group : str, default='total'
        Group to filter.
    outcome_col : str, default='outcome'
        Outcome variable name.
    figsize : tuple, default=(10, 6)
        Figure size.

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes
    """
    _setup_plot_style()
    sns.set_palette("husl")
    df = _standardize_columns(quantiles_df)

    if outcome_col not in df.columns:
        legacy_names = ["births", "count", "y"]
        for lname in legacy_names:
            if lname in df.columns:
                outcome_col = lname
                break

    df_plot = df[(df["group"] == group) & (df["unit"] == unit_name)].copy()
    if df_plot.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, f"No data for {unit_name} / {group}", ha="center", va="center"
        )
        ax.set_title(f"{unit_name} ({group})")
        return fig, ax

    if "time" in df_plot.columns and not pd.api.types.is_datetime64_any_dtype(
        df_plot["time"]
    ):
        df_plot["time"] = pd.to_datetime(df_plot["time"])

    df_plot = df_plot.sort_values("time")

    # Drop rows where prediction or outcome is missing (NaN) to avoid spurious lines
    df_pred = df_plot.dropna(subset=["ypred_mean", "ypred_lower", "ypred_upper"])
    df_obs = df_plot.dropna(subset=[outcome_col])

    fig, ax = plt.subplots(figsize=figsize)

    # Credible Interval (only over timepoints with predictions)
    if not df_pred.empty:
        ax.fill_between(
            df_pred["time"],
            df_pred["ypred_lower"],
            df_pred["ypred_upper"],
            alpha=0.3,
            color=sns.color_palette()[0],
            label="95% CI",
        )

        # Mean Prediction line — break at gaps in time index
        sns.lineplot(
            data=df_pred,
            x="time",
            y="ypred_mean",
            color=sns.color_palette()[3],
            linewidth=2,
            label="Predicted Mean",
            ax=ax,
        )

    # Observed Data — only plot finite observations
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

    # Treatment line
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

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    return fig, ax


def make_unit_gap_plot(
    quantiles_df: pd.DataFrame,
    unit_name: str,
    group: str = "total",
    outcome_col: str = "outcome",
    figsize: tuple[int, int] = (10, 6),
) -> tuple[Figure, Axes]:
    """
    Make gap plot showing difference between observed and predicted relative to prediction
    for a specific unit.

    Parameters
    ----------
    quantiles_df : pd.DataFrame
        DataFrame with quantiles (ypred_mean, ypred_lower, ypred_upper).
    unit_name : str
        Name of the unit to plot.
    group : str, default='total'
        Group to filter.
    outcome_col : str, default='outcome'
        Outcome variable name.
    figsize : tuple, default=(10, 6)
        Figure size.

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axes
    """
    _setup_plot_style()
    df = _standardize_columns(quantiles_df)

    if outcome_col not in df.columns:
        legacy_names = ["births", "count", "y"]
        for lname in legacy_names:
            if lname in df.columns:
                outcome_col = lname
                break

    df_plot = df[(df["group"] == group) & (df["unit"] == unit_name)].copy()
    if df_plot.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, f"No data for {unit_name} / {group}", ha="center", va="center"
        )
        ax.set_title(f"Gap: {unit_name} ({group})")
        return fig, ax

    if "time" in df_plot.columns and not pd.api.types.is_datetime64_any_dtype(
        df_plot["time"]
    ):
        df_plot["time"] = pd.to_datetime(df_plot["time"])

    df_plot = df_plot.sort_values("time")

    # Drop rows where outcome or predictions are NaN/non-positive (ratio undefined)
    df_plot = df_plot.dropna(
        subset=[outcome_col, "ypred_mean", "ypred_lower", "ypred_upper"]
    )
    df_plot = df_plot[
        (df_plot["ypred_mean"] > 0)
        & (df_plot["ypred_lower"] > 0)
        & (df_plot["ypred_upper"] > 0)
    ]

    if df_plot.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid data points", ha="center", va="center")
        return fig, ax

    # Treatment Date
    t_date = None
    if (
        "treated_unit" in df_plot.columns
        and df_plot["treated_unit"].iloc[0]
        and "treatment" in df_plot.columns
    ):
        treated_times = df_plot[df_plot["treatment"] == 1]["time"]
        if not treated_times.empty:
            t_date = treated_times.iloc[0]

    fig, ax = plt.subplots(figsize=figsize)

    gap_mean = df_plot[outcome_col] / df_plot["ypred_mean"] - 1
    gap_lower = df_plot[outcome_col] / df_plot["ypred_upper"] - 1
    gap_upper = df_plot[outcome_col] / df_plot["ypred_lower"] - 1

    color_ribbon = sns.color_palette("muted")[0]
    color_line = sns.color_palette("dark")[3]

    if t_date is not None:
        # Separate pre/post so ribbon and line do not connect across the intervention
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

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

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
    """
    Generate interval plots showing causal effects with credible intervals using seaborn aesthetics.

    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame with draws (e.g., .draw, time, unit, group, mu, mu_treated, etc.)
    units : list of str, optional
        Units to include.
    group_var : str, default='unit'
        Variable to aggregate by.
    categories : list of str, optional
        Categories to include.
    outcome_col : str, default='outcome'
        Outcome variable name.
    denom_col : str, default='denominator'
        Denominator variable name.
    rate_normalizer : float, default=1000
    estimand : str, default='diff' ('diff' or 'ratio')
    method : str, default='mu' ('pred' or 'mu')
    x_var : str, default='unit'
    color_group : str, optional
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    _setup_plot_style()
    df = _standardize_columns(merged_df)

    if color_group is None:
        color_group = "group" if group_var == "unit" else "unit"

    if outcome_col not in df.columns:
        legacy_names = ["births", "count", "y"]
        for lname in legacy_names:
            if lname in df.columns:
                outcome_col = lname
                break

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

    # Compute unit-level effect
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
            denom_val = np.mean(grp[denom_col] * grp["years"]) if len(grp) > 0 else 1

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

    # Sort
    sort_order = plot_df.groupby(x_var)["median"].median().sort_values().index
    plot_df[x_var] = pd.Categorical(plot_df[x_var], categories=sort_order, ordered=True)
    plot_df = plot_df.sort_values(x_var)

    fig, ax = plt.subplots(figsize=figsize)
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

    plt.tight_layout()
    return fig, ax


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

    if outcome_col not in df.columns:
        legacy_names = ["births", "count", "y"]
        for lname in legacy_names:
            if lname in df.columns:
                outcome_col = lname
                break

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
        res = {
            "ypred": np.sum(g["ypred"]),
            "outcome": np.sum(g[outcome_col]),
            "treated": np.sum(np.exp(g["mu_treated"]))
            if "mu_treated" in g.columns
            else np.nan,
            "untreated": np.sum(np.exp(g["mu"])) if "mu" in g.columns else np.nan,
            "denom_val": np.sum(g[denom_col] * g["years"]),
        }
        return pd.Series(res)

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
    """
    Generate all PPC plots and optionally save to files.

    Parameters
    ----------
    draws_df : pd.DataFrame
        DataFrame with MCMC draws.
    output_dir : str, optional
        Directory to save plots. If None, plots are not saved.
    outcome_col : str, default='outcome'
        Name of the outcome column.
    categories : list of str, optional
        Categories (K dimension) to include.
    figsize : tuple, default=(12, 8)
        Figure size for plots.
    acf_lag : int, default=6
        Lag for ACF computation.
    max_treat_date : str, optional
        Maximum date for unit correlation plot.
    ndraws : int, default=1000
        Number of draws for unit correlation plot.

    Returns
    -------
    dict
        Dictionary with keys 'abs', 'acf', 'rmse', 'unit_corr'.
        Each value is a dict with 'fig' and 'pvals' keys.
    """
    results = {}

    logger.info("Generating PPC plots...")

    # Maximum absolute residual
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

    # ACF
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

    # RMSE
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

    # Unit correlation
    logger.info("  - Unit correlation plot...")
    fig_corr, pvals_corr = make_unit_corr_ppc_plot(
        draws_df,
        max_treat_date=max_treat_date,
        outcome_col=outcome_col,
        categories=categories,
        ndraws=ndraws,
        figsize=(10, 6),
        ppc_units=ppc_units,
        ppc_exclude_units=ppc_exclude_units,
    )
    results["unit_corr"] = {"fig": fig_corr, "pvals": pvals_corr}
    logger.debug(f"    Generated {len(pvals_corr)} p-values for unit correlation check")

    # Save plots if output_dir provided
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

        # Save p-values to CSV
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
