"""
Visualization utilities for bayesian_panel_nmf.

This module provides two types of visualization functions:

1. **PPC (Posterior Predictive Check) Plots**: Diagnostic plots comparing observed vs 
   predicted statistics in the control period. These help assess model fit.
   - make_abs_ppc_plot: Maximum absolute residual comparison
   - make_acf_ppc_plot: Autocorrelation of residuals at specified lag
   - make_rmse_ppc_plot: RMSE comparison (observed vs predicted)
   - make_unit_corr_ppc_plot: Cross-unit correlation (spectral norm)
   - make_all_ppc_plots: Convenience function to generate all PPC plots

2. **Time Series Plots**: Visualizations for exploring the data and model results.
   - make_raw_rate_plot: Rate by treatment group over time
   - make_group_comparison_plot: Faceted comparison of rates across outcome groups
   - make_state_fit_plot: Observed vs predicted for a specific unit with CI

The module handles BOTH standardized and legacy column names:
- Standardized: unit, group, denominator, treatment
- Legacy: state, category, population, exposure_code

PPC functions return (fig, pvals_df) tuples where fig is a matplotlib Figure
and pvals_df is a DataFrame with p-values. Time series functions return (fig, ax/axes).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List, Tuple, Union
import warnings


# =============================================================================
# Column Name Standardization
# =============================================================================

# Mapping from legacy column names to standardized names
_COLUMN_MAPPING = {
    'state': 'unit',
    'category': 'group',
    'population': 'denominator',
    'exposure_code': 'treatment',
    'banned_state': 'treated_unit',  # derived column
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


def _identify_treated_units(df: pd.DataFrame) -> List[str]:
    """
    Identify units (states) that have treatment at any time point.
    
    In the R code, this is the 'banned_state' concept - states that have
    exposure_code == 1 at some point.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with standardized columns including 'unit' and 'treatment'.
        
    Returns
    -------
    list of str
        List of unit names that have treatment == 1 at any time point.
    """
    if 'treated_unit' in df.columns:
        # Legacy column was renamed
        return df[df['treated_unit'] == 1]['unit'].unique().tolist()
    elif 'treatment' in df.columns:
        # Infer from treatment column
        treated = df[df['treatment'] == 1]['unit'].unique().tolist()
        return treated
    else:
        # No treatment column - return all units
        return df['unit'].unique().tolist()


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
    var = np.nansum(x_centered ** 2)
    if var == 0:
        return np.nan
    
    # Autocorrelation at lag
    acf = np.nansum(x_centered[:-lag] * x_centered[lag:]) / var
    
    return acf


def _setup_plot_style():
    """Set up matplotlib style similar to ggplot2 theme_bw()."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'grid.color': 'lightgray',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
    })


def _create_faceted_histograms(
    stats_df: pd.DataFrame,
    pvals_df: pd.DataFrame,
    x_col: str,
    title: str,
    xlabel: str,
    facet_cols: List[str],
    figsize: Tuple[int, int],
    ncol: int = 3,
) -> plt.Figure:
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
        facet_labels = [' + '.join(str(v) for v in k) for k in facet_keys]
    
    n_facets = len(facet_keys)
    if n_facets == 0:
        # No data to plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    nrow = int(np.ceil(n_facets / ncol))
    
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, (facet_key, facet_label) in enumerate(zip(facet_keys, facet_labels)):
        ax = axes[i]
        
        # Filter data for this facet
        if len(facet_cols) == 1:
            mask = stats_df[facet_cols[0]] == facet_key
            pval_mask = pvals_df[facet_cols[0]] == facet_key
        else:
            mask = np.all([stats_df[col] == val for col, val in zip(facet_cols, facet_key)], axis=0)
            pval_mask = np.all([pvals_df[col] == val for col, val in zip(facet_cols, facet_key)], axis=0)
        
        facet_data = stats_df.loc[mask, x_col].dropna()
        
        if len(facet_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(facet_label)
            continue
        
        # Draw histogram
        ax.hist(facet_data, bins=30, alpha=0.5, color='steelblue', edgecolor='white')
        
        # Draw vertical line at x=0
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        
        # Add p-value annotation
        pval_row = pvals_df.loc[pval_mask]
        if len(pval_row) > 0:
            pval = pval_row['pval'].values[0]
            ax.text(
                0.95, 0.95, f'{pval:.3f}',
                transform=ax.transAxes,
                ha='right', va='top',
                color='red', fontsize=10, fontweight='bold'
            )
        
        ax.set_title(facet_label, fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Set common labels
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.supxlabel(xlabel, fontsize=12)
    fig.supylabel('Count', fontsize=12)
    
    plt.tight_layout()
    
    return fig


# =============================================================================
# PPC Plot Functions
# =============================================================================

def make_abs_ppc_plot(
    draws_df: pd.DataFrame,
    outcome_col: str = 'outcome',
    categories: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[plt.Figure, pd.DataFrame]:
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
        - unit (or state): panel entity
        - group (or category): outcome category
        - treatment (or exposure_code): binary treatment indicator
        - outcome (or the specified outcome_col): observed outcome
        - ypred: posterior predictive value
        - mu: counterfactual log-rate
    outcome_col : str, default='outcome'
        Name of the outcome column.
    categories : list of str, optional
        Categories to include. If None, uses all categories.
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
    if outcome_col not in df.columns and 'outcome' in df.columns:
        outcome_col = 'outcome'
    elif outcome_col not in df.columns:
        # Try legacy names
        for legacy in ['births', 'count', 'y']:
            if legacy in df.columns:
                outcome_col = legacy
                break
    
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in DataFrame. "
                         f"Available columns: {list(df.columns)}")
    
    # Get categories
    if categories is None:
        categories = df['group'].unique().tolist()
    
    # Filter to categories
    df = df[df['group'].isin(categories)]
    
    # Identify treated units (banned states)
    treated_units = _identify_treated_units(df)
    
    # Filter to control period only
    df_control = df[df['treatment'] == 0].copy()
    
    # Filter to treated units only
    df_control = df_control[df_control['unit'].isin(treated_units)]
    
    # Exclude aggregate states if present
    df_control = df_control[~df_control['unit'].str.contains('Ban States', case=False, na=False)]
    
    # Compute residuals
    df_control['pred_diff'] = df_control['ypred'] - np.exp(df_control['mu'])
    df_control['obs_diff'] = df_control[outcome_col] - np.exp(df_control['mu'])
    
    # Compute max absolute residual per unit/group/draw
    max_stats = df_control.groupby(['unit', 'group', '.draw']).agg(
        max_pred_diff=('pred_diff', lambda x: np.nanmax(np.abs(x))),
        max_obs_diff=('obs_diff', lambda x: np.nanmax(np.abs(x)))
    ).reset_index()
    
    max_stats['diff_in_diff'] = max_stats['max_obs_diff'] - max_stats['max_pred_diff']
    
    # Compute p-values
    pvals_df = max_stats.groupby(['unit', 'group']).agg(
        pval=('diff_in_diff', lambda x: np.mean(x < 0))
    ).reset_index()
    
    # Filter pvals to categories and treated units
    pvals_df = pvals_df[pvals_df['group'].isin(categories)]
    pvals_df = pvals_df[pvals_df['unit'].isin(treated_units)]
    
    # Create faceted plot
    fig = _create_faceted_histograms(
        stats_df=max_stats[max_stats['unit'].isin(treated_units)],
        pvals_df=pvals_df,
        x_col='diff_in_diff',
        title='Difference in Maximum Absolute Predicted Residual',
        xlabel='Observed - Predicted Max Residual',
        facet_cols=['unit', 'group'],
        figsize=figsize,
    )
    
    return fig, pvals_df


def make_acf_ppc_plot(
    draws_df: pd.DataFrame,
    lag: int = 6,
    outcome_col: str = 'outcome',
    categories: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[plt.Figure, pd.DataFrame]:
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
        Categories to include. If None, uses all categories.
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
    if outcome_col not in df.columns and 'outcome' in df.columns:
        outcome_col = 'outcome'
    elif outcome_col not in df.columns:
        for legacy in ['births', 'count', 'y']:
            if legacy in df.columns:
                outcome_col = legacy
                break
    
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in DataFrame.")
    
    # Get categories
    if categories is None:
        categories = df['group'].unique().tolist()
    
    # Filter to categories
    df = df[df['group'].isin(categories)]
    
    # Identify treated units
    treated_units = _identify_treated_units(df)
    
    # Filter to control period and treated units
    df_control = df[df['treatment'] == 0].copy()
    df_control = df_control[df_control['unit'].isin(treated_units)]
    df_control = df_control[~df_control['unit'].str.contains('Ban States', case=False, na=False)]
    
    # Compute residuals
    df_control['pred_diff'] = df_control['ypred'] - np.exp(df_control['mu'])
    df_control['obs_diff'] = df_control[outcome_col] - np.exp(df_control['mu'])
    
    # Sort by time within each group for proper ACF computation
    if 'time' in df_control.columns:
        df_control = df_control.sort_values(['unit', 'group', '.draw', 'time'])
    
    # Compute ACF per unit/group/draw
    def compute_acf_stats(group_df):
        obs_vals = group_df['obs_diff'].values
        pred_vals = group_df['pred_diff'].values
        
        obs_acf = _compute_autocorrelation(obs_vals, lag)
        pred_acf = _compute_autocorrelation(pred_vals, lag)
        
        return pd.Series({
            'obs_ac': obs_acf,
            'pred_ac': pred_acf,
            'diff_in_ac': obs_acf - pred_acf
        })
    
    acf_stats = df_control.groupby(['unit', 'group', '.draw']).apply(
        compute_acf_stats, include_groups=False
    ).reset_index()
    
    # Remove NaN values
    acf_stats = acf_stats.dropna(subset=['diff_in_ac'])
    
    # Compute p-values
    pvals_df = acf_stats.groupby(['unit', 'group']).agg(
        pval=('diff_in_ac', lambda x: np.mean(x < 0))
    ).reset_index()
    
    pvals_df = pvals_df[pvals_df['group'].isin(categories)]
    pvals_df = pvals_df[pvals_df['unit'].isin(treated_units)]
    
    # Create faceted plot
    fig = _create_faceted_histograms(
        stats_df=acf_stats,
        pvals_df=pvals_df,
        x_col='diff_in_ac',
        title=f'Difference in Residual Autocorrelation (Lag {lag})',
        xlabel='Observed - Predicted Autocorrelation',
        facet_cols=['unit', 'group'],
        figsize=figsize,
    )
    
    return fig, pvals_df


def make_rmse_ppc_plot(
    draws_df: pd.DataFrame,
    outcome_col: str = 'outcome',
    categories: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[plt.Figure, pd.DataFrame]:
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
        Categories to include. If None, uses all categories.
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
    if outcome_col not in df.columns and 'outcome' in df.columns:
        outcome_col = 'outcome'
    elif outcome_col not in df.columns:
        for legacy in ['births', 'count', 'y']:
            if legacy in df.columns:
                outcome_col = legacy
                break
    
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in DataFrame.")
    
    # Get categories
    if categories is None:
        categories = df['group'].unique().tolist()
    
    # Filter to categories
    df = df[df['group'].isin(categories)]
    
    # Identify treated units
    treated_units = _identify_treated_units(df)
    
    # Filter to control period (all units, not just treated, per R code)
    df_control = df[df['treatment'] == 0].copy()
    
    # Compute residuals
    df_control['pred_diff'] = df_control['ypred'] - np.exp(df_control['mu'])
    df_control['obs_diff'] = df_control[outcome_col] - np.exp(df_control['mu'])
    
    # Compute RMSE per unit/group/draw
    rmse_stats = df_control.groupby(['unit', 'group', '.draw']).agg(
        rmse_pred_diff=('pred_diff', lambda x: np.sqrt(np.nanmean(x ** 2))),
        rmse_obs_diff=('obs_diff', lambda x: np.sqrt(np.nanmean(x ** 2)))
    ).reset_index()
    
    rmse_stats['diff_in_diff'] = rmse_stats['rmse_obs_diff'] - rmse_stats['rmse_pred_diff']
    
    # Compute p-values (filter to treated units for display)
    pvals_df = rmse_stats.groupby(['unit', 'group']).agg(
        pval=('diff_in_diff', lambda x: np.mean(x < 0))
    ).reset_index()
    
    pvals_df = pvals_df[pvals_df['group'].isin(categories)]
    pvals_df = pvals_df[pvals_df['unit'].isin(treated_units)]
    
    # Filter stats to treated units for plotting
    rmse_stats_plot = rmse_stats[rmse_stats['unit'].isin(treated_units)]
    rmse_stats_plot = rmse_stats_plot[rmse_stats_plot['group'].isin(categories)]
    
    # Create faceted plot
    fig = _create_faceted_histograms(
        stats_df=rmse_stats_plot,
        pvals_df=pvals_df,
        x_col='diff_in_diff',
        title='Difference in RMSE',
        xlabel='Observed - Predicted RMSE',
        facet_cols=['unit', 'group'],
        figsize=figsize,
    )
    
    return fig, pvals_df


def make_unit_corr_ppc_plot(
    draws_df: pd.DataFrame,
    max_treat_date: Optional[str] = None,
    outcome_col: str = 'outcome',
    categories: Optional[List[str]] = None,
    ndraws: int = 1000,
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Posterior Predictive Check: Cross-Unit Correlation (Spectral Norm).
    
    Computes the spectral norm (largest eigenvalue) of the correlation matrix
    of residuals across units for each time point. Tests whether the model
    captures cross-sectional dependence.
    
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
        Categories to include. If None, uses all categories.
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
    if outcome_col not in df.columns and 'outcome' in df.columns:
        outcome_col = 'outcome'
    elif outcome_col not in df.columns:
        for legacy in ['births', 'count', 'y']:
            if legacy in df.columns:
                outcome_col = legacy
                break
    
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in DataFrame.")
    
    # Get categories
    if categories is None:
        categories = df['group'].unique().tolist()
    
    # Filter to categories
    df = df[df['group'].isin(categories)]
    
    # Filter by max_treat_date if provided
    if max_treat_date is not None and 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df[df['time'] < pd.to_datetime(max_treat_date)]
    elif 'treatment' in df.columns:
        # Use control period only
        df = df[df['treatment'] == 0]
    
    # Limit draws for computational efficiency
    if '.draw' in df.columns:
        unique_draws = df['.draw'].unique()
        if len(unique_draws) > ndraws:
            selected_draws = unique_draws[:ndraws]
            df = df[df['.draw'].isin(selected_draws)]
    
    # Compute residuals
    df['obs_residual'] = df[outcome_col] - np.exp(df['mu'])
    df['pred_residual'] = df['ypred'] - np.exp(df['mu'])
    
    # Filter out units with too much missing data (>25%)
    na_frac = df.groupby(['unit', 'group'])[outcome_col].apply(lambda x: x.isna().mean())
    na_frac = na_frac.reset_index(name='na_frac')
    valid_units = na_frac[na_frac['na_frac'] < 0.25][['unit', 'group']].drop_duplicates()
    df = df.merge(valid_units, on=['unit', 'group'], how='inner')
    
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
        group_df = df[df['group'] == group]
        
        if len(group_df) == 0:
            continue
        
        unique_draws = group_df['.draw'].unique()
        units = group_df['unit'].unique()
        times = group_df['time'].unique() if 'time' in group_df.columns else [0]
        
        for draw in unique_draws:
            draw_df = group_df[group_df['.draw'] == draw]
            
            # Create residual matrices (time x unit)
            obs_matrix = np.full((len(times), len(units)), np.nan)
            pred_matrix = np.full((len(times), len(units)), np.nan)
            
            time_to_idx = {t: i for i, t in enumerate(times)}
            unit_to_idx = {u: i for i, u in enumerate(units)}
            
            for _, row in draw_df.iterrows():
                t_idx = time_to_idx.get(row.get('time', 0), 0)
                u_idx = unit_to_idx.get(row['unit'])
                if u_idx is not None:
                    obs_matrix[t_idx, u_idx] = row['obs_residual']
                    # Mask pred_residual where obs_residual is NaN
                    if not np.isnan(row['obs_residual']):
                        pred_matrix[t_idx, u_idx] = row['pred_residual']
            
            obs_sval = compute_spectral_norm(obs_matrix)
            pred_sval = compute_spectral_norm(pred_matrix)
            
            eval_results.append({
                'group': group,
                '.draw': draw,
                'obs_sval': obs_sval,
                'pred_sval': pred_sval,
                'eval_diff': obs_sval - pred_sval
            })
    
    eval_stats = pd.DataFrame(eval_results)
    eval_stats = eval_stats.dropna(subset=['eval_diff'])
    
    if len(eval_stats) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Insufficient data for spectral norm computation',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Difference in State Correlations')
        return fig, pd.DataFrame(columns=['group', 'pval'])
    
    # Compute p-values
    pvals_df = eval_stats.groupby('group').agg(
        pval=('eval_diff', lambda x: np.mean(x < 0))
    ).reset_index()
    
    pvals_df = pvals_df[pvals_df['group'].isin(categories)]
    
    # Create faceted plot (by category only)
    fig = _create_faceted_histograms(
        stats_df=eval_stats,
        pvals_df=pvals_df,
        x_col='eval_diff',
        title='Difference in State Correlations',
        xlabel='Observed - Predicted Spectral Norm',
        facet_cols=['group'],
        figsize=figsize,
        ncol=2,
    )
    
    return fig, pvals_df


# =============================================================================
# Time Series Visualization Functions
# =============================================================================

def make_raw_rate_plot(
    df: pd.DataFrame,
    group: Optional[str] = None,
    rate_multiplier: float = 1000,
    treatment_dates: Optional[dict] = None,
    separate_texas: bool = False,
    smooth_window: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a time series plot showing rates by treatment group.
    
    Creates a line plot with different colors for Treated, Control, and optionally
    Texas as a separate group. Supports smoothing and treatment date markers.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: unit, time, group, outcome, denominator, treatment
        (or legacy column names: state, category, population, exposure_code).
    group : str, optional
        Group/category to filter to (e.g., 'total', 'usborn'). If None, aggregates
        all groups together.
    rate_multiplier : float, default=1000
        Multiplier for rate calculation (e.g., 1000 = per 1,000 population).
    treatment_dates : dict, optional
        Dictionary of {label: date} for vertical marker lines.
        Example: {'Dobbs': '2022-06-24', 'Roe v Wade Overturned': '2022-06-24'}
    separate_texas : bool, default=False
        If True, show Texas as a separate group from other treated states.
    smooth_window : int, optional
        Rolling window size for smoothing. If None, no smoothing applied.
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
    >>> fig, ax = make_raw_rate_plot(df, group='total', rate_multiplier=1000)
    >>> fig.savefig('rate_plot.png')
    
    >>> fig, ax = make_raw_rate_plot(
    ...     df, 
    ...     group='total',
    ...     treatment_dates={'Dobbs': '2022-06-24'},
    ...     separate_texas=True,
    ...     smooth_window=3
    ... )
    """
    _setup_plot_style()
    
    # Standardize column names
    df = _standardize_columns(df)
    
    # Ensure time is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Filter to specific group if provided
    if group is not None and 'group' in df.columns:
        df = df[df['group'] == group].copy()
    
    # Identify treated units
    treated_units = _identify_treated_units(df)
    
    # Assign treatment group labels
    def assign_treatment_group(row):
        if separate_texas and row['unit'] == 'Texas':
            return 'Texas'
        elif row['unit'] in treated_units:
            return 'Treated'
        else:
            return 'Control'
    
    df['treatment_group'] = df.apply(assign_treatment_group, axis=1)
    
    # Aggregate by treatment_group and time
    agg_df = df.groupby(['treatment_group', 'time']).agg(
        outcome=('outcome', 'sum'),
        denominator=('denominator', 'sum')
    ).reset_index()
    
    # Compute rate
    agg_df['rate'] = (agg_df['outcome'] / agg_df['denominator']) * rate_multiplier
    
    # Apply smoothing if requested
    if smooth_window is not None and smooth_window > 1:
        agg_df = agg_df.sort_values(['treatment_group', 'time'])
        agg_df['rate_smooth'] = agg_df.groupby('treatment_group')['rate'].transform(
            lambda x: x.rolling(window=smooth_window, center=True, min_periods=1).mean()
        )
    else:
        agg_df['rate_smooth'] = agg_df['rate']
    
    # Sort by time
    agg_df = agg_df.sort_values('time')
    
    # Define colors
    colors = {
        'Treated': '#E41A1C',   # Red
        'Control': '#999999',   # Gray
        'Texas': '#FF7F00',     # Orange
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot order: Control first (in back), then Treated, then Texas
    plot_order = ['Control', 'Treated']
    if separate_texas:
        plot_order.append('Texas')
    
    for tgroup in plot_order:
        group_data = agg_df[agg_df['treatment_group'] == tgroup]
        if len(group_data) == 0:
            continue
        
        color = colors.get(tgroup, '#333333')
        linewidth = 2.0 if tgroup != 'Control' else 1.5
        alpha = 1.0 if tgroup != 'Control' else 0.7
        
        ax.plot(
            group_data['time'],
            group_data['rate_smooth'],
            label=tgroup,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        
        # If smoothed, also show raw data as faint points
        if smooth_window is not None and smooth_window > 1:
            ax.scatter(
                group_data['time'],
                group_data['rate'],
                color=color,
                alpha=0.2,
                s=10,
            )
    
    # Add treatment date markers
    if treatment_dates is not None:
        for label, date in treatment_dates.items():
            date = pd.to_datetime(date)
            ax.axvline(x=date, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
            # Add label at top of plot
            ax.text(
                date, ax.get_ylim()[1], f' {label}',
                ha='left', va='top', fontsize=9, rotation=0,
                color='black', alpha=0.8,
            )
    
    # Styling
    group_label = f' ({group})' if group else ''
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel(f'Rate per {rate_multiplier:,.0f}{group_label}', fontsize=11)
    ax.set_title(f'Rate by Treatment Group{group_label}', fontsize=13, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True)
    
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig, ax


def make_group_comparison_plot(
    df: pd.DataFrame,
    groups: Optional[List[str]] = None,
    rate_multiplier: float = 1000,
    treatment_dates: Optional[dict] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a faceted plot comparing rates across different outcome groups.
    
    Each subplot shows treated vs control time series for a different group/category.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: unit, time, group, outcome, denominator, treatment
        (or legacy column names).
    groups : list of str, optional
        List of groups to plot. If None, uses all unique groups in the data.
    rate_multiplier : float, default=1000
        Multiplier for rate calculation.
    treatment_dates : dict, optional
        Dictionary of {label: date} for vertical marker lines.
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
    ...     df, 
    ...     groups=['usborn', 'foreign'],
    ...     treatment_dates={'Dobbs': '2022-06-24'}
    ... )
    >>> fig.savefig('group_comparison.png')
    """
    _setup_plot_style()
    
    # Standardize column names
    df = _standardize_columns(df)
    
    # Ensure time is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Get groups to plot
    if groups is None:
        groups = df['group'].unique().tolist()
    
    n_groups = len(groups)
    if n_groups == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No groups to plot', ha='center', va='center', transform=ax.transAxes)
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
        'Treated': '#E41A1C',
        'Control': '#999999',
    }
    
    for idx, group in enumerate(groups):
        ax = axes_flat[idx]
        
        # Filter to this group
        group_df = df[df['group'] == group].copy()
        
        if len(group_df) == 0:
            ax.text(0.5, 0.5, f'No data for {group}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(group, fontsize=11, fontweight='bold')
            continue
        
        # Assign treatment group
        group_df['treatment_group'] = group_df['unit'].apply(
            lambda u: 'Treated' if u in treated_units else 'Control'
        )
        
        # Aggregate by treatment_group and time
        agg_df = group_df.groupby(['treatment_group', 'time']).agg(
            outcome=('outcome', 'sum'),
            denominator=('denominator', 'sum')
        ).reset_index()
        
        # Compute rate
        agg_df['rate'] = (agg_df['outcome'] / agg_df['denominator']) * rate_multiplier
        agg_df = agg_df.sort_values('time')
        
        # Plot each treatment group
        for tgroup in ['Control', 'Treated']:
            tg_data = agg_df[agg_df['treatment_group'] == tgroup]
            if len(tg_data) == 0:
                continue
            
            color = colors.get(tgroup, '#333333')
            linewidth = 2.0 if tgroup == 'Treated' else 1.5
            alpha = 1.0 if tgroup == 'Treated' else 0.7
            
            ax.plot(
                tg_data['time'],
                tg_data['rate'],
                label=tgroup,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
        
        # Add treatment date markers
        if treatment_dates is not None:
            for label, date in treatment_dates.items():
                date = pd.to_datetime(date)
                ax.axvline(x=date, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
        
        ax.set_title(group, fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Rate per {rate_multiplier:,.0f}', fontsize=9)
        
        # Only add legend to first subplot
        if idx == 0:
            ax.legend(loc='best', frameon=True, fontsize=8)
    
    # Hide unused axes
    for idx in range(len(groups), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Set common x-label
    fig.supxlabel('Time', fontsize=11)
    fig.suptitle('Rate Comparison by Group', fontsize=13, fontweight='bold', y=1.02)
    
    # Rotate x-axis labels
    for ax in axes_flat[:len(groups)]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig, axes


def make_state_fit_plot(
    draws_df: pd.DataFrame,
    unit: str,
    group: Optional[str] = None,
    ci_level: float = 0.95,
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a plot showing observed vs predicted for a specific unit.
    
    Shows the observed outcome as points, the predicted mean as a line,
    and a credible interval as a ribbon.
    
    Parameters
    ----------
    draws_df : pd.DataFrame
        DataFrame with posterior draws. Must contain columns:
        - .draw: draw identifier
        - unit (or state): panel entity
        - time: time period
        - outcome: observed outcome value
        - ypred: posterior predictive value
        - treatment: binary treatment indicator
        Optionally:
        - group (or category): outcome category
        - mu: counterfactual log-rate (for showing counterfactual)
    unit : str
        Unit (e.g., state name) to plot.
    group : str, optional
        Group/category to filter to. If None and multiple groups exist,
        uses the first group.
    ci_level : float, default=0.95
        Credible interval level (e.g., 0.95 for 95% CI).
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
    >>> fig, ax = make_state_fit_plot(draws_df, unit='Texas', group='total')
    >>> fig.savefig('texas_fit.png')
    
    >>> fig, ax = make_state_fit_plot(draws_df, unit='Oklahoma', ci_level=0.90)
    """
    _setup_plot_style()
    
    # Standardize column names
    df = _standardize_columns(draws_df)
    
    # Filter to specific unit
    df = df[df['unit'] == unit].copy()
    
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'No data for unit: {unit}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Observed vs Predicted: {unit}')
        return fig, ax
    
    # Filter to specific group if provided
    if group is not None and 'group' in df.columns:
        df = df[df['group'] == group].copy()
    elif 'group' in df.columns:
        # Use first group if not specified
        available_groups = df['group'].unique()
        if len(available_groups) > 1:
            warnings.warn(f"Multiple groups found for {unit}. Using first group: {available_groups[0]}. "
                          f"Specify group parameter to select a different one.")
        group = available_groups[0]
        df = df[df['group'] == group].copy()
    
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'No data for unit: {unit}, group: {group}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Observed vs Predicted: {unit}')
        return fig, ax
    
    # Ensure time is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Compute quantiles of ypred across draws
    alpha = 1 - ci_level
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    
    summary_df = df.groupby('time').agg(
        outcome=('outcome', 'first'),  # Observed is same across draws
        treatment=('treatment', 'first'),
        ypred_mean=('ypred', 'mean'),
        ypred_lower=('ypred', lambda x: np.quantile(x, lower_q)),
        ypred_upper=('ypred', lambda x: np.quantile(x, upper_q)),
    ).reset_index()
    
    # Check if mu is available for counterfactual
    has_mu = 'mu' in df.columns
    if has_mu:
        mu_summary = df.groupby('time').agg(
            mu_mean=('mu', lambda x: np.mean(np.exp(x))),
            mu_lower=('mu', lambda x: np.quantile(np.exp(x), lower_q)),
            mu_upper=('mu', lambda x: np.quantile(np.exp(x), upper_q)),
        ).reset_index()
        summary_df = summary_df.merge(mu_summary, on='time', how='left')
    
    summary_df = summary_df.sort_values('time')
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot credible interval as ribbon
    ax.fill_between(
        summary_df['time'],
        summary_df['ypred_lower'],
        summary_df['ypred_upper'],
        alpha=0.3,
        color='steelblue',
        label=f'{int(ci_level*100)}% CI',
    )
    
    # Plot predicted mean line
    ax.plot(
        summary_df['time'],
        summary_df['ypred_mean'],
        color='steelblue',
        linewidth=2,
        label='Predicted Mean',
    )
    
    # Plot observed as points
    ax.scatter(
        summary_df['time'],
        summary_df['outcome'],
        color='black',
        s=40,
        zorder=5,
        label='Observed',
    )
    
    # If counterfactual (mu) is available and there's treatment, show it
    if has_mu and summary_df['treatment'].sum() > 0:
        # Show counterfactual only in treatment period
        treated_mask = summary_df['treatment'] == 1
        if treated_mask.any():
            ax.plot(
                summary_df.loc[treated_mask, 'time'],
                summary_df.loc[treated_mask, 'mu_mean'],
                color='#E41A1C',
                linewidth=2,
                linestyle='--',
                label='Counterfactual',
            )
            ax.fill_between(
                summary_df.loc[treated_mask, 'time'],
                summary_df.loc[treated_mask, 'mu_lower'],
                summary_df.loc[treated_mask, 'mu_upper'],
                alpha=0.2,
                color='#E41A1C',
            )
    
    # Add vertical line at first treatment date
    if summary_df['treatment'].sum() > 0:
        first_treatment_time = summary_df.loc[summary_df['treatment'] == 1, 'time'].min()
        ax.axvline(
            x=first_treatment_time,
            color='black',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
        )
        ax.text(
            first_treatment_time, ax.get_ylim()[1], ' Treatment Start',
            ha='left', va='top', fontsize=9, color='black', alpha=0.8,
        )
    
    # Styling
    group_label = f' ({group})' if group else ''
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Outcome', fontsize=11)
    ax.set_title(f'Observed vs Predicted: {unit}{group_label}', fontsize=13, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig, ax


# =============================================================================
# Convenience Functions
# =============================================================================

def make_all_ppc_plots(
    draws_df: pd.DataFrame,
    output_dir: Optional[str] = None,
    outcome_col: str = 'outcome',
    categories: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    acf_lag: int = 6,
    max_treat_date: Optional[str] = None,
    ndraws: int = 1000,
) -> dict:
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
        Categories to include.
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
    
    print("Generating PPC plots...")
    
    # Maximum absolute residual
    print("  - Maximum absolute residual plot...")
    fig_abs, pvals_abs = make_abs_ppc_plot(
        draws_df, outcome_col=outcome_col, categories=categories, figsize=figsize
    )
    results['abs'] = {'fig': fig_abs, 'pvals': pvals_abs}
    
    # ACF
    print(f"  - ACF plot (lag={acf_lag})...")
    fig_acf, pvals_acf = make_acf_ppc_plot(
        draws_df, lag=acf_lag, outcome_col=outcome_col, categories=categories, figsize=figsize
    )
    results['acf'] = {'fig': fig_acf, 'pvals': pvals_acf}
    
    # RMSE
    print("  - RMSE plot...")
    fig_rmse, pvals_rmse = make_rmse_ppc_plot(
        draws_df, outcome_col=outcome_col, categories=categories, figsize=figsize
    )
    results['rmse'] = {'fig': fig_rmse, 'pvals': pvals_rmse}
    
    # Unit correlation
    print("  - Unit correlation plot...")
    fig_corr, pvals_corr = make_unit_corr_ppc_plot(
        draws_df, max_treat_date=max_treat_date, outcome_col=outcome_col,
        categories=categories, ndraws=ndraws, figsize=(10, 6)
    )
    results['unit_corr'] = {'fig': fig_corr, 'pvals': pvals_corr}
    
    # Save plots if output_dir provided
    if output_dir is not None:
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving plots to {output_dir}...")
        fig_abs.savefig(output_path / 'ppc_abs_residual.png', dpi=150, bbox_inches='tight')
        fig_acf.savefig(output_path / 'ppc_acf.png', dpi=150, bbox_inches='tight')
        fig_rmse.savefig(output_path / 'ppc_rmse.png', dpi=150, bbox_inches='tight')
        fig_corr.savefig(output_path / 'ppc_unit_corr.png', dpi=150, bbox_inches='tight')
        
        # Save p-values to CSV
        all_pvals = []
        for name, data in results.items():
            pvals = data['pvals'].copy()
            pvals['check_type'] = name
            all_pvals.append(pvals)
        
        if all_pvals:
            combined_pvals = pd.concat(all_pvals, ignore_index=True)
            combined_pvals.to_csv(output_path / 'ppc_pvalues.csv', index=False)
            print(f"  Saved p-values to {output_path / 'ppc_pvalues.csv'}")
    
    print("Done.")
    return results
