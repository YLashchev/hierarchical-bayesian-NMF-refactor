"""
Posterior Predictive Check (PPC) visualization utilities for bayesian_panel_nmf.

This module provides functions to generate PPC plots that compare observed vs predicted
statistics in the control period. These diagnostics help assess model fit.

The module handles BOTH standardized and legacy column names:
- Standardized: unit, group, denominator, treatment
- Legacy: state, category, population, exposure_code

All functions return (fig, pvals_df) tuples where fig is a matplotlib Figure
and pvals_df is a DataFrame with p-values for each facet.
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
