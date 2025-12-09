"""
Posterior processing utilities for bayesian_panel_nmf.

Simplified output: clean long format draws + optional summary statistics.
Aggregations (e.g., "Ban States") are handled in R for consistency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def merge_draws_and_data(
    samples: Dict[str, np.ndarray],
    predictions: np.ndarray,
    df_preprocessed: pd.DataFrame,
    groups: List[str],
    units: List,
    times: List,
    unit_col: str = "unit",
    time_col: str = "time",
    group_col: str = "group",
    outcome_col: str = "outcome",
    denominator_col: str = "denominator",
    treatment_col: str = "treatment"
) -> pd.DataFrame:
    """
    Merge posterior draws with observed data in tidy format.
    
    This is a simplified version that outputs raw observation-level draws.
    Aggregations (Ban States, totals, etc.) are handled downstream in R.
    
    Parameters
    ----------
    samples : dict
        Posterior samples with keys 'mu_ctrl', optionally 'te'
        Shape: (num_chains, num_samples, K, D, N)
    predictions : np.ndarray
        Posterior predictive draws, shape (num_chains, num_samples, K, D, N)
    df_preprocessed : pd.DataFrame
        Preprocessed long format data
    groups : list
        Group labels (K dimension order)
    units : list
        Unit labels (D dimension order)
    times : list
        Time values (N dimension order)
    
    Returns
    -------
    pd.DataFrame
        Tidy draws format with columns:
        .draw, .chain, .iteration, category, unit, time, outcome, denominator,
        treatment, ypred, mu, mu_treated
    """
    # Get dimensions
    C, S, K, D, N = predictions.shape  # chains, samples, groups, units, times
    
    # Extract posterior arrays
    mu_ctrl = samples['mu_ctrl']  # (C, S, K, D, N)
    has_te = 'te' in samples
    if has_te:
        te = samples['te']
    
    print(f"  Building draws dataframe: {C} chains x {S} samples x {K} groups x {D} units x {N} times")
    total_rows = C * S * K * D * N
    print(f"  Total rows: {total_rows:,}")
    
    # OPTIMIZED: Use numpy broadcasting instead of nested loops
    # Create index arrays using meshgrid
    c_idx, s_idx, k_idx, d_idx, n_idx = np.meshgrid(
        np.arange(C), np.arange(S), np.arange(K), np.arange(D), np.arange(N),
        indexing='ij'
    )
    
    # Flatten all arrays
    c_flat = c_idx.ravel()
    s_flat = s_idx.ravel()
    k_flat = k_idx.ravel()
    d_flat = d_idx.ravel()
    n_flat = n_idx.ravel()
    
    # Compute draw indices (1-indexed for R compatibility)
    draw_num = c_flat * S + s_flat + 1
    chain_num = c_flat + 1
    iter_num = s_flat + 1
    
    # Flatten predictions and mu_ctrl
    ypred_flat = predictions.ravel()
    mu_flat = mu_ctrl.ravel()
    
    # Compute mu_treated
    if has_te:
        te_flat = te.ravel()
        mu_treated_flat = mu_flat + te_flat
    else:
        mu_treated_flat = mu_flat.copy()
    
    # Map indices to labels
    groups_arr = np.array(groups)
    units_arr = np.array(units)
    times_arr = np.array(times)
    
    category_flat = groups_arr[k_flat]
    unit_flat = units_arr[d_flat]
    time_flat = times_arr[n_flat]
    
    # Build dataframe directly from arrays (much faster than row-by-row)
    draws_df = pd.DataFrame({
        '.draw': draw_num,
        '.chain': chain_num,
        '.iteration': iter_num,
        'K': k_flat,
        'D': d_flat,
        'N': n_flat,
        'category': category_flat,
        unit_col: unit_flat,
        time_col: time_flat,
        'ypred': ypred_flat,
        'mu': mu_flat,
        'mu_treated': mu_treated_flat,
    })
    
    print(f"  Draws dataframe built: {len(draws_df):,} rows")
    
    # Merge with observed data
    # Prepare observation data (one row per K, D, N combination)
    obs_df = df_preprocessed.copy()
    
    # Create index mappings
    group_to_idx = {g: i for i, g in enumerate(groups)}
    unit_to_idx = {u: i for i, u in enumerate(units)}
    time_to_idx = {t: i for i, t in enumerate(times)}
    
    obs_df['K'] = obs_df[group_col].map(group_to_idx)
    obs_df['D'] = obs_df[unit_col].map(unit_to_idx)
    obs_df['N'] = obs_df[time_col].map(time_to_idx)
    
    # Select columns to merge
    merge_cols = ['K', 'D', 'N']
    obs_cols = [outcome_col]
    if denominator_col in obs_df.columns:
        obs_cols.append(denominator_col)
    if treatment_col in obs_df.columns:
        obs_cols.append(treatment_col)
    
    # Add any additional columns present
    additional = ['banned_state', 'start_date', 'end_date', 'exposure_code']
    for col in additional:
        if col in obs_df.columns:
            obs_cols.append(col)
    
    obs_subset = obs_df[merge_cols + obs_cols].drop_duplicates()
    
    # Merge
    merged = draws_df.merge(obs_subset, on=['K', 'D', 'N'], how='left')
    
    # Rename for R compatibility
    merged = merged.rename(columns={
        outcome_col: 'outcome',
        denominator_col: 'population' if denominator_col else None,
        treatment_col: 'exposure_code' if treatment_col else None,
        unit_col: 'state'  # Standard R column name
    })
    
    # Reorder columns
    col_order = ['.draw', '.chain', '.iteration', 'K', 'D', 'N',
                 'category', 'state', time_col, 'outcome', 'population',
                 'exposure_code', 'ypred', 'mu', 'mu_treated']
    col_order = [c for c in col_order if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in col_order]
    merged = merged[col_order + other_cols]
    
    return merged


def compute_summary_statistics(
    draws_df: pd.DataFrame,
    quantiles: Tuple[float, float] = (0.025, 0.975)
) -> pd.DataFrame:
    """
    Compute summary statistics from draws.
    
    Parameters
    ----------
    draws_df : pd.DataFrame
        Tidy draws format from merge_draws_and_data
    quantiles : tuple
        Lower and upper quantile bounds (default: 95% CI)
        
    Returns
    -------
    pd.DataFrame
        Summary with one row per observation, columns:
        category, state, time, outcome, population, exposure_code,
        ypred_mean, ypred_lower, ypred_upper, mu_mean, mu_treated_mean, te_mean, te_lower, te_upper
    """
    lower_q, upper_q = quantiles
    
    # Group by observation identifiers
    group_cols = ['K', 'D', 'N', 'category', 'state']
    
    # Add time if present
    time_cols = [c for c in draws_df.columns if c in ['time', 'start_date', 'end_date']]
    group_cols.extend(time_cols)
    
    # Add other observation-level columns
    obs_cols = ['outcome', 'population', 'exposure_code', 'banned_state']
    obs_cols = [c for c in obs_cols if c in draws_df.columns]
    
    # Compute summaries
    summary = draws_df.groupby(group_cols).agg(
        ypred_mean=('ypred', 'mean'),
        ypred_lower=('ypred', lambda x: x.quantile(lower_q)),
        ypred_upper=('ypred', lambda x: x.quantile(upper_q)),
        mu_mean=('mu', 'mean'),
        mu_treated_mean=('mu_treated', 'mean'),
    ).reset_index()
    
    # Add treatment effect if mu != mu_treated
    if 'mu' in draws_df.columns and 'mu_treated' in draws_df.columns:
        # include_groups=False since we only use mu/mu_treated columns, not grouping columns
        te_stats = draws_df.groupby(group_cols).apply(
            lambda g: pd.Series({
                'te_mean': (g['mu_treated'] - g['mu']).mean(),
                'te_lower': (g['mu_treated'] - g['mu']).quantile(lower_q),
                'te_upper': (g['mu_treated'] - g['mu']).quantile(upper_q),
            }),
            include_groups=False
        ).reset_index()
        summary = summary.merge(te_stats, on=group_cols)
    
    # Add observation-level data
    obs_data = draws_df.groupby(group_cols)[obs_cols].first().reset_index()
    summary = summary.merge(obs_data, on=group_cols, how='left')
    
    return summary


def save_results(
    draws_df: pd.DataFrame,
    output_path: str,
    save_draws: bool = True,
    save_summary: bool = True,
    summary_quantiles: Tuple[float, float] = (0.025, 0.975)
) -> Dict[str, str]:
    """
    Save results to CSV files.
    
    Parameters
    ----------
    draws_df : pd.DataFrame
        Tidy draws from merge_draws_and_data
    output_path : str
        Base path for output files (without extension)
    save_draws : bool
        Whether to save full draws file
    save_summary : bool
        Whether to save summary statistics file
    summary_quantiles : tuple
        Quantiles for summary CI
        
    Returns
    -------
    dict
        Paths to saved files
    """
    output_path = Path(output_path)
    saved_files = {}
    
    if save_draws:
        draws_path = output_path.with_suffix('.csv')
        draws_df.to_csv(draws_path, index=False)
        saved_files['draws'] = str(draws_path)
        print(f"Saved draws to: {draws_path}")
    
    if save_summary:
        summary_df = compute_summary_statistics(draws_df, summary_quantiles)
        summary_path = output_path.parent / (output_path.stem + '_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        saved_files['summary'] = str(summary_path)
        print(f"Saved summary to: {summary_path}")
    
    return saved_files


def merge_and_save(
    samples: Dict[str, np.ndarray],
    predictions: np.ndarray,
    data_dict: Dict,
    df_preprocessed: pd.DataFrame,
    output_path: str,
    config: Dict,
    unit_col: str = "unit",
    time_col: str = "time",
    group_col: str = "group",
    outcome_col: str = "outcome",
    denominator_col: str = "denominator",
    treatment_col: str = "treatment"
) -> Dict[str, str]:
    """
    Convenience function: merge draws with data and save.
    
    Parameters
    ----------
    samples : dict
        MCMC samples
    predictions : np.ndarray
        Posterior predictive samples
    data_dict : dict
        Model data dictionary from preprocessing
    df_preprocessed : pd.DataFrame
        Preprocessed data
    output_path : str
        Base output path
    config : dict
        Configuration with output settings
        
    Returns
    -------
    dict
        Paths to saved files
    """
    # Merge draws
    draws_df = merge_draws_and_data(
        samples=samples,
        predictions=predictions,
        df_preprocessed=df_preprocessed,
        groups=data_dict['groups'],
        units=data_dict['units'],
        times=data_dict['times'],
        unit_col=unit_col,
        time_col=time_col,
        group_col=group_col,
        outcome_col=outcome_col,
        denominator_col=denominator_col,
        treatment_col=treatment_col
    )
    
    # Get output settings
    output_config = config.get('output', {})
    save_draws = output_config.get('save_draws', True)
    save_summary = output_config.get('save_summary', True)
    
    # Save
    return save_results(
        draws_df=draws_df,
        output_path=output_path,
        save_draws=save_draws,
        save_summary=save_summary
    )
