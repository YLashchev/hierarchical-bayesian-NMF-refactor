"""
Output formatting utilities for bayesian_panel_nmf.

This module provides a single function to format MCMC output into a tidy DataFrame
with FIXED standardized column names. No column name parameters - all names are fixed.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def format_draws(
    samples: Dict[str, np.ndarray],
    predictions: np.ndarray,
    data_dict: Dict
) -> pd.DataFrame:
    """
    Merge MCMC draws with observed data into tidy DataFrame.
    
    Uses FIXED standardized column names throughout - no column name parameters.
    
    Parameters
    ----------
    samples : dict
        MCMC samples from mcmc.get_samples(group_by_chain=True).
        Must contain 'mu_ctrl', optionally 'te'.
        Shape: (num_chains, num_samples, K, D, N)
    predictions : np.ndarray
        Posterior predictive samples, shape (C, S, K, D, N)
    data_dict : dict
        Output from load_and_prepare() with keys:
        - groups: list of str (K labels)
        - units: list of str (D labels)
        - times: list of datetime (N labels)
        - df_preprocessed: DataFrame with standardized columns
          (unit, time, group, outcome, denominator, treatment)
    
    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with FIXED columns:
        .draw, .chain, .iteration, unit, time, group,
        outcome, denominator, treatment, ypred, mu, mu_treated
    """
    # Extract dimensions from predictions: (chains, samples, groups, units, times)
    C, S, K, D, N = predictions.shape
    
    # Extract posterior arrays
    mu_ctrl = samples['mu_ctrl']  # (C, S, K, D, N)
    has_te = 'te' in samples
    
    total_rows = C * S * K * D * N
    print(f"  Building draws dataframe: {C} chains x {S} samples x {K} groups x {D} units x {N} times")
    print(f"  Total rows: {total_rows:,}")
    
    # Create index arrays using meshgrid (vectorized, fast)
    c_idx, s_idx, k_idx, d_idx, n_idx = np.meshgrid(
        np.arange(C), np.arange(S), np.arange(K), np.arange(D), np.arange(N),
        indexing='ij'
    )
    
    # Flatten all index arrays
    c_flat = c_idx.ravel()
    s_flat = s_idx.ravel()
    k_flat = k_idx.ravel()
    d_flat = d_idx.ravel()
    n_flat = n_idx.ravel()
    
    # Compute draw indices (1-indexed)
    draw_num = c_flat * S + s_flat + 1
    chain_num = c_flat + 1
    iter_num = s_flat + 1
    
    # Flatten predictions and mu_ctrl
    ypred_flat = predictions.ravel()
    mu_flat = mu_ctrl.ravel()
    
    # Compute mu_treated (mu + treatment effect if available)
    if has_te:
        te_flat = samples['te'].ravel()
        mu_treated_flat = mu_flat + te_flat
    else:
        mu_treated_flat = mu_flat.copy()
    
    # Map indices to labels
    groups = data_dict['groups']
    units = data_dict['units']
    times = data_dict['times']
    
    groups_arr = np.array(groups)
    units_arr = np.array(units)
    times_arr = np.array(times)
    
    group_flat = groups_arr[k_flat]
    unit_flat = units_arr[d_flat]
    time_flat = times_arr[n_flat]
    
    # Build draws dataframe with FIXED column names
    draws_df = pd.DataFrame({
        '.draw': draw_num,
        '.chain': chain_num,
        '.iteration': iter_num,
        'K': k_flat,
        'D': d_flat,
        'N': n_flat,
        'group': group_flat,
        'unit': unit_flat,
        'time': time_flat,
        'ypred': ypred_flat,
        'mu': mu_flat,
        'mu_treated': mu_treated_flat,
    })
    
    print(f"  Draws dataframe built: {len(draws_df):,} rows")
    
    # Prepare observation data for merging
    df_preprocessed = data_dict['df_preprocessed']
    obs_df = df_preprocessed.copy()
    
    # Create index mappings for merge
    group_to_idx = {g: i for i, g in enumerate(groups)}
    unit_to_idx = {u: i for i, u in enumerate(units)}
    time_to_idx = {t: i for i, t in enumerate(times)}
    
    obs_df['K'] = obs_df['group'].map(group_to_idx)
    obs_df['D'] = obs_df['unit'].map(unit_to_idx)
    obs_df['N'] = obs_df['time'].map(time_to_idx)
    
    # Select columns to merge (FIXED names)
    obs_cols = ['outcome']
    if 'denominator' in obs_df.columns:
        obs_cols.append('denominator')
    if 'treatment' in obs_df.columns:
        obs_cols.append('treatment')
    
    obs_subset = obs_df[['K', 'D', 'N'] + obs_cols].drop_duplicates()
    
    # Merge draws with observed data
    merged = draws_df.merge(obs_subset, on=['K', 'D', 'N'], how='left')
    
    # Drop internal index columns
    merged = merged.drop(columns=['K', 'D', 'N'])
    
    # Reorder columns to FIXED standard order
    col_order = [
        '.draw', '.chain', '.iteration',
        'unit', 'time', 'group',
        'outcome', 'denominator', 'treatment',
        'ypred', 'mu', 'mu_treated'
    ]
    col_order = [c for c in col_order if c in merged.columns]
    merged = merged[col_order]
    
    return merged
