"""
Output formatting utilities for bayesian_panel_nmf.

This module provides functions to format MCMC output into a tidy DataFrame
with FIXED standardized column names. No column name parameters - all names are fixed.

Memory Optimization:
- Categorical dtypes for low-cardinality string columns (unit, group)
- Smaller integer dtypes (int8, int32) where possible
- Float32 for predictions (half the memory of float64)
"""

import numpy as np
import pandas as pd
from typing import Dict

from loguru import logger
from bayesian_panel_nmf.validation import (
    validate_samples,
    validate_predictions,
    DataError,
)


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage with appropriate dtypes."""
    # Log initial memory usage
    initial_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.debug(f"DataFrame memory before optimization: {initial_mem:.1f} MB")
    
    # Categorical for repeated strings
    for col in ['unit', 'group']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    
    # Smaller integers
    int_dtypes = {'.chain': 'int8', '.draw': 'int32', '.iteration': 'int32', 'treatment': 'int8'}
    for col, dtype in int_dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    
    # Float32 for predictions (sufficient precision, half memory)
    for col in ['ypred', 'mu', 'mu_treated', 'outcome', 'denominator']:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].astype('float32')
    
    # Log final memory usage and savings
    final_mem = df.memory_usage(deep=True).sum() / 1024**2
    savings_pct = (1 - final_mem / initial_mem) * 100 if initial_mem > 0 else 0
    logger.debug(f"DataFrame memory after optimization: {final_mem:.1f} MB ({savings_pct:.1f}% reduction)")
    
    return df


def format_draws(
    samples: Dict[str, np.ndarray],
    predictions: np.ndarray,
    data_dict: Dict
) -> pd.DataFrame:
    """
    Merge MCMC draws with observed data into tidy DataFrame.
    
    Uses FIXED standardized column names throughout - no column name parameters.
    Automatically optimizes memory usage via dtype downcasting.
    
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
        
        Memory-optimized with:
        - Categorical dtypes for unit, group
        - int8/int32 for integer columns
        - float32 for numeric columns
        
    Raises
    ------
    DataError
        If samples is missing 'mu_ctrl' key, predictions has wrong shape,
        data_dict is missing required keys, or dimensions don't match
        
    Notes
    -----
    For a typical run with 4 chains x 250 samples x 2 groups x 51 units x 48 times:
    - Before optimization: ~470 MB
    - After optimization: ~120 MB (75% reduction)
    """
    # =========================================================================
    # Input validation
    # =========================================================================
    logger.debug("Validating format_draws inputs...")
    validate_samples(samples)
    validate_predictions(predictions, samples)
    
    # Validate data_dict has required keys
    required_keys = ['groups', 'units', 'times', 'df_preprocessed']
    missing = [k for k in required_keys if k not in data_dict]
    if missing:
        raise DataError(f"data_dict missing keys: {missing}")
    logger.debug("Input validation passed")
    
    # Extract dimensions from predictions: (chains, samples, groups, units, times)
    C, S, K, D, N = predictions.shape
    
    # Extract posterior arrays
    mu_ctrl = samples['mu_ctrl']  # (C, S, K, D, N)
    has_te = 'te' in samples
    
    total_rows = C * S * K * D * N
    logger.info(f"Building draws dataframe: {C} chains x {S} samples x {K} groups x {D} units x {N} times")
    logger.info(f"Total rows: {total_rows:,}")
    
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
    
    # Flatten predictions and mu_ctrl - use float32 from the start
    ypred_flat = predictions.ravel().astype(np.float32)
    mu_flat = mu_ctrl.ravel().astype(np.float32)
    
    # Compute mu_treated (mu + treatment effect if available)
    if has_te:
        te_flat = samples['te'].ravel().astype(np.float32)
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
    # Use optimized dtypes from the start where possible
    draws_df = pd.DataFrame({
        '.draw': draw_num.astype(np.int32),
        '.chain': chain_num.astype(np.int8),
        '.iteration': iter_num.astype(np.int32),
        'K': k_flat.astype(np.int16),
        'D': d_flat.astype(np.int16),
        'N': n_flat.astype(np.int16),
        'group': pd.Categorical(group_flat, categories=groups),
        'unit': pd.Categorical(unit_flat, categories=units),
        'time': time_flat,
        'ypred': ypred_flat,
        'mu': mu_flat,
        'mu_treated': mu_treated_flat,
    })
    
    logger.debug(f"Draws dataframe built: {len(draws_df):,} rows")
    
    # Prepare observation data for merging
    df_preprocessed = data_dict['df_preprocessed']
    obs_df = df_preprocessed.copy()
    
    # Create index mappings for merge
    group_to_idx = {g: i for i, g in enumerate(groups)}
    unit_to_idx = {u: i for i, u in enumerate(units)}
    time_to_idx = {t: i for i, t in enumerate(times)}
    
    obs_df['K'] = obs_df['group'].map(group_to_idx).astype(np.int16)
    obs_df['D'] = obs_df['unit'].map(unit_to_idx).astype(np.int16)
    obs_df['N'] = obs_df['time'].map(time_to_idx).astype(np.int16)
    
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
    
    # Apply dtype optimization for remaining columns
    merged = _optimize_dtypes(merged)
    
    # Final memory usage summary
    final_mem_mb = merged.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Output DataFrame: {len(merged):,} rows, {final_mem_mb:.1f} MB")
    
    return merged
