"""
Data preprocessing utilities for nativity analysis.

This module handles transformation of nativity birth data from wide to long format,
aggregation to bimonthly periods, and preparation for Bayesian modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import timedelta


def wide_to_long(
    df: pd.DataFrame,
    sub_groups: List[str] = ['usborn', 'foreign']
) -> pd.DataFrame:
    """
    Convert wide format data to long format with subgroups stacked.
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide format data with separate columns for each subgroup
    sub_groups : list of str
        Subgroups to include (default: ['usborn', 'foreign'])
        For 'total', creates total columns by summing usborn + foreign
        
    Returns
    -------
    pd.DataFrame
        Long format data with columns: state, year, month, time, group, births, population
    """
    df = df.copy()
    
    # Handle "total" case: create total columns by summing usborn + foreign
    if sub_groups == ['total']:
        if 'births_usborn' in df.columns and 'births_foreign' in df.columns:
            df['births_total'] = df['births_usborn'] + df['births_foreign']
            df['pop_total'] = df['pop_usborn'] + df['pop_foreign']
        else:
            raise ValueError("Cannot create 'total' - missing usborn/foreign columns")
    
    # Prepare birth columns
    birth_cols = [f'births_{group}' for group in sub_groups]
    pop_cols = [f'pop_{group}' for group in sub_groups]
    
    # Check that required columns exist
    missing_births = set(birth_cols) - set(df.columns)
    missing_pops = set(pop_cols) - set(df.columns)
    if missing_births or missing_pops:
        raise ValueError(f"Missing columns - births: {missing_births}, population: {missing_pops}")
    
    # Melt births
    df_births = df[['state', 'year', 'month', 'time', 'banned_state', 'exposed'] + birth_cols].melt(
        id_vars=['state', 'year', 'month', 'time', 'banned_state', 'exposed'],
        value_vars=birth_cols,
        var_name='group',
        value_name='births'
    )
    
    # Melt population
    df_pop = df[['state', 'year', 'month', 'time'] + pop_cols].melt(
        id_vars=['state', 'year', 'month', 'time'],
        value_vars=pop_cols,
        var_name='group',
        value_name='population'
    )
    
    # Clean group names (remove 'births_' or 'pop_' prefix)
    df_births['group'] = df_births['group'].str.replace('births_', '')
    df_pop['group'] = df_pop['group'].str.replace('pop_', '')
    
    # Merge births and population
    df_long = df_births.merge(
        df_pop,
        on=['state', 'year', 'month', 'time', 'group'],
        how='left'
    )
    
    # Sort for consistency
    df_long = df_long.sort_values(['state', 'time', 'group']).reset_index(drop=True)
    
    return df_long


def aggregate_to_bimonthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly data to bimonthly periods.
    
    Bimonthly codes (bmcode):
    - 1: Jan-Feb
    - 2: Mar-Apr
    - 3: May-Jun
    - 4: Jul-Aug
    - 5: Sep-Oct
    - 6: Nov-Dec
    
    Parameters
    ----------
    df : pd.DataFrame
        Long format monthly data
        
    Returns
    -------
    pd.DataFrame
        Bimonthly aggregated data with bmcode column
    """
    # Create bimonthly code
    df = df.copy()
    df['bmcode'] = ((df['month'] - 1) // 2) + 1
    
    # Aggregate by state, year, bmcode, and group
    agg_dict = {
        'births': 'sum',
        'population': 'mean',  # Average population over the two months
        'banned_state': 'first',
        'exposed': 'max',  # If exposed in either month, mark as exposed
    }
    
    # Add time as the first month of the bimonthly period
    df['bimonthly_time'] = pd.to_datetime(
        df['year'].astype(str) + '-' + (df['bmcode'] * 2 - 1).astype(str) + '-01'
    )
    
    df_bm = df.groupby(['state', 'year', 'bmcode', 'group'], as_index=False).agg(agg_dict)
    
    # Add time column (first month of bimonthly period)
    df_bm['time'] = pd.to_datetime(
        df_bm['year'].astype(str) + '-' + (df_bm['bmcode'] * 2 - 1).astype(str) + '-01'
    )
    
    # Add start and end dates for the bimonthly period
    df_bm['start_date'] = df_bm['time']
    df_bm['end_date'] = df_bm['time'] + pd.DateOffset(months=2) - timedelta(days=1)
    
    return df_bm


def create_exposure_codes(
    df: pd.DataFrame,
    use_raw_exposed: bool = True
) -> pd.DataFrame:
    """
    Create exposure_code column using the raw 'exposed' column from source data.
    
    The raw 'exposed' column already has the correct state-specific exposure timing:
    - Texas: Exposure starts March 2022 (SB8 implementation)
    - Alabama, Arkansas, Kentucky, Louisiana, Mississippi, Missouri, Oklahoma, 
      South Dakota, Tennessee, West Virginia, Wisconsin: January 2023
    - Georgia: May 2023 (6-week heartbeat ban)
    - Idaho: March 2023
    - And other state-specific dates
    
    The source data (nativity_analyticdata.csv) already has these correctly coded
    in the 'exposed' column, so we should use it directly rather than applying
    blanket rules that don't account for state-specific variation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'exposed' and/or 'banned_state' column and 'time' column
    use_raw_exposed : bool, default=True
        If True, use the raw 'exposed' column from source data (RECOMMENDED)
        If False, fall back to a simplified rule (not recommended - loses state variation)
        
    Returns
    -------
    pd.DataFrame
        Data with 'exposure_code' column added
    """
    df = df.copy()
    
    if use_raw_exposed:
        # Use the raw exposed column which has correct state-specific timing
        if 'exposed' not in df.columns:
            raise ValueError("Expected 'exposed' column to be present in the data.")
        df['exposure_code'] = df['exposed'].astype(int)
        print("Using raw 'exposed' column from source data (has correct state-specific timing)")
        
    else:
        # Fallback: simplified rule (not recommended - loses state-specific variation)
        print("Warning: Using simplified exposure rules - this loses state-specific variation!")
        df['exposure_code'] = 0
        
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        # Texas exposure starts March 2022
        texas_mask = (df['state'] == 'Texas') & (df['time'] >= pd.to_datetime('2022-03-01'))
        df.loc[texas_mask, 'exposure_code'] = 1
        
        # Most other banned states start January 2023 (simplified - not accurate for all)
        if 'banned_state' in df.columns:
            banned_states = df[df['banned_state'] == 1]['state'].unique()
            other_banned = [s for s in banned_states if s != 'Texas']
            
            for state in other_banned:
                state_mask = (df['state'] == state) & (df['time'] >= pd.to_datetime('2023-01-01'))
                df.loc[state_mask, 'exposure_code'] = 1
    
    return df


def prepare_model_data(
    df: pd.DataFrame,
    sub_groups: List[str] = ['usborn', 'foreign'],
    outcome_type: str = 'births'
) -> Dict[str, np.ndarray]:
    """
    Prepare data in the format expected by the Bayesian panel model.
    
    This function reshapes long format data into multi-dimensional arrays
    organized by (categories, states, time).
    
    Parameters
    ----------
    df : pd.DataFrame
        Long format data with columns: state, time, group, births, population, exposure_code
    sub_groups : list of str
        List of subgroups (must match 'group' column values)
    outcome_type : str
        Type of outcome ('births' or other)
        
    Returns
    -------
    dict
        Dictionary with keys:
        - Y: outcome array (K x D x N)
        - denominators: population array (K x D x N) 
        - control_idx_array: boolean array for control periods (K x D x N)
        - missing_idx_array: boolean array for missing data (K x D x N)
        - variables: list of group names
        where K=number of groups, D=number of states, N=number of time periods
    """
    # Ensure data is sorted
    df = df.sort_values(['state', 'time', 'group']).reset_index(drop=True)
    
    # Get dimensions
    states = sorted(df['state'].unique())
    times = sorted(df['time'].unique())
    num_states = len(states)
    num_times = len(times)
    num_groups = len(sub_groups)
    
    # Initialize arrays
    Y = np.zeros((num_groups, num_states, num_times))
    denominators = np.zeros((num_groups, num_states, num_times))
    control_idx = np.ones((num_groups, num_states, num_times), dtype=bool)
    missing_idx = np.zeros((num_groups, num_states, num_times), dtype=bool)
    
    # Create state and time mappings
    state_to_idx = {s: i for i, s in enumerate(states)}
    time_to_idx = {t: i for i, t in enumerate(times)}
    group_to_idx = {g: i for i, g in enumerate(sub_groups)}
    
    # Fill arrays
    for _, row in df.iterrows():
        k = group_to_idx[row['group']]
        d = state_to_idx[row['state']]
        n = time_to_idx[row['time']]
        
        Y[k, d, n] = row['births'] if not pd.isna(row['births']) else 0
        denominators[k, d, n] = row['population'] / 1e4  # Per 10k population
        control_idx[k, d, n] = (row['exposure_code'] == 0)
        missing_idx[k, d, n] = pd.isna(row['births'])
    
    return {
        'Y': Y,
        'denominators': denominators,
        'control_idx_array': control_idx,
        'missing_idx_array': missing_idx,
        'variables': sub_groups,
        'states': states,
        'times': times,
    }


def filter_time_period(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter data to specified time period.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'time' column
    start_date : str, optional
        Start date (inclusive)
    end_date : str, optional
        End date (exclusive)
        
    Returns
    -------
    pd.DataFrame
        Filtered data
    """
    df_filtered = df.copy()
    
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df_filtered = df_filtered[df_filtered['time'] >= start_dt]
    
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        df_filtered = df_filtered[df_filtered['time'] < end_dt]
    
    return df_filtered
