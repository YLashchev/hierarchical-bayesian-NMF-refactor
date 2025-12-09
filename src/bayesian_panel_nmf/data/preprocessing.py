"""
Data preprocessing utilities for bayesian_panel_nmf.

This module handles temporal aggregation and model data preparation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from datetime import timedelta


def aggregate_to_period(
    df: pd.DataFrame,
    period: str = "bimonthly",
    custom_func: Optional[Callable[[int], int]] = None,
    outcome_col: str = "outcome",
    denominator_col: Optional[str] = "denominator",
    unit_col: str = "unit",
    time_col: str = "time",
    group_col: str = "group",
    treatment_col: str = "treatment"
) -> pd.DataFrame:
    """
    Aggregate data to specified time periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long format data with time column as datetime
    period : str
        One of: 'monthly', 'bimonthly', 'quarterly', 'yearly', 'custom'
    custom_func : callable, optional
        For period='custom': function mapping month (1-12) to period code
    outcome_col : str
        Column name for outcome variable
    denominator_col : str, optional
        Column name for denominator (averaged within period)
    unit_col : str
        Column name for panel unit
    time_col : str
        Column name for time
    group_col : str
        Column name for outcome group
    treatment_col : str
        Column name for treatment indicator
        
    Returns
    -------
    pd.DataFrame
        Aggregated data with period indicator
    """
    df = df.copy()
    
    # Ensure time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract year and month
    df['_year'] = df[time_col].dt.year
    df['_month'] = df[time_col].dt.month
    
    # Calculate period code based on aggregation type
    if period == 'monthly':
        df['_period'] = df['_month']
        months_per_period = 1
    elif period == 'bimonthly':
        df['_period'] = ((df['_month'] - 1) // 2) + 1  # 1-6
        months_per_period = 2
    elif period == 'quarterly':
        df['_period'] = ((df['_month'] - 1) // 3) + 1  # 1-4
        months_per_period = 3
    elif period == 'yearly':
        df['_period'] = 1  # All months → same period
        months_per_period = 12
    elif period == 'custom':
        if custom_func is None:
            raise ValueError("custom_func required for period='custom'")
        df['_period'] = df['_month'].apply(custom_func)
        months_per_period = 12 // df['_period'].max()  # Approximate
    else:
        raise ValueError(f"Unknown period: {period}. Use: monthly, bimonthly, quarterly, yearly, custom")
    
    # Define aggregation functions
    agg_dict = {
        outcome_col: 'sum',
        treatment_col: 'max',  # If treated in any sub-period, treated
    }
    
    if denominator_col and denominator_col in df.columns:
        agg_dict[denominator_col] = 'mean'  # Average population over period
    
    # Add any other columns (take first value)
    other_cols = [c for c in df.columns if c not in 
                  [unit_col, time_col, group_col, treatment_col, outcome_col, 
                   denominator_col, '_year', '_month', '_period']]
    for col in other_cols:
        agg_dict[col] = 'first'
    
    # Aggregate
    group_by_cols = [unit_col, '_year', '_period']
    if group_col in df.columns:
        group_by_cols.append(group_col)
    
    df_agg = df.groupby(group_by_cols, as_index=False).agg(agg_dict)
    
    # Create representative time (first month of period)
    first_month = (df_agg['_period'] - 1) * months_per_period + 1
    df_agg[time_col] = pd.to_datetime(
        df_agg['_year'].astype(str) + '-' + first_month.astype(str) + '-01'
    )
    
    # Add period boundaries
    df_agg['start_date'] = df_agg[time_col]
    df_agg['end_date'] = df_agg[time_col] + pd.DateOffset(months=months_per_period) - timedelta(days=1)
    
    # Clean up temporary columns
    df_agg = df_agg.drop(columns=['_year', '_period'])
    
    # Sort
    sort_cols = [unit_col, time_col]
    if group_col in df_agg.columns:
        sort_cols.append(group_col)
    df_agg = df_agg.sort_values(sort_cols).reset_index(drop=True)
    
    return df_agg


def filter_time_period(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_col: str = "time"
) -> pd.DataFrame:
    """
    Filter data to specified time period.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with time column
    start_date : str, optional
        Start date (inclusive), e.g., "2016-01-01"
    end_date : str, optional
        End date (exclusive), e.g., "2024-01-01"
    time_col : str
        Column name for time
        
    Returns
    -------
    pd.DataFrame
        Filtered data
    """
    df_filtered = df.copy()
    
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df_filtered = df_filtered[df_filtered[time_col] >= start_dt]
    
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        df_filtered = df_filtered[df_filtered[time_col] < end_dt]
    
    return df_filtered.reset_index(drop=True)


def prepare_model_data(
    df: pd.DataFrame,
    groups: List[str],
    outcome_col: str = "outcome",
    denominator_col: Optional[str] = "denominator",
    unit_col: str = "unit",
    time_col: str = "time",
    group_col: str = "group",
    treatment_col: str = "treatment",
    denominator_scale: float = 1e4
) -> Dict[str, np.ndarray]:
    """
    Prepare data in the format expected by the Bayesian panel model.
    
    Reshapes long format data into multi-dimensional arrays
    organized by (categories, units, time).
    
    Parameters
    ----------
    df : pd.DataFrame
        Long format data
    groups : list of str
        Group labels to include (order determines K dimension)
    outcome_col : str
        Outcome column name
    denominator_col : str, optional
        Denominator column name (None if no denominator)
    unit_col : str
        Unit column name
    time_col : str
        Time column name
    group_col : str
        Group column name
    treatment_col : str
        Treatment column name
    denominator_scale : float
        Scale factor for denominators (default 1e4 = per 10k)
        
    Returns
    -------
    dict
        Dictionary with keys:
        - Y: outcome array (K x D x N)
        - denominators: scaled denominator array (K x D x N)
        - control_idx_array: boolean array for control periods (K x D x N)
        - missing_idx_array: boolean array for missing data (K x D x N)
        - groups: list of group names (K dimension)
        - units: list of unit names (D dimension)
        - times: list of time values (N dimension)
    """
    # Filter to requested groups
    df = df[df[group_col].isin(groups)].copy()
    
    # Ensure sorted
    df = df.sort_values([unit_col, time_col, group_col]).reset_index(drop=True)
    
    # Get dimensions
    units = sorted(df[unit_col].unique())
    times = sorted(df[time_col].unique())
    num_groups = len(groups)
    num_units = len(units)
    num_times = len(times)
    
    # Initialize arrays
    Y = np.zeros((num_groups, num_units, num_times))
    denominators = np.ones((num_groups, num_units, num_times))  # Default to 1 if no denominator
    control_idx = np.ones((num_groups, num_units, num_times), dtype=bool)
    missing_idx = np.zeros((num_groups, num_units, num_times), dtype=bool)
    
    # Create mappings
    group_to_idx = {g: i for i, g in enumerate(groups)}
    unit_to_idx = {u: i for i, u in enumerate(units)}
    time_to_idx = {t: i for i, t in enumerate(times)}
    
    # Fill arrays
    for _, row in df.iterrows():
        k = group_to_idx[row[group_col]]
        d = unit_to_idx[row[unit_col]]
        n = time_to_idx[row[time_col]]
        
        # Outcome
        outcome_val = row[outcome_col]
        if pd.isna(outcome_val):
            Y[k, d, n] = 0
            missing_idx[k, d, n] = True
        else:
            Y[k, d, n] = outcome_val
        
        # Denominator
        if denominator_col and denominator_col in df.columns:
            denom_val = row[denominator_col]
            if pd.notna(denom_val) and denom_val > 0:
                denominators[k, d, n] = denom_val / denominator_scale
        
        # Control status (treatment=0 means control)
        control_idx[k, d, n] = (row[treatment_col] == 0)
    
    return {
        'Y': Y,
        'denominators': denominators,
        'control_idx_array': control_idx,
        'missing_idx_array': missing_idx,
        'groups': groups,
        'units': units,
        'times': times,
    }


def preprocess_pipeline(
    df: pd.DataFrame,
    groups: List[str],
    config: Dict,
    outcome_col: str = "outcome",
    denominator_col: Optional[str] = "denominator",
    unit_col: str = "unit",
    time_col: str = "time",
    group_col: str = "group",
    treatment_col: str = "treatment"
) -> Dict[str, np.ndarray]:
    """
    Full preprocessing pipeline: filter, aggregate, prepare.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long format data
    groups : list of str
        Groups to include
    config : dict
        Configuration with 'data' section
        
    Returns
    -------
    dict
        Model-ready data dictionary
    """
    data_config = config.get('data', {})
    
    # Filter time period
    df = filter_time_period(
        df,
        start_date=data_config.get('start_date'),
        end_date=data_config.get('end_date'),
        time_col=time_col
    )
    
    # Aggregate if enabled
    agg_config = data_config.get('aggregation', {})
    if agg_config.get('enabled', False):
        df = aggregate_to_period(
            df,
            period=agg_config.get('period', 'bimonthly'),
            outcome_col=outcome_col,
            denominator_col=denominator_col,
            unit_col=unit_col,
            time_col=time_col,
            group_col=group_col,
            treatment_col=treatment_col
        )
    
    # Prepare model data
    data_dict = prepare_model_data(
        df,
        groups=groups,
        outcome_col=outcome_col,
        denominator_col=denominator_col,
        unit_col=unit_col,
        time_col=time_col,
        group_col=group_col,
        treatment_col=treatment_col
    )
    
    # Store preprocessed dataframe for output merging later
    data_dict['df_preprocessed'] = df
    
    return data_dict
