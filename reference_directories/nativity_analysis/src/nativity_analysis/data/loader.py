"""
Data loading utilities for nativity analysis.

This module provides functions to load and validate the nativity birth data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union


def load_nativity_data(
    filepath: Union[str, Path],
    validate: bool = True
) -> pd.DataFrame:
    """
    Load nativity birth data from CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the nativity_analyticdata.csv file
    validate : bool, default=True
        Whether to validate the loaded data
        
    Returns
    -------
    pd.DataFrame
        Loaded data with datetime conversions applied
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    ValueError
        If data validation fails
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Standardize column names to match births_{group} and pop_{group} pattern
    # The raw data uses inconsistent naming for ethnicity-specific columns
    column_mapping = {
        # Hispanic births and population
        'birthshisp_usborn': 'births_hisp_usborn',
        'birthshisp_foreign': 'births_hisp_foreign',
        'pophisp_usborn': 'pop_hisp_usborn',
        'pophisp_foreign': 'pop_hisp_foreign',
        # Non-Hispanic births and population
        'birthsnh_usborn': 'births_nh_usborn',
        'birthsnh_foreign': 'births_nh_foreign',
        'popnh_usborn': 'pop_nh_usborn',
        'popnh_foreign': 'pop_nh_foreign',
    }
    df = df.rename(columns=column_mapping)
    
    # Normalize date-like columns to datetime with explicit format (M/D/YY)
    # Example values: 1/1/16 -> 2016-01-01
    date_fmt = "%m/%d/%y"
    df['time'] = pd.to_datetime(df['time'], format=date_fmt, errors='coerce')
    
    # Convert date columns
    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'], format=date_fmt, errors='coerce')
    if 'end_date' in df.columns:
        df['end_date'] = pd.to_datetime(df['end_date'], format=date_fmt, errors='coerce')
    
    # Validate if requested
    if validate:
        _validate_data(df)
    
    return df


def _validate_data(df: pd.DataFrame) -> None:
    """
    Validate the structure and contents of nativity data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
        
    Raises
    ------
    ValueError
        If validation fails
    """
    required_columns = [
        'state', 'year', 'month', 'time',
        'births_usborn', 'births_foreign',
        'pop_usborn', 'pop_foreign',
        'banned_state', 'exposed'
    ]
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for null values in critical columns
    critical_cols = ['state', 'year', 'month', 'time']
    null_counts = df[critical_cols].isna().sum()
    if null_counts.any():
        raise ValueError(f"Null values found in critical columns: {null_counts[null_counts > 0]}")
    
    # Check that births and population are non-negative
    numeric_cols = ['births_usborn', 'births_foreign', 'pop_usborn', 'pop_foreign']
    for col in numeric_cols:
        if (df[col] < 0).any():
            raise ValueError(f"Negative values found in {col}")
    
    # Check year range is reasonable
    if df['year'].min() < 2000 or df['year'].max() > 2030:
        raise ValueError(f"Year range outside expected bounds: {df['year'].min()}-{df['year'].max()}")
    
    print(f"Data validation passed:")
    print(f"  - {len(df)} rows")
    print(f"  - {df['state'].nunique()} states")
    print(f"  - Time range: {df['time'].min()} to {df['time'].max()}")
    print(f"  - {df['banned_state'].sum()} state-month observations from banned states")
