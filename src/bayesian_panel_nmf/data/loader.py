"""
Generic data loading utilities for bayesian_panel_nmf.

This module provides schema-based loading for any panel data format.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

from .schema import DataSchema


def load_panel_data(
    filepath: Union[str, Path],
    schema: Optional[DataSchema] = None,
    config: Optional[Dict[str, Any]] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load panel data from CSV file using schema-based column mapping.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV data file
    schema : DataSchema, optional
        Schema defining column mappings. If not provided, uses config.
    config : dict, optional
        Configuration dict to create schema from. Ignored if schema is provided.
    validate : bool, default=True
        Whether to validate the loaded data against schema
        
    Returns
    -------
    pd.DataFrame
        Loaded data with datetime conversions applied
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    ValueError
        If neither schema nor config is provided, or validation fails
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Create schema if needed
    if schema is None:
        if config is None:
            raise ValueError("Either schema or config must be provided")
        schema = DataSchema.from_config(config)
    
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Parse time column
    df = _parse_time_column(df, schema)
    
    # Validate if requested
    if validate:
        schema.validate(df)
        _print_data_summary(df, schema)
    
    return df


def _parse_time_column(df: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    """
    Parse time column to datetime.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with time column
    schema : DataSchema
        Schema specifying time column and date format
        
    Returns
    -------
    pd.DataFrame
        Data with parsed datetime column
    """
    df = df.copy()
    time_col = schema.time_col
    
    if time_col not in df.columns:
        return df
    
    # Try to parse based on format specification
    if schema.date_format == "auto":
        # Try common formats in order
        formats_to_try = [
            None,           # pandas auto-detection
            "%Y-%m-%d",     # ISO format
            "%m/%d/%y",     # US short year
            "%m/%d/%Y",     # US long year
            "%d-%m-%Y",     # European
            "%d/%m/%Y",     # European with slash
        ]
        
        for fmt in formats_to_try:
            try:
                if fmt is None:
                    df[time_col] = pd.to_datetime(df[time_col])
                else:
                    df[time_col] = pd.to_datetime(df[time_col], format=fmt)
                break
            except (ValueError, TypeError):
                continue
        else:
            print(f"Warning: Could not parse time column '{time_col}'. "
                  f"Leaving as-is. Sample values: {df[time_col].head(3).tolist()}")
    else:
        try:
            df[time_col] = pd.to_datetime(df[time_col], format=schema.date_format)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse time column '{time_col}' "
                  f"with format '{schema.date_format}': {e}")
    
    return df


def _print_data_summary(df: pd.DataFrame, schema: DataSchema) -> None:
    """Print summary of loaded data."""
    print(f"Data loaded successfully:")
    print(f"  - {len(df):,} rows")
    print(f"  - {df[schema.unit_col].nunique()} unique {schema.unit_col}s")
    
    if pd.api.types.is_datetime64_any_dtype(df[schema.time_col]):
        print(f"  - Time range: {df[schema.time_col].min()} to {df[schema.time_col].max()}")
    
    if schema.treatment_col in df.columns:
        treated = (df[schema.treatment_col] == 1).sum()
        print(f"  - Treated observations: {treated:,} ({100*treated/len(df):.1f}%)")


def wide_to_long(
    df: pd.DataFrame,
    schema: DataSchema
) -> pd.DataFrame:
    """
    Convert wide format data to long format based on schema.
    
    This transforms data with multiple outcome columns (one per group)
    into long format with a single outcome column and a group identifier.
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide format data with separate columns for each outcome group
    schema : DataSchema
        Schema defining outcome columns and labels
        
    Returns
    -------
    pd.DataFrame
        Long format data with columns: unit, time, group, outcome, [denominator], treatment, ...
    """
    df = df.copy()
    
    # Get ID columns (non-outcome columns to preserve)
    id_cols = [schema.unit_col, schema.time_col, schema.treatment_col]
    id_cols.extend(schema.additional_cols)
    id_cols = [c for c in id_cols if c in df.columns]
    
    # Build long format by stacking outcomes
    long_dfs = []
    
    for outcome_spec in schema.outcomes:
        outcome_col = outcome_spec.outcome_col
        denom_col = outcome_spec.denominator_col
        label = outcome_spec.label
        
        # Select columns for this outcome
        cols_to_select = id_cols + [outcome_col]
        if denom_col and denom_col in df.columns:
            cols_to_select.append(denom_col)
        
        # Create subset
        subset = df[cols_to_select].copy()
        subset['group'] = label
        subset = subset.rename(columns={outcome_col: 'outcome'})
        
        if denom_col and denom_col in df.columns:
            subset = subset.rename(columns={denom_col: 'denominator'})
        
        long_dfs.append(subset)
    
    # Combine all groups
    df_long = pd.concat(long_dfs, ignore_index=True)
    
    # Rename columns to standard names
    df_long = df_long.rename(columns={
        schema.unit_col: 'unit',
        schema.time_col: 'time',
        schema.treatment_col: 'treatment'
    })
    
    # Sort for consistency
    df_long = df_long.sort_values(['unit', 'time', 'group']).reset_index(drop=True)
    
    return df_long


def create_total_outcome(
    df: pd.DataFrame,
    outcome_cols: List[str],
    denominator_cols: Optional[List[str]] = None,
    total_outcome_col: str = "outcome_total",
    total_denominator_col: str = "denominator_total"
) -> pd.DataFrame:
    """
    Create total outcome by summing multiple outcome columns.
    
    Useful when you want to analyze both individual groups and their total.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with multiple outcome columns
    outcome_cols : list of str
        Columns to sum for total outcome
    denominator_cols : list of str, optional
        Columns to sum for total denominator
    total_outcome_col : str
        Name for the new total outcome column
    total_denominator_col : str
        Name for the new total denominator column
        
    Returns
    -------
    pd.DataFrame
        Data with new total columns added
    """
    df = df.copy()
    
    # Sum outcomes
    df[total_outcome_col] = df[outcome_cols].sum(axis=1)
    
    # Sum denominators if provided
    if denominator_cols:
        df[total_denominator_col] = df[denominator_cols].sum(axis=1)
    
    return df
