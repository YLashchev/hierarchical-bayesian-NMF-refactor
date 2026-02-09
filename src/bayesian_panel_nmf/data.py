"""
Unified data loading and preparation for bayesian_panel_nmf.

This module provides a single entry point for loading panel data and
preparing it for the Bayesian hierarchical model. All internal operations
use FIXED standardized column names: unit, time, group, outcome, denominator, treatment.
"""

from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Standard column names (FIXED - used internally after loading)
# =============================================================================
UNIT_COL = "unit"
TIME_COL = "time"
GROUP_COL = "group"
OUTCOME_COL = "outcome"
DENOMINATOR_COL = "denominator"
TREATMENT_COL = "treatment"


# =============================================================================
# Main public function
# =============================================================================
def load_and_prepare(
    filepath: str,
    config: Dict[str, Any],
    groups: List[str]
) -> Dict[str, Any]:
    """
    Load CSV, standardize columns, filter, aggregate, and prepare model arrays.

    This is the single entry point for all data preparation. After loading,
    all operations use fixed column names: unit, time, group, outcome,
    denominator, treatment.

    Parameters
    ----------
    filepath : str
        Path to input CSV file
    config : dict
        Full config dict with 'data', 'model', 'mcmc' sections
    groups : list of str
        Which outcome groups to include (e.g., ["total"] or ["usborn", "foreign"])

    Returns
    -------
    dict
        data_dict with keys:
        - Y: ndarray (K, D, N) outcome counts
        - denominators: ndarray (K, D, N) populations (scaled)
        - control_idx_array: ndarray (K, D, N) bool, True=control period
        - missing_idx_array: ndarray (K, D, N) bool, True=missing
        - groups: list of str (K labels)
        - units: list of str (D labels)
        - times: list of datetime (N labels)
        - df_preprocessed: DataFrame with standardized columns
    """
    data_config = config.get("data", {})

    # Parse schema from config
    schema_info = _parse_schema(config)

    # Load and standardize column names
    df = _load_and_standardize(filepath, schema_info, data_config.get("date_format", "auto"))

    # Convert wide format to long format with standard columns
    df = _wide_to_long(df, schema_info, groups)

    # Filter time range
    df = _filter_time_range(
        df,
        data_config.get("start_date"),
        data_config.get("end_date")
    )

    # Aggregate temporally if configured
    agg_config = data_config.get("aggregation", {})
    if agg_config.get("enabled", False):
        df = _aggregate_temporal(df, agg_config.get("period", "bimonthly"))

    # Build model arrays
    data_dict = _build_model_arrays(df, groups)

    # Include preprocessed DataFrame for output merging
    data_dict["df_preprocessed"] = df

    return data_dict


# =============================================================================
# Internal helper functions (all use FIXED column names)
# =============================================================================
def _parse_schema(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract column mapping from config['data']['schema'].

    Returns a dict with keys: unit_col, time_col, treatment_col, outcomes.
    Each outcome has: outcome_col, denominator_col, label.
    """
    schema_cfg = config["data"]["schema"]

    outcomes = []
    for o in schema_cfg["outcomes"]:
        outcomes.append({
            "outcome_col": o["outcome_col"],
            "denominator_col": o.get("denominator_col"),
            "label": o["label"]
        })

    return {
        "unit_col": schema_cfg["unit_col"],
        "time_col": schema_cfg["time_col"],
        "treatment_col": schema_cfg["treatment_col"],
        "outcomes": outcomes,
        "additional_cols": schema_cfg.get("additional_cols", [])
    }


def _load_and_standardize(
    filepath: str,
    schema_info: Dict[str, Any],
    date_format: str = "auto"
) -> pd.DataFrame:
    """
    Load CSV and parse time column.

    Does NOT rename columns yet - that happens in _wide_to_long.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(path)

    # Validate required columns exist
    time_col = schema_info["time_col"]
    unit_col = schema_info["unit_col"]
    treatment_col = schema_info["treatment_col"]

    required = [unit_col, time_col, treatment_col]
    for o in schema_info["outcomes"]:
        required.append(o["outcome_col"])
        if o["denominator_col"]:
            required.append(o["denominator_col"])

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse time column
    if date_format == "auto":
        formats_to_try = [None, "%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y", "%d-%m-%Y"]
        for fmt in formats_to_try:
            try:
                df[time_col] = pd.to_datetime(df[time_col], format=fmt)
                break
            except (ValueError, TypeError):
                continue
    else:
        df[time_col] = pd.to_datetime(df[time_col], format=date_format)

    return df


def _wide_to_long(
    df: pd.DataFrame,
    schema_info: Dict[str, Any],
    groups: List[str]
) -> pd.DataFrame:
    """
    Convert wide format to long format with standard column names.

    Handles "total" group by summing all outcomes/denominators.
    Output columns: unit, time, group, outcome, denominator, treatment.
    """
    unit_col = schema_info["unit_col"]
    time_col = schema_info["time_col"]
    treatment_col = schema_info["treatment_col"]
    outcomes = schema_info["outcomes"]

    # Check if "total" is requested but not defined
    defined_labels = [o["label"] for o in outcomes]
    needs_total = "total" in groups and "total" not in defined_labels

    long_dfs = []

    for o in outcomes:
        label = o["label"]
        # Skip if not in requested groups (unless we need it for total)
        if label not in groups and not needs_total:
            continue

        subset = df[[unit_col, time_col, treatment_col, o["outcome_col"]]].copy()
        subset[GROUP_COL] = label
        subset = subset.rename(columns={o["outcome_col"]: OUTCOME_COL})

        if o["denominator_col"] and o["denominator_col"] in df.columns:
            subset[DENOMINATOR_COL] = df[o["denominator_col"]]

        long_dfs.append(subset)

    if not long_dfs:
        raise ValueError(f"No matching outcomes found for groups: {groups}")

    df_long = pd.concat(long_dfs, ignore_index=True)

    # Rename ID columns to standard names
    df_long = df_long.rename(columns={
        unit_col: UNIT_COL,
        time_col: TIME_COL,
        treatment_col: TREATMENT_COL
    })

    # Handle "total" group by aggregating all outcomes
    if needs_total:
        df_total = df_long.groupby([UNIT_COL, TIME_COL, TREATMENT_COL], as_index=False).agg({
            OUTCOME_COL: "sum",
            DENOMINATOR_COL: "sum" if DENOMINATOR_COL in df_long.columns else "first"
        })
        df_total[GROUP_COL] = "total"

        # Filter to only requested groups
        df_long = df_long[df_long[GROUP_COL].isin(groups)]
        if "total" in groups:
            df_long = pd.concat([df_long, df_total], ignore_index=True)
    else:
        df_long = df_long[df_long[GROUP_COL].isin(groups)]

    # Sort for consistency
    df_long = df_long.sort_values([UNIT_COL, TIME_COL, GROUP_COL]).reset_index(drop=True)

    return df_long


def _filter_time_range(
    df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str]
) -> pd.DataFrame:
    """Filter DataFrame by date range using fixed TIME_COL."""
    if start_date is not None:
        df = df[df[TIME_COL] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df[TIME_COL] < pd.to_datetime(end_date)]
    return df.reset_index(drop=True)


def _aggregate_temporal(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Aggregate data to specified time periods.

    Supports: monthly, bimonthly, quarterly, yearly.
    """
    df = df.copy()

    # Extract year/month
    df["_year"] = df[TIME_COL].dt.year
    df["_month"] = df[TIME_COL].dt.month

    # Calculate period code
    period_map = {
        "monthly": (1, lambda m: m),
        "bimonthly": (2, lambda m: ((m - 1) // 2) + 1),
        "quarterly": (3, lambda m: ((m - 1) // 3) + 1),
        "yearly": (12, lambda m: 1)
    }

    if period not in period_map:
        raise ValueError(f"Unknown period: {period}. Use: monthly, bimonthly, quarterly, yearly")

    months_per_period, period_func = period_map[period]
    df["_period"] = df["_month"].apply(period_func)

    # Define aggregation
    agg_dict = {
        OUTCOME_COL: "sum",
        TREATMENT_COL: "max"  # Treated if any sub-period is treated
    }
    if DENOMINATOR_COL in df.columns:
        agg_dict[DENOMINATOR_COL] = "mean"

    group_cols = [UNIT_COL, "_year", "_period", GROUP_COL]
    df_agg = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Create representative time (first month of period)
    first_month = (df_agg["_period"] - 1) * months_per_period + 1
    df_agg[TIME_COL] = pd.to_datetime(
        df_agg["_year"].astype(str) + "-" + first_month.astype(str) + "-01"
    )

    # Add period boundaries for reference
    df_agg["start_date"] = df_agg[TIME_COL]
    df_agg["end_date"] = df_agg[TIME_COL] + pd.DateOffset(months=months_per_period) - timedelta(days=1)

    # Clean up
    df_agg = df_agg.drop(columns=["_year", "_period"])
    df_agg = df_agg.sort_values([UNIT_COL, TIME_COL, GROUP_COL]).reset_index(drop=True)

    return df_agg


def _build_model_arrays(
    df: pd.DataFrame,
    groups: List[str],
    denominator_scale: float = 1e4
) -> Dict[str, Any]:
    """
    Convert long-format DataFrame to K×D×N arrays for the model.

    Uses FIXED column names: unit, time, group, outcome, denominator, treatment.

    Parameters
    ----------
    df : pd.DataFrame
        Long format data with standardized columns
    groups : list of str
        Group labels in desired order (defines K dimension)
    denominator_scale : float
        Scale factor for denominators (default 1e4 = per 10k)

    Returns
    -------
    dict
        Y, denominators, control_idx_array, missing_idx_array, groups, units, times
    """
    # Filter to requested groups and sort
    df = df[df[GROUP_COL].isin(groups)].copy()
    df = df.sort_values([UNIT_COL, TIME_COL, GROUP_COL]).reset_index(drop=True)

    # Get dimensions
    units = sorted(df[UNIT_COL].unique())
    times = sorted(df[TIME_COL].unique())
    K, D, N = len(groups), len(units), len(times)

    # Initialize arrays
    Y = np.zeros((K, D, N))
    denominators = np.ones((K, D, N))  # Default to 1 for count modeling
    control_idx = np.ones((K, D, N), dtype=bool)
    missing_idx = np.zeros((K, D, N), dtype=bool)

    # Create index mappings
    group_to_k = {g: i for i, g in enumerate(groups)}
    unit_to_d = {u: i for i, u in enumerate(units)}
    time_to_n = {t: i for i, t in enumerate(times)}

    # Fill arrays
    has_denominator = DENOMINATOR_COL in df.columns

    for _, row in df.iterrows():
        k = group_to_k[row[GROUP_COL]]
        d = unit_to_d[row[UNIT_COL]]
        n = time_to_n[row[TIME_COL]]

        # Outcome
        outcome_val = row[OUTCOME_COL]
        if pd.isna(outcome_val):
            Y[k, d, n] = 0
            missing_idx[k, d, n] = True
        else:
            Y[k, d, n] = outcome_val

        # Denominator (if present)
        if has_denominator:
            denom_val = row[DENOMINATOR_COL]
            if pd.notna(denom_val) and denom_val > 0:
                denominators[k, d, n] = denom_val / denominator_scale

        # Control status (treatment=0 means control)
        control_idx[k, d, n] = (row[TREATMENT_COL] == 0)

    return {
        "Y": Y,
        "denominators": denominators,
        "control_idx_array": control_idx,
        "missing_idx_array": missing_idx,
        "groups": groups,
        "units": units,
        "times": times
    }
