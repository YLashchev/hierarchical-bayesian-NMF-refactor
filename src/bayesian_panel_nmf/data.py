"""Load, validate, and reshape panel CSVs into model arrays.

After loading, all operations use fixed column names: unit, time, group,
outcome, denominator, treatment.
"""

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from bayesian_panel_nmf.validation import (
    validate_filepath,
    validate_config,
    validate_groups,
    DataError,
    ConfigError,
)


# Standard column names (fixed after loading)
UNIT_COL = "unit"
TIME_COL = "time"
GROUP_COL = "group"
OUTCOME_COL = "outcome"
DENOMINATOR_COL = "denominator"
TREATMENT_COL = "treatment"


def load_and_prepare(
    filepath: str,
    config: dict,
    groups: list[str],
    exclude_units: list[str] | None = None,
    type_config: dict | None = None,
) -> dict:
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
    exclude_units : list of str, optional
        Units (e.g., states) to exclude from the analysis. Useful when
        certain units have missing data for specific subgroups (e.g.,
        California lacks marital status data for births).

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

    Raises
    ------
    ValidationError
        If filepath doesn't exist or is invalid
    ConfigValidationError
        If config is missing required sections or keys
    DataValidationError
        If data file is missing required columns or has invalid values
    """
    validate_filepath(filepath)
    validate_config(config)
    validate_groups(groups)

    data_config = config.get("data", {})

    # Peek at CSV columns to resolve prefix-based outcomes
    _header = pd.read_csv(filepath, nrows=0)
    available_columns = _header.columns.tolist()

    schema_info = _parse_schema(config, available_columns=available_columns)

    df = _load_and_standardize(
        filepath, schema_info, data_config.get("date_format", "auto")
    )

    total_from_labels = _validate_and_resolve_total(
        groups, schema_info["outcomes"], type_config
    )
    df = _wide_to_long(df, schema_info, groups, total_from_labels=total_from_labels)

    dupe_cols = [GROUP_COL, UNIT_COL, TIME_COL]
    dupes = df[df.duplicated(subset=dupe_cols, keep=False)]
    if not dupes.empty:
        n_dupes = dupes.duplicated(subset=dupe_cols).sum()
        sample_units = sorted(dupes[UNIT_COL].unique().tolist())[:3]
        raise DataError(
            f"duplicate group/unit/time combinations: {n_dupes} rows affected "
            f"(e.g., units: {sample_units})"
        )

    start_date = data_config.get("start_date")
    end_date = data_config.get("end_date")
    df = _filter_time_range(df, start_date, end_date)

    if exclude_units:
        before_count = df[UNIT_COL].nunique()
        df = df[~df[UNIT_COL].isin(exclude_units)].reset_index(drop=True)
        after_count = df[UNIT_COL].nunique()
        logger.info(
            f"Excluded units {exclude_units}: {before_count} → {after_count} units"
        )

    agg_config = data_config.get("aggregation", {})
    if agg_config.get("enabled", False):
        period = agg_config.get("period", "bimonthly")
        df = _aggregate_temporal(df, period)
        logger.info(f"Aggregated to {period}: {len(df)} rows")

    allow_unbalanced = data_config.get("allow_unbalanced_panel", False)
    data_dict = _build_model_arrays(df, groups, allow_unbalanced_panel=allow_unbalanced)

    K, D, N = data_dict["Y"].shape
    logger.info(f"Model arrays built: K={K} groups, D={D} units, N={N} time periods")

    # Preprocessed DataFrame returned for downstream output merging
    data_dict["df_preprocessed"] = df

    return data_dict


def _validate_and_resolve_total(
    groups: list[str],
    schema_outcomes: list[dict],
    type_config: dict | None,
) -> list[str] | None:
    """
    Validate synthetic 'total' configuration and return labels to sum.

    Returns list of outcome labels to aggregate for 'total', or None if
    'total' is explicitly defined or not requested.

    Raises
    ------
    ConfigError
        If 'total' is requested but not properly configured.
    """
    defined_labels = [o["label"] for o in schema_outcomes]

    # If total is not requested or is explicitly defined, no synthetic aggregation needed
    if "total" not in groups or "total" in defined_labels:
        return None

    # Total is requested but not defined — need synthetic aggregation
    if type_config is None:
        raise ConfigError(
            "'total' is not defined as an outcome label in the schema. "
            "Provide type_config with 'total_from' or 'total_all' to create it."
        )

    total_from = type_config.get("total_from")
    total_all = type_config.get("total_all", False)

    if total_from is not None:
        # Validate total_from labels
        if len(total_from) != len(set(total_from)):
            raise ConfigError(f"total_from contains duplicate labels: {total_from}")
        undefined = [lbl for lbl in total_from if lbl not in defined_labels]
        if undefined:
            raise ConfigError(
                f"total_from references undefined outcome labels: {undefined}"
            )
        return total_from
    if total_all:
        return defined_labels
    raise ConfigError(
        "'total' is not defined as an outcome label in the schema. "
        "Provide type_config with 'total_from' or 'total_all' to create it."
    )


def _resolve_outcomes_from_prefixes(
    prefix_cfg: dict, available_columns: list[str]
) -> list[dict[str, str | None]]:
    if available_columns is None:
        raise DataError(
            "available_columns must be provided when using outcomes_from_prefixes"
        )

    outcome_prefix = prefix_cfg["outcome_prefix"]
    denominator_prefix = prefix_cfg.get("denominator_prefix")
    include = set(prefix_cfg.get("include") or [])

    outcome_cols = [c for c in available_columns if c.startswith(outcome_prefix)]
    if not outcome_cols:
        raise DataError(f"No columns found with outcome prefix '{outcome_prefix}'")

    outcome_labels = {
        c[len(outcome_prefix) :] for c in outcome_cols if c[len(outcome_prefix) :]
    }

    if include:
        if missing := include - outcome_labels:
            raise DataError(
                f"Missing outcome columns for include labels: {list(missing)}"
            )
        outcome_cols = [c for c in outcome_cols if c[len(outcome_prefix) :] in include]

    outcomes = []
    for col in outcome_cols:
        label = col[len(outcome_prefix) :]
        if not label:
            continue

        denom_col = f"{denominator_prefix}{label}" if denominator_prefix else None
        if denom_col and denom_col not in available_columns:
            raise DataError(f"Missing denominator columns for labels: ['{label}']")

        outcomes.append(
            {
                "outcome_col": col,
                "denominator_col": denom_col,
                "label": label,
            }
        )

    if not outcomes:
        raise DataError(f"No outcome groups found with prefix '{outcome_prefix}'")

    return outcomes


def _resolve_outcomes(
    schema_cfg: dict, available_columns: list[str] | None
) -> list[dict[str, str | None]]:
    if schema_cfg.get("outcomes") is not None:
        outcomes = [
            {
                "outcome_col": outcome["outcome_col"],
                "denominator_col": outcome.get("denominator_col"),
                "label": outcome["label"],
            }
            for outcome in schema_cfg["outcomes"]
        ]
    else:
        if available_columns is None:
            raise DataError(
                "outcomes_from_prefixes requires available_columns from the CSV header"
            )
        outcomes = _resolve_outcomes_from_prefixes(
            schema_cfg["outcomes_from_prefixes"],
            available_columns,
        )

    labels = [outcome["label"] for outcome in outcomes]
    if len(labels) != len(set(labels)):
        raise ConfigError(f"duplicate outcome labels: {sorted(labels)}")

    return outcomes


def _parse_schema(
    config: dict,
    available_columns: list[str] | None = None,
) -> dict:
    """Extract schema configuration from config dict."""
    schema_cfg = config["data"]["schema"]
    outcomes = _resolve_outcomes(schema_cfg, available_columns)
    return {
        "unit_col": schema_cfg["unit_col"],
        "time_col": schema_cfg["time_col"],
        "treatment_col": schema_cfg["treatment_col"],
        "outcomes": outcomes,
    }


def _validate_schema_cols_in_df(
    df: pd.DataFrame, schema_info: dict
) -> tuple[str, str, str]:
    time_col = schema_info["time_col"]
    unit_col = schema_info["unit_col"]
    treatment_col = schema_info["treatment_col"]

    required = [
        unit_col,
        time_col,
        treatment_col,
        *[o["outcome_col"] for o in schema_info["outcomes"]],
        *[
            o["denominator_col"]
            for o in schema_info["outcomes"]
            if o["denominator_col"]
        ],
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise DataError(f"Missing columns: {sorted(missing)}")

    _validate_denominators(df, schema_info, unit_col)
    return time_col, unit_col, treatment_col


def _validate_denominators(df: pd.DataFrame, schema_info: dict, unit_col: str) -> None:
    for o in schema_info["outcomes"]:
        denom_col = o.get("denominator_col")
        if denom_col and denom_col in df.columns:
            bad_mask = df[denom_col].isna() | (df[denom_col] <= 0)
            if bad_mask.any():
                bad_rows = df[bad_mask]
                bad_units = bad_rows[unit_col].unique().tolist()
                n_bad = bad_mask.sum()
                raise DataError(
                    f"NaN or non-positive denominator in column '{denom_col}': "
                    f"{n_bad} rows affected (units: {bad_units})"
                )


def _load_and_standardize(
    filepath: str, schema_info: dict, date_format: str = "auto"
) -> pd.DataFrame:
    """
    Load CSV and parse time column.

    Does NOT rename columns yet - that happens in _wide_to_long.

    Raises
    ------
    DataValidationError
        If required columns are missing, time cannot be parsed, or treatment has invalid values
    """
    path = Path(filepath)
    if not path.exists():
        logger.error(f"Data file not found: {filepath}")
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(path)

    time_col, unit_col, treatment_col = _validate_schema_cols_in_df(df, schema_info)

    # Parse time column
    if date_format == "auto":
        formats_to_try = [None, "%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y", "%d-%m-%Y"]
        parsed = False
        for fmt in formats_to_try:
            try:
                df[time_col] = pd.to_datetime(df[time_col], format=fmt)
                parsed = True
                break
            except (ValueError, TypeError):
                continue
        if not parsed:
            raise DataError(f"Could not parse time column '{time_col}'")
    else:
        try:
            df[time_col] = pd.to_datetime(df[time_col], format=date_format)
        except ValueError as e:
            raise DataError(f"Could not parse time column '{time_col}': {e}")

    return df


def _build_long_subset(
    outcome: dict, df: pd.DataFrame, unit_col: str, time_col: str, treatment_col: str
) -> pd.DataFrame:
    """Build a long-format subset for one outcome group."""
    subset = df[[unit_col, time_col, treatment_col, outcome["outcome_col"]]].copy()
    subset[GROUP_COL] = outcome["label"]
    subset = subset.rename(columns={outcome["outcome_col"]: OUTCOME_COL})
    if outcome["denominator_col"] and outcome["denominator_col"] in df.columns:
        subset[DENOMINATOR_COL] = df[outcome["denominator_col"]]
    return subset


def _build_long_dfs(
    df: pd.DataFrame, schema_info: dict, groups: list[str], total_labels: list[str]
) -> list[pd.DataFrame]:
    unit_col = schema_info["unit_col"]
    time_col = schema_info["time_col"]
    treatment_col = schema_info["treatment_col"]

    long_dfs = [
        _build_long_subset(o, df, unit_col, time_col, treatment_col)
        for o in schema_info["outcomes"]
        if o["label"] in groups or o["label"] in total_labels
    ]

    if not long_dfs:
        logger.error(f"No matching outcomes found for groups: {groups}")
        raise ValueError(f"No matching outcomes found for groups: {groups}")

    return long_dfs


def _wide_to_long(
    df: pd.DataFrame,
    schema_info: dict,
    groups: list[str],
    total_from_labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert wide format to long format with standard column names.

    Handles "total" group by summing all outcomes/denominators.
    Output columns: unit, time, group, outcome, denominator, treatment.
    """
    # Check if "total" is requested but not defined
    defined_labels = [o["label"] for o in schema_info["outcomes"]]
    needs_total = "total" in groups and "total" not in defined_labels

    # Determine which labels to sum for synthetic total
    if needs_total and total_from_labels is None:
        raise ConfigError(
            "'total' is not defined as an outcome label in the schema. "
            "Provide type_config with 'total_from' or 'total_all' to create it."
        )
    total_labels: list[str] = (
        total_from_labels if needs_total and total_from_labels else []
    )

    long_dfs = _build_long_dfs(df, schema_info, groups, total_labels)

    df_long = pd.concat(long_dfs, ignore_index=True)

    # Rename ID columns to standard names
    df_long = df_long.rename(
        columns={
            schema_info["unit_col"]: UNIT_COL,
            schema_info["time_col"]: TIME_COL,
            schema_info["treatment_col"]: TREATMENT_COL,
        }
    )

    # Handle "total" group by aggregating specified subgroups
    if needs_total and total_labels:
        # Filter to only the labels we need to sum
        df_for_total = df_long[df_long[GROUP_COL].isin(total_labels)]
        agg_dict: dict[str, str] = {OUTCOME_COL: "sum"}
        has_denom = DENOMINATOR_COL in df_for_total.columns
        if has_denom:
            agg_dict[DENOMINATOR_COL] = "sum"
        df_total = df_for_total.groupby(
            [UNIT_COL, TIME_COL, TREATMENT_COL], as_index=False
        ).agg(agg_dict)
        df_total[GROUP_COL] = "total"
        if not has_denom:
            df_total[DENOMINATOR_COL] = 1

        # Filter to only requested groups
        df_long = df_long[df_long[GROUP_COL].isin(groups)]
        if "total" in groups:
            df_long = pd.concat([df_long, df_total], ignore_index=True)
    else:
        df_long = df_long[df_long[GROUP_COL].isin(groups)]

    # Sort for consistency
    df_long = df_long.sort_values([UNIT_COL, TIME_COL, GROUP_COL]).reset_index(
        drop=True
    )

    return df_long


def _filter_time_range(
    df: pd.DataFrame, start_date: str | None, end_date: str | None
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
        "yearly": (12, lambda m: 1),
    }

    if period not in period_map:
        logger.error(f"Unknown aggregation period: {period}")
        raise ValueError(
            f"Unknown period: {period}. Use: monthly, bimonthly, quarterly, yearly"
        )

    months_per_period, period_func = period_map[period]
    df["_period"] = df["_month"].apply(period_func)

    # Define aggregation
    agg_dict = {
        OUTCOME_COL: "sum",
        TREATMENT_COL: "max",  # Treated if any sub-period is treated
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
    df_agg["end_date"] = (
        df_agg[TIME_COL] + pd.DateOffset(months=months_per_period) - timedelta(days=1)
    )

    # Clean up
    df_agg = df_agg.drop(columns=["_year", "_period"])
    df_agg = df_agg.sort_values([UNIT_COL, TIME_COL, GROUP_COL]).reset_index(drop=True)

    return df_agg


def _build_model_arrays(
    df: pd.DataFrame,
    groups: list[str],
    denominator_scale: float = 1e4,
    allow_unbalanced_panel: bool = False,
) -> dict[str, Any]:
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

    # Get dimensions — use FULL set of units/times across all data
    units = sorted(df[UNIT_COL].unique())
    times = sorted(df[TIME_COL].unique())
    K, D, N = len(groups), len(units), len(times)

    # Initialize arrays
    Y = np.zeros((K, D, N))
    denominators = np.ones((K, D, N))  # Default to 1 for count modeling
    control_idx = np.ones((K, D, N), dtype=bool)
    missing_idx = np.zeros((K, D, N), dtype=bool)
    filled = np.zeros((K, D, N), dtype=bool)  # Track which cells have data

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
        control_idx[k, d, n] = row[TREATMENT_COL] == 0
        filled[k, d, n] = True

    # Check for structurally absent cells (no row in data)
    absent_mask = ~filled
    n_absent = int(absent_mask.sum())
    n_total = K * D * N

    if n_absent > 0:
        if not allow_unbalanced_panel:
            raise DataError(
                f"Unbalanced panel: {n_absent} of {n_total} cells are missing. "
                "Set allow_unbalanced_panel=True in config to allow."
            )
        # Mark absent cells as missing
        missing_idx = missing_idx | absent_mask
        logger.warning(
            f"Unbalanced panel: {n_absent} of {n_total} cells are structurally absent (marked as missing)"
        )

    return {
        "Y": Y,
        "denominators": denominators,
        "control_idx_array": control_idx,
        "missing_idx_array": missing_idx,
        "groups": groups,
        "units": units,
        "times": times,
    }
