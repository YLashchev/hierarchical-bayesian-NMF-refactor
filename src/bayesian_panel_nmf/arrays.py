"""Reshape a standardized long DataFrame into (K, D, N) model arrays.

The single job here is the tensor build: long rows -> dense group x unit x
time arrays for the NumPyro model. Column names are fixed (unit, time, group,
outcome, denominator, treatment); ingestion/standardization lives in data.py.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from bayesian_panel_nmf.validation import DataError

UNIT_COL = "unit"
TIME_COL = "time"
GROUP_COL = "group"
OUTCOME_COL = "outcome"
DENOMINATOR_COL = "denominator"
TREATMENT_COL = "treatment"


def build_model_arrays(
    df: pd.DataFrame,
    groups: list[str],
    denominator_scale: float = 1e4,
    allow_unbalanced_panel: bool = False,
) -> dict[str, Any]:
    """
    Convert a long-format DataFrame to K x D x N arrays for the model.

    Uses FIXED column names: unit, time, group, outcome, denominator, treatment.
    Each (group, unit, time) cell is unique here — ``load_and_prepare`` raises
    on duplicate keys upstream — so the fill is a vectorized scatter, not a
    row-by-row loop.

    Parameters
    ----------
    df : pd.DataFrame
        Long format data with standardized columns.
    groups : list of str
        Group labels in desired order (defines K dimension).
    denominator_scale : float
        Scale factor for denominators (default 1e4 = per 10k).
    allow_unbalanced_panel : bool
        If False, raise when any (group, unit, time) cell has no row.

    Returns
    -------
    dict
        Y, denominators, control_idx_array, missing_idx_array, groups, units, times
    """
    # Filter to requested groups and sort (sort kept for deterministic
    # units/times ordering; the scatter itself is order-independent).
    df = df[df[GROUP_COL].isin(groups)].copy()
    df = df.sort_values([UNIT_COL, TIME_COL, GROUP_COL]).reset_index(drop=True)

    # Dimensions — full set of units/times across the filtered data.
    units = sorted(df[UNIT_COL].unique())
    times = sorted(df[TIME_COL].unique())
    K, D, N = len(groups), len(units), len(times)

    # Defaults match the legacy loop: Y=0, denominators=1, control=True,
    # missing=False, filled=False.
    Y = np.zeros((K, D, N))
    denominators = np.ones((K, D, N))
    control_idx = np.ones((K, D, N), dtype=bool)
    missing_idx = np.zeros((K, D, N), dtype=bool)
    filled = np.zeros((K, D, N), dtype=bool)

    # Map each row's labels to integer axis indices via pandas categoricals
    # (vectorized equivalent of the per-row dict lookups).
    k_idx = pd.Categorical(df[GROUP_COL], categories=groups).codes.astype(np.intp)
    d_idx = pd.Categorical(df[UNIT_COL], categories=units).codes.astype(np.intp)
    n_idx = pd.Categorical(df[TIME_COL], categories=times).codes.astype(np.intp)

    outcome = df[OUTCOME_COL].to_numpy()
    outcome_missing = pd.isna(df[OUTCOME_COL]).to_numpy()

    # Y: NaN -> 0 (already zero) + mark missing; else the value.
    present = ~outcome_missing
    Y[k_idx[present], d_idx[present], n_idx[present]] = outcome[present].astype(float)
    missing_idx[k_idx[outcome_missing], d_idx[outcome_missing], n_idx[outcome_missing]] = True

    # Denominator (if present): notna AND >0 -> val/scale; else leave default 1.
    if DENOMINATOR_COL in df.columns:
        denom = df[DENOMINATOR_COL].to_numpy()
        denom_ok = pd.notna(df[DENOMINATOR_COL]).to_numpy() & (
            np.nan_to_num(denom, nan=0.0) > 0
        )
        denominators[k_idx[denom_ok], d_idx[denom_ok], n_idx[denom_ok]] = (
            denom[denom_ok].astype(float) / denominator_scale
        )

    # Control status: treatment == 0 means control.
    is_control = (df[TREATMENT_COL].to_numpy() == 0)
    control_idx[k_idx, d_idx, n_idx] = is_control
    filled[k_idx, d_idx, n_idx] = True

    # Structurally absent cells (no row in data).
    absent_mask = ~filled
    n_absent = int(absent_mask.sum())
    n_total = K * D * N

    if n_absent > 0:
        if not allow_unbalanced_panel:
            raise DataError(
                f"Unbalanced panel: {n_absent} of {n_total} cells are missing. "
                "Set allow_unbalanced_panel=True in config to allow."
            )
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
