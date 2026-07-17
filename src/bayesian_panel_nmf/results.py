"""Draws -> tidy DataFrame: the reporting schema for both inference modes.

Joint mode uses ``format_draws``; cut mode adds source-tagged component
frames (``format_cut_component_draws``), the Stage-1 PPC frame
(``format_stage1_ppc_draws``), and the convergence manifest
(``build_cut_convergence_manifest``). Transforms only -- all file I/O stays in
cli.py. Uses fixed column names; downcasts dtypes
(categorical for unit/group, int8/int32 for integer cols, float32 for
predictions).
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from bayesian_panel_nmf.checks import validate_predictions, validate_samples
from bayesian_panel_nmf.validation import DataError

if TYPE_CHECKING:
    from bayesian_panel_nmf.cut import Stage1DrawRef

# Columns tagging which Stage-1 draw each cut row came from.
STAGE1_SOURCE_COLUMNS = [
    "cut_component",
    "stage1_draw",
    "stage1_chain",
    "stage1_iteration",
]


def drop_scoped_samples(samples: dict) -> dict:
    """Drop sample keys containing '/' (from numpyro.handlers.scope).

    xarray DataTree rejects '/' in variable names (path-separator conflict),
    so the convergence gate and trace sidecars need this filter before
    handing samples to az.from_dict.
    """
    return {k: v for k, v in samples.items() if "/" not in k}


def _downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast string/int/float columns in place to shrink the draws frame."""
    initial_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.debug(f"DataFrame memory before downcast: {initial_mem:.1f} MB")

    for col in ["unit", "group"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype("category")

    int_dtypes = {
        ".chain": "int8",
        ".draw": "int32",
        ".iteration": "int32",
        "treatment": "int8",
    }
    for col, dtype in int_dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    for col in ["ypred", "mu", "mu_treated", "outcome", "denominator"]:
        if col in df.columns and df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].astype("float32")

    final_mem = df.memory_usage(deep=True).sum() / 1024**2
    savings_pct = (1 - final_mem / initial_mem) * 100 if initial_mem > 0 else 0
    logger.debug(
        f"DataFrame memory after downcast: {final_mem:.1f} MB ({savings_pct:.1f}% reduction)"
    )

    return df


def _build_posterior_draws_frame(
    samples: dict[str, np.ndarray],
    predictions: np.ndarray,
    groups: list[str],
    units: list[str],
    times: list,
) -> pd.DataFrame:
    """Build the long-format posterior draws frame (one row per
    chain x sample x group x unit x time) from raw MCMC arrays.

    Returns the draws frame with .draw/.chain/.iteration, group, unit,
    time, ypred, mu, mu_treated, plus integer K/D/N index columns used
    as merge keys by _merge_observed_columns.
    """
    C, S, K, D, N = (
        predictions.shape[0],
        predictions.shape[1],
        predictions.shape[2],
        predictions.shape[3],
        predictions.shape[4],
    )

    mu_ctrl = samples["mu_ctrl"]  # (C, S, K, D, N)
    has_te = "te" in samples

    total_rows = C * S * K * D * N
    logger.info(
        f"Building draws dataframe: {C} chains x {S} samples x {K} groups x {D} units x {N} times"
    )
    logger.info(f"Total rows: {total_rows:,}")

    # meshgrid builds the full (C,S,K,D,N) index grid vectorized
    idx_arrays = np.meshgrid(
        np.arange(C),
        np.arange(S),
        np.arange(K),
        np.arange(D),
        np.arange(N),
        indexing="ij",
    )
    c_idx, s_idx, k_idx, d_idx, n_idx = (
        idx_arrays[0],
        idx_arrays[1],
        idx_arrays[2],
        idx_arrays[3],
        idx_arrays[4],
    )

    c_flat = c_idx.ravel()
    s_flat = s_idx.ravel()
    k_flat = k_idx.ravel()
    d_flat = d_idx.ravel()
    n_flat = n_idx.ravel()

    # draw_num / chain_num / iter_num are 1-indexed for downstream R/tidybayes consumers
    draw_num = c_flat * S + s_flat + 1
    chain_num = c_flat + 1
    iter_num = s_flat + 1

    ypred_flat = predictions.ravel().astype(np.float32)
    mu_flat = mu_ctrl.ravel().astype(np.float32)

    if has_te:
        te_flat = samples["te"].ravel().astype(np.float32)
        mu_treated_flat = mu_flat + te_flat
    else:
        mu_treated_flat = mu_flat.copy()

    groups_arr = np.array(groups)
    units_arr = np.array(units)
    times_arr = np.array(times)

    group_flat = groups_arr[k_flat]
    unit_flat = units_arr[d_flat]
    time_flat = times_arr[n_flat]

    draws_df = pd.DataFrame(
        {
            ".draw": draw_num.astype(np.int32),
            ".chain": chain_num.astype(np.int8),
            ".iteration": iter_num.astype(np.int32),
            "K": k_flat.astype(np.int16),
            "D": d_flat.astype(np.int16),
            "N": n_flat.astype(np.int16),
            "group": pd.Categorical(group_flat, categories=groups),
            "unit": pd.Categorical(unit_flat, categories=units),
            "time": time_flat,
            "ypred": ypred_flat,
            "mu": mu_flat,
            "mu_treated": mu_treated_flat,
        }
    )

    logger.debug(f"Draws dataframe built: {len(draws_df):,} rows")

    return draws_df


def _merge_observed_columns(
    draws_df: pd.DataFrame,
    groups: list[str],
    units: list[str],
    times: list,
    df_preprocessed: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join observed outcome/denominator/treatment columns onto the
    posterior draws frame, keyed on (group, unit, time) via integer
    index columns, then drop the join-key columns and fix output column
    order.
    """
    obs_df = df_preprocessed.copy()

    group_to_idx = {g: i for i, g in enumerate(groups)}
    unit_to_idx = {u: i for i, u in enumerate(units)}
    time_to_idx = {t: i for i, t in enumerate(times)}

    obs_df["K"] = obs_df["group"].map(group_to_idx).astype(np.int16)
    obs_df["D"] = obs_df["unit"].map(unit_to_idx).astype(np.int16)
    obs_df["N"] = obs_df["time"].map(time_to_idx).astype(np.int16)

    obs_cols = ["outcome"]
    if "denominator" in obs_df.columns:
        obs_cols.append("denominator")
    if "treatment" in obs_df.columns:
        obs_cols.append("treatment")

    obs_subset = obs_df[["K", "D", "N"] + obs_cols].drop_duplicates()

    merged = draws_df.merge(obs_subset, on=["K", "D", "N"], how="left")
    merged = merged.drop(columns=["K", "D", "N"])

    col_order = [
        ".draw",
        ".chain",
        ".iteration",
        "unit",
        "time",
        "group",
        "outcome",
        "denominator",
        "treatment",
        "ypred",
        "mu",
        "mu_treated",
    ]
    col_order = [c for c in col_order if c in merged.columns]
    return merged[col_order]


def format_draws(
    samples: dict[str, np.ndarray], predictions: np.ndarray, data_dict: dict
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
    For 4 chains x 250 samples x 2 groups x 51 units x 48 times: ~75% memory reduction
    vs naive float64/object construction.
    """
    validate_samples(samples)
    validate_predictions(predictions, samples)

    required_keys = ["groups", "units", "times", "df_preprocessed"]
    missing = [k for k in required_keys if k not in data_dict]
    if missing:
        raise DataError(f"data_dict missing keys: {missing}")

    groups = data_dict["groups"]
    units = data_dict["units"]
    times = data_dict["times"]

    draws_df = _build_posterior_draws_frame(samples, predictions, groups, units, times)

    merged = _merge_observed_columns(
        draws_df, groups, units, times, data_dict["df_preprocessed"]
    )

    merged = _downcast_dtypes(merged)

    final_mem_mb = merged.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Output DataFrame: {len(merged):,} rows, {final_mem_mb:.1f} MB")

    return merged


# ---------------------------------------------------------------------------
# Cut-posterior formatting (merged from the former cut_output.py)
# ---------------------------------------------------------------------------
# The combined cut CSV keeps the reporting schema built by format_draws (one
# component at a time) plus the Stage-1 source columns. ``.draw`` is globally
# unique across components; ``.chain`` is the real Stage-2 chain within the
# component; ``.iteration`` indexes the component's output subsample.


def format_stage1_ppc_draws(mu_ctrl, ypred, data_dict: dict) -> pd.DataFrame:
    """Full Stage-1 posterior product (all retained draws, no ``te``).

    This frame feeds ONLY the JAMA PPC suite; effect summaries come from the
    combined cut draws.
    """
    return format_draws({"mu_ctrl": np.asarray(mu_ctrl)}, np.asarray(ypred), data_dict)


def format_cut_component_draws(
    te_draws,
    ypred,
    chain_ids,
    ref: "Stage1DrawRef",
    data_dict: dict,
    draw_offset: int,
) -> pd.DataFrame:
    """Format one conditional Stage-2 component into the reporting schema.

    Parameters
    ----------
    te_draws, ypred : arrays (n_out, K, D, N)
        Output-subsampled treatment effects and independently generated
        untreated predictive counts, draws concatenated in ascending
        Stage-2-chain order.
    chain_ids : array (n_out,)
        1-based real Stage-2 chain of each draw, grouped ascending.
    draw_offset : int
        Global ``.draw`` offset (sum of output draws of prior components).
    """
    te_draws = np.asarray(te_draws)
    ypred = np.asarray(ypred)
    chain_ids = np.asarray(chain_ids)
    n_out = te_draws.shape[0]
    cell_count = int(np.prod(te_draws.shape[1:]))
    mu_grid = np.broadcast_to(ref.mu_ctrl, te_draws.shape)

    df = format_draws(
        {"mu_ctrl": mu_grid[None], "te": te_draws[None]},
        ypred[None],
        data_dict,
    )

    # format_draws saw a (1, n_out) grid; restore real per-draw coordinates.
    # Rows are draw-major: each draw occupies cell_count consecutive rows.
    _, counts = np.unique(chain_ids, return_counts=True)
    iter_ids = np.concatenate([np.arange(1, c + 1) for c in counts])
    df[".chain"] = np.repeat(chain_ids.astype(np.int8), cell_count)
    df[".iteration"] = np.repeat(iter_ids.astype(np.int32), cell_count)
    df[".draw"] = (df[".draw"] + draw_offset).astype(np.int32)

    df["cut_component"] = np.int16(ref.component)
    df["stage1_draw"] = np.int32(ref.stage1_draw)
    df["stage1_chain"] = np.int8(ref.stage1_chain)
    df["stage1_iteration"] = np.int32(ref.stage1_iteration)
    assert n_out == 0 or len(df) == n_out * cell_count
    return df


def build_cut_convergence_manifest(
    stage1_diag: dict, component_records: list[dict]
) -> dict:
    """One manifest: Stage-1 gate plus every conditional fit's gate.

    Top-level ``converged`` is true only when Stage 1 AND every component
    passed. Aggregate counts are provided for convenience, but R-hat/ESS are
    never computed across pooled conditional targets.
    """
    all_converged = all(bool(r["converged"]) for r in component_records)
    return {
        "inference_mode": "cut",
        "converged": bool(stage1_diag["converged"]) and all_converged,
        "stage1": stage1_diag,
        "stage2": {
            "all_converged": all_converged,
            "failed_fits": int(sum(not r["converged"] for r in component_records)),
            "fits": component_records,
        },
    }
