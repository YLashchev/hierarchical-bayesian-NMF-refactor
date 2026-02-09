"""
Posterior processing utilities.

This module provides:
- merge_draws_and_data: R-style merge of posterior draws with observed data
- dict_to_tidybayes: convert posterior arrays to a tidy draws DataFrame
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def merge_draws_and_data(
    samples: Dict[str, np.ndarray],
    predictions: np.ndarray,
    df_long: pd.DataFrame,
    sub_groups: List[str],
    add_total_category: bool = False,
    add_ban_states: bool = True,
) -> pd.DataFrame:
    """
    Replicate the behavior of merge_draws_and_data() from dobbs_fertility/plot_utilities.R.

    Inputs
    ------
    samples : dict
        Posterior samples from NumPyro with keys like 'mu_ctrl', optional 'te', (shapes: chains x iters x K x D x N).
        - Expected untreated baseline:  samples['mu_ctrl']
        - Optional treatment effect:    samples['te']
    predictions : np.ndarray
        Posterior predictive (counterfactual) draws with shape (num_chains, num_samples, K, D, N)
        produced by generate_predictions(..., model_treated=False).
        We will name this column 'ypred'.
    df_long : pd.DataFrame
        Long-format data after preprocessing (and bimonthly aggregation if used) with columns:
        ['state','group','time','births','population','exposure_code','banned_state', 'start_date','end_date'].
        - 'group' values must match sub_groups order used in modeling
    sub_groups : list[str]
        The list of group names (K dimension order used in modeling).
    add_total_category : bool, default False
        If True, add an aggregated "total" category across groups (K) by summing births/ypred and
        log-sum-exp for mu and mu_treated, mirroring the R implementation.
    add_ban_states : bool, default True
        If True, add a synthetic "Ban States" state by aggregating over states with banned_state==1.

    Returns
    -------
    pd.DataFrame
        A merged long DataFrame of posterior draws and observed data, with columns including:
        - .chain, .iteration, .draw (draw identifiers)
        - K (category index), D (state index), N (time index)
        - ypred, mu, mu_treated (on log scale for mu/mu_treated)
        - state, category (group), time, births, population, exposure_code, banned_state, start_date, end_date, etc.

    Notes
    -----
    - mu is the untreated baseline (log-rate) from 'mu_ctrl'.
    - mu_treated = mu_ctrl + te if 'te' present, else mu_treated = mu_ctrl.
    - ypred is a counterfactual draw under no-treatment (model_treated=False).
    - This function constructs integer codes for state/category/time to align with (K,D,N) indexing.
    - "Ban States" rows are appended by summing across banned states for each (K,N,.draw,.chain).
    """
    # Defensive copy
    dat = df_long.copy()

    # Normalize and code dimensions consistent with modeling order
    # Category (K) order should match sub_groups
    cat_to_idx = {g: i for i, g in enumerate(sub_groups)}
    dat = dat[dat["group"].isin(sub_groups)].copy()

    states = sorted(dat["state"].unique())
    times = sorted(dat["time"].unique())
    state_to_idx = {s: i for i, s in enumerate(states)}
    time_to_idx = {t: i for i, t in enumerate(times)}

    dat["K"] = dat["group"].map(cat_to_idx)
    dat["D"] = dat["state"].map(state_to_idx)
    dat["N"] = dat["time"].map(time_to_idx)

    # Ensure required columns exist
    for col in ["births", "population", "exposure_code"]:
        if col not in dat.columns:
            raise ValueError(f"Expected column '{col}' in df_long")

    if "banned_state" not in dat.columns:
        # Allow downstream logic; default to not banned if missing
        dat["banned_state"] = 0

    # Prepare draw id indexing
    C, S, K, D, N = predictions.shape  # chains, samples, K, D, N

    # Utility to flatten (C,S,K,D,N) arrays to a tidy DataFrame
    def flatten_draws(arr: np.ndarray, name: str) -> pd.DataFrame:
        chains, iters = arr.shape[0], arr.shape[1]
        df = pd.DataFrame(
            arr.reshape(chains * iters, K, D, N).reshape(chains * iters * K * D * N),
            index=pd.MultiIndex.from_product(
                [range(chains), range(iters), range(K), range(D), range(N)],
                names=[".chain_idx", ".iter_idx", "K", "D", "N"],
            ),
            columns=[name],
        ).reset_index()
        # 1-based indices for chain/iteration to align with tidybayes conventions
        df[".chain"] = df[".chain_idx"] + 1
        df[".iteration"] = df[".iter_idx"] + 1
        df[".draw"] = (df[".chain"] - 1) * iters + df[".iteration"]
        df = df.drop(columns=[".chain_idx", ".iter_idx"])
        return df

    # Build ypred draws
    ypred_df = flatten_draws(predictions, "ypred")

    # Extract mu (untreated) and te (if present)
    if "mu_ctrl" not in samples:
        raise ValueError("Expected 'mu_ctrl' in samples (untreated baseline log-rate).")
    mu_ctrl = samples["mu_ctrl"]  # (C,S,K,D,N)

    mu_df = flatten_draws(mu_ctrl, "mu")

    if "te" in samples:
        te_df = flatten_draws(samples["te"], "te")
        mu_treated_df = mu_df.merge(te_df, on=[".chain", ".iteration", ".draw", "K", "D", "N"], how="left")
        mu_treated_df["mu_treated"] = mu_treated_df["mu"] + mu_treated_df["te"].fillna(0.0)
        mu_treated_df = mu_treated_df.drop(columns=["te"])
    else:
        mu_treated_df = mu_df.copy()
        mu_treated_df["mu_treated"] = mu_treated_df["mu"]

    # Merge ypred with mu/mu_treated
    draws_df = ypred_df.merge(
        mu_treated_df, on=[".chain", ".iteration", ".draw", "K", "D", "N"], how="left"
    )

    # Optionally add aggregated "total" category across K
    if add_total_category:
        # Aggregate ypred via sum; mu and mu_treated via log-sum-exp across K
        agg = (
            draws_df.groupby([".chain", ".iteration", ".draw", "D", "N"], as_index=False)
            .agg(
                ypred=("ypred", "sum"),
            )
        )

        # Compute log-sum-exp for mu and mu_treated
        # log(sum(exp(x))) = m + log(sum(exp(x - m))) for numerical stability
        def logsumexp_group(x: pd.Series) -> float:
            m = x.max()
            return float(m + np.log(np.exp(x - m).sum()))

        mu_agg = (
            draws_df.groupby([".chain", ".iteration", ".draw", "D", "N"])["mu"]
            .apply(logsumexp_group)
            .reset_index(name="mu")
        )
        mu_treated_agg = (
            draws_df.groupby([".chain", ".iteration", ".draw", "D", "N"])["mu_treated"]
            .apply(logsumexp_group)
            .reset_index(name="mu_treated")
        )
        agg = agg.merge(mu_agg, on=[".chain", ".iteration", ".draw", "D", "N"])
        agg = agg.merge(mu_treated_agg, on=[".chain", ".iteration", ".draw", "D", "N"])

        # Assign a new K index for the aggregated category (max + 1)
        new_K = int(draws_df["K"].max()) + 1 if not draws_df.empty else len(sub_groups)
        agg["K"] = new_K
        draws_df = pd.concat([draws_df, agg], ignore_index=True)

        # Also extend dat with the aggregated category, summing births/pop and exposure as in R
        dat_totals = (
            dat.groupby(["state", "D", "time", "N"], as_index=False)
            .agg(
                births=("births", "sum"),
                population=("population", "sum"),
                exposure_code=("exposure_code", "max"),
                banned_state=("banned_state", "max"),
                start_date=("start_date", "first") if "start_date" in dat.columns else ("time", "first"),
                end_date=("end_date", "first") if "end_date" in dat.columns else ("time", "first"),
            )
        )
        dat_totals["group"] = "total"
        dat_totals["K"] = new_K
        dat = pd.concat([dat, dat_totals], ignore_index=True)

    # Optionally add "Ban States" aggregate over banned states (D indices)
    if add_ban_states:
        banned_D = sorted(dat.loc[dat["banned_state"] == 1, "D"].unique().tolist())
        if len(banned_D) > 0:
            # Aggregate draws over banned states
            ban_draws = (
                draws_df[draws_df["D"].isin(banned_D)]
                .groupby([".chain", ".iteration", ".draw", "K", "N"], as_index=False)
                .agg(
                    ypred=("ypred", "sum"),
                )
            )

            def logsumexp_group(x: pd.Series) -> float:
                m = x.max()
                return float(m + np.log(np.exp(x - m).sum()))

            mu_ban = (
                draws_df[draws_df["D"].isin(banned_D)]
                .groupby([".chain", ".iteration", ".draw", "K", "N"])["mu"]
                .apply(logsumexp_group)
                .reset_index(name="mu")
            )
            mu_treated_ban = (
                draws_df[draws_df["D"].isin(banned_D)]
                .groupby([".chain", ".iteration", ".draw", "K", "N"])["mu_treated"]
                .apply(logsumexp_group)
                .reset_index(name="mu_treated")
            )
            ban_draws = ban_draws.merge(mu_ban, on=[".chain", ".iteration", ".draw", "K", "N"])
            ban_draws = ban_draws.merge(
                mu_treated_ban, on=[".chain", ".iteration", ".draw", "K", "N"]
            )
            # Assign new D index for "Ban States"
            new_D = int(dat["D"].max()) + 1 if not dat.empty else len(states)
            ban_draws["D"] = new_D

            draws_df = pd.concat([draws_df, ban_draws], ignore_index=True)

            # Build corresponding dat rows for "Ban States"
            # For each time and group, sum births/pop, exposure_code=max, banned_state=1
            dat_ban = (
                dat[dat["D"].isin(banned_D)]
                .groupby(
                    [
                        "time",
                        "N",
                        "group",
                        "K",
                    ]
                    + (["start_date", "end_date"] if "start_date" in dat.columns and "end_date" in dat.columns else []),
                    as_index=False,
                )
                .agg(
                    births=("births", "sum"),
                    population=("population", "sum"),
                    exposure_code=("exposure_code", "max"),
                )
            )
            dat_ban["state"] = "Ban States"
            dat_ban["banned_state"] = 1
            dat_ban["D"] = new_D
            dat = pd.concat([dat, dat_ban], ignore_index=True)

    # Final merge: align by K, D, N
    merged = draws_df.merge(
        dat,
        on=["K", "D", "N"],
        how="left",
        suffixes=("", "_obs"),
    )

    # Attach readable labels for categories; ensure presence
    merged["category"] = merged.get("group", None)
    if merged["category"].isnull().any():
        # Fill from K index if needed
        inv_cat = {v: k for k, v in cat_to_idx.items()}
        merged["category"] = merged["K"].map(inv_cat).fillna(merged["category"])

    # Reorder/select common useful columns
    col_order = [
        ".draw",
        ".chain",
        ".iteration",
        "K",
        "D",
        "N",
        "category",
        "state",
        "time",
        "start_date" if "start_date" in merged.columns else None,
        "end_date" if "end_date" in merged.columns else None,
        "banned_state",
        "exposure_code",
        "births",
        "population",
        "ypred",
        "mu",
        "mu_treated",
    ]
    col_order = [c for c in col_order if c is not None and c in merged.columns]
    merged = merged[col_order + [c for c in merged.columns if c not in col_order]]

    return merged


def dict_to_tidybayes(samples_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Convert dictionary of samples to tidybayes-style format.

    Parameters
    ----------
    samples_dict : dict
        Dictionary with parameter names as keys and sample arrays as values
        Arrays should have shape (num_chains, num_samples_per_chain, ...)

    Returns
    -------
    pd.DataFrame
        Tidy dataframe with .chain, .iteration, .draw columns and parameter columns
    """
    if not samples_dict:
        raise ValueError("Empty samples dictionary")

    # Get dimensions from first parameter
    first_param = list(samples_dict.values())[0]
    num_chains, num_samples = first_param.shape[:2]

    # Initialize base dataframe with chain/iteration/draw info
    total_samples = num_chains * num_samples
    df = pd.DataFrame({
        '.chain': np.repeat(np.arange(1, num_chains + 1), num_samples),
        '.iteration': np.tile(np.arange(1, num_samples + 1), num_chains),
        '.draw': np.arange(1, total_samples + 1)
    })

    # Add each parameter
    for param_name, param_array in samples_dict.items():
        if param_array.ndim == 2:
            # Simple parameter: (chains, samples)
            df[param_name] = param_array.flatten()
        elif param_array.ndim > 2:
            # Multi-dimensional parameter
            # Flatten extra dimensions and create separate columns
            extra_shape = param_array.shape[2:]
            param_flat = param_array.reshape(num_chains * num_samples, -1)

            # Create column for each element
            for idx in np.ndindex(extra_shape):
                flat_idx = np.ravel_multi_index(idx, extra_shape)
                col_name = f"{param_name}_{'_'.join(map(str, idx))}"
                df[col_name] = param_flat[:, flat_idx]

    return df
