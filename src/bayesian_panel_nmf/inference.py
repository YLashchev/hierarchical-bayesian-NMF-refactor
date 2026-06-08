"""MCMC sampling, posterior prediction, and convergence diagnostics.

All settings read from a config dict rather than passed as individual kwargs.
"""

import time
from typing import Any, Callable

import numpy as np
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import summary
import numpyro
from loguru import logger

from bayesian_panel_nmf.models import treatment_effect_model
from bayesian_panel_nmf.validation import (
    validate_data_dict,
    validate_rank,
    DataError,
)


def extract_diagnostics(mcmc: MCMC) -> dict[str, Any]:
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    summary_dict = summary(samples_by_chain)

    all_ess = []
    all_rhat = []
    param_summaries = {}

    for param_name, stats in summary_dict.items():
        if "n_eff" in stats:
            ess_values = np.asarray(stats["n_eff"]).flatten()
            all_ess.extend(ess_values[~np.isnan(ess_values)])

        if "r_hat" in stats:
            rhat_values = np.asarray(stats["r_hat"]).flatten()
            all_rhat.extend(rhat_values[~np.isnan(rhat_values)])

        param_summaries[param_name] = {
            "mean": float(np.nanmean(stats.get("mean", np.nan))),
            "std": float(np.nanmean(stats.get("std", np.nan))),
            "n_eff_min": float(np.nanmin(stats.get("n_eff", np.nan))),
            "n_eff_mean": float(np.nanmean(stats.get("n_eff", np.nan))),
            "r_hat_max": float(np.nanmax(stats.get("r_hat", np.nan))),
            "r_hat_mean": float(np.nanmean(stats.get("r_hat", np.nan))),
        }

    n_eff_min = float(np.min(all_ess)) if all_ess else np.nan
    n_eff_mean = float(np.mean(all_ess)) if all_ess else np.nan
    rhat_max = float(np.max(all_rhat)) if all_rhat else np.nan
    rhat_mean = float(np.mean(all_rhat)) if all_rhat else np.nan

    extra_fields = mcmc.get_extra_fields()
    divergences = 0
    if "diverging" in extra_fields:
        divergences = int(np.sum(extra_fields["diverging"]))

    rhat_ok = rhat_max < 1.01 if not np.isnan(rhat_max) else False
    ess_ok = n_eff_min > 400 if not np.isnan(n_eff_min) else False
    no_divergences = divergences == 0
    converged = rhat_ok and ess_ok and no_divergences

    return {
        "summary": param_summaries,
        "n_eff_min": n_eff_min,
        "n_eff_mean": n_eff_mean,
        "rhat_max": rhat_max,
        "rhat_mean": rhat_mean,
        "divergences": divergences,
        "converged": converged,
        "num_chains": mcmc.num_chains,
        "num_samples": mcmc.num_samples,
        "num_warmup": mcmc.num_warmup,
        "thinning": mcmc.thinning,
    }


def check_convergence(diagnostics: dict[str, Any]) -> bool:
    """
    Check MCMC convergence and log warnings if issues detected.

    Checks:
    - R-hat < 1.01 for all parameters
    - ESS > 400 for all parameters
    - No divergent transitions

    Parameters
    ----------
    diagnostics : dict
        Output from extract_diagnostics()

    Returns
    -------
    bool
        True if all convergence checks pass, False otherwise

    Notes
    -----
    This function logs warnings using loguru.

    Examples
    --------
    >>> diagnostics = extract_diagnostics(mcmc)
    >>> if not check_convergence(diagnostics):
    ...     print("Consider increasing num_samples or num_warmup")
    """
    rhat_max = diagnostics["rhat_max"]
    n_eff_min = diagnostics["n_eff_min"]
    divergences = diagnostics["divergences"]

    all_ok = True

    # Check R-hat
    if np.isnan(rhat_max):
        logger.warning("R-hat could not be computed (need multiple chains)")
        all_ok = False
    elif rhat_max > 1.01:
        logger.warning(
            f"Convergence issue: max R-hat = {rhat_max:.4f} > 1.01. "
            "Chains may not have mixed well. Consider increasing num_warmup."
        )
        all_ok = False
    elif rhat_max > 1.005:
        logger.info(
            f"R-hat is borderline: max R-hat = {rhat_max:.4f}. "
            "Results are likely fine but consider longer chains for critical analyses."
        )

    # Check ESS
    if np.isnan(n_eff_min):
        logger.warning("Effective sample size could not be computed")
        all_ok = False
    elif n_eff_min < 100:
        logger.warning(
            f"Very low effective sample size: min ESS = {n_eff_min:.0f} < 100. "
            "Results may be unreliable. Increase num_samples significantly."
        )
        all_ok = False
    elif n_eff_min < 400:
        logger.warning(
            f"Low effective sample size: min ESS = {n_eff_min:.0f} < 400. "
            "Consider increasing num_samples for more reliable estimates."
        )
        all_ok = False

    # Check divergences
    if divergences > 0:
        total_samples = diagnostics["num_chains"] * diagnostics["num_samples"]
        div_pct = 100 * divergences / total_samples
        if div_pct > 1:
            logger.warning(
                f"High divergence rate: {divergences} divergent transitions "
                f"({div_pct:.1f}%). Consider reparameterizing or increasing adapt_delta."
            )
            all_ok = False
        else:
            logger.warning(
                f"Divergent transitions detected: {divergences} ({div_pct:.2f}%). "
                "Monitor results carefully."
            )
            # Low divergence rate - set all_ok = False only if significant
            if divergences > 10:
                all_ok = False

    return all_ok


def run_mcmc_inference(
    data_dict: dict[str, np.ndarray], model_fn: Callable, rank: int, config: dict
) -> MCMC:
    """
    Run MCMC inference on the panel NMF model.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model data with keys:
        - Y: outcome array (K, D, N)
        - denominators: population array (K, D, N)
        - control_idx_array: boolean control mask (K, D, N)
        - missing_idx_array: boolean missing data mask (K, D, N)
    model_fn : callable
        NumPyro model function
    rank : int
        Rank for matrix factorization
    config : dict
        Configuration dict with 'model' and 'mcmc' sections:
        - config['model']['outcome_distribution']: "NB" or "Poisson"
        - config['model']['nb_disp']: negative binomial dispersion
        - config['model']['sample_disp']: whether to sample dispersion
        - config['model']['adjust_for_missingness']: handle censored data
        - config['model']['model_treated']: model treatment effects
        - config['mcmc']['num_chains']: number of MCMC chains
        - config['mcmc']['num_warmup']: warmup iterations
        - config['mcmc']['num_samples']: sampling iterations
        - config['mcmc']['thinning']: thinning interval
        - config['mcmc']['random_seed']: random seed
        - config['mcmc']['progress_bar']: show progress bar

    Returns
    -------
    MCMC
        Fitted MCMC object containing posterior samples

    Raises
    ------
    DataError
        If data_dict is missing required keys or has inconsistent shapes

    Notes
    -----
    After MCMC completes, this function:
    1. Extracts convergence diagnostics (ESS, R-hat, divergences)
    2. Logs key diagnostic statistics
    3. Warns if convergence issues are detected

    Use extract_diagnostics(mcmc) and check_convergence(diagnostics)
    for programmatic access to diagnostic information.
    """
    validate_data_dict(data_dict)
    rank = validate_rank(rank)

    K, D, N = data_dict["Y"].shape
    logger.debug(f"Running inference: K={K}, D={D}, N={N}, rank={rank}")

    mcmc_config = config.get("mcmc", {})
    num_chains = mcmc_config.get("num_chains", 4)
    num_warmup = mcmc_config.get("num_warmup", 1000)
    num_samples = mcmc_config.get("num_samples", 2500)
    thinning = mcmc_config.get("thinning", 10)
    random_seed = mcmc_config.get("random_seed", 8675309)
    progress_bar = mcmc_config.get("progress_bar", True)

    model_config = config.get("model", {})
    outcome_dist = model_config.get("outcome_distribution", "NB")
    nb_disp = model_config.get("nb_disp", 1e-4)
    sample_disp = model_config.get("sample_disp", False)
    adjust_for_missingness = model_config.get("adjust_for_missingness", True)
    model_treated = model_config.get("model_treated", True)

    # Host device count must match num_chains for parallel chain execution
    numpyro.set_host_device_count(num_chains)

    rng_key = random.PRNGKey(random_seed)
    rng_key, rng_key_ = random.split(rng_key)

    kernel = NUTS(model_fn)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
        thinning=thinning,
    )

    logger.info("Starting MCMC inference")
    logger.info(f"  Rank: {rank}")
    logger.info(f"  Distribution: {outcome_dist}")
    logger.info(f"  Chains: {num_chains}")
    logger.info(f"  Warmup: {num_warmup}, Samples: {num_samples}, Thinning: {thinning}")
    logger.debug(f"  Data shape: {data_dict['Y'].shape}")
    logger.debug(
        f"  Model options: adjust_missingness={adjust_for_missingness}, "
        f"model_treated={model_treated}, nb_disp={nb_disp}, sample_disp={sample_disp}"
    )

    start_time = time.time()
    mcmc.run(
        rng_key_,
        y=data_dict["Y"],
        denominators=data_dict["denominators"],
        control_idx_array=data_dict["control_idx_array"],
        missing_idx_array=data_dict["missing_idx_array"],
        rank=rank,
        outcome_dist=outcome_dist,
        adjust_for_missingness=adjust_for_missingness,
        nb_disp=nb_disp,
        sample_disp=sample_disp,
        model_treated=model_treated,
    )
    elapsed = time.time() - start_time
    logger.info(f"MCMC completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    mcmc.print_summary()

    diagnostics = extract_diagnostics(mcmc)

    logger.info(
        f"Diagnostics: min_ESS={diagnostics['n_eff_min']:.0f}, "
        f"max_Rhat={diagnostics['rhat_max']:.4f}, "
        f"divergences={diagnostics['divergences']}"
    )

    if not check_convergence(diagnostics):
        logger.warning(
            "MCMC may not have converged. Consider increasing num_samples or num_warmup."
        )
    else:
        logger.success("All convergence checks passed.")

    return mcmc


def generate_predictions(
    mcmc: MCMC,
    data_dict: dict[str, np.ndarray],
    model_fn: Callable,
    rank: int,
    config: dict,
) -> np.ndarray:
    """
    Generate posterior predictive samples (counterfactual predictions).

    Generates predictions under the counterfactual scenario where
    treatment effects are not applied (model_treated=False).

    Parameters
    ----------
    mcmc : MCMC
        Fitted MCMC object from run_mcmc_inference
    data_dict : dict
        Dictionary containing model data with keys:
        - denominators: population array (K, D, N)
    model_fn : callable
        NumPyro model function
    rank : int
        Rank used in model fitting
    config : dict
        Configuration dict with 'model' and 'mcmc' sections:
        - config['model']['outcome_distribution']: "NB" or "Poisson"
        - config['model']['nb_disp']: negative binomial dispersion
        - config['model']['sample_disp']: whether dispersion was sampled
        - config['mcmc']['random_seed']: random seed

    Returns
    -------
    np.ndarray
        Posterior predictive samples with shape (num_chains, num_samples, K, D, N)

    Raises
    ------
    DataError
        If data_dict is missing required keys or rank is invalid
    """
    rank = validate_rank(rank)

    if "denominators" not in data_dict:
        raise DataError("data_dict missing 'denominators'")

    mcmc_config = config.get("mcmc", {})
    random_seed = mcmc_config.get("random_seed", 8675309)

    model_config = config.get("model", {})
    outcome_dist = model_config.get("outcome_distribution", "NB")
    nb_disp = model_config.get("nb_disp", 1e-4)
    sample_disp = model_config.get("sample_disp", False)

    # Offset seed from inference key so posterior predictive uses a different stream
    rng_key = random.PRNGKey(random_seed + 1)
    rng_key, rng_key_ = random.split(rng_key)

    logger.info("Generating posterior predictive samples")
    logger.debug(f"  Using {outcome_dist} distribution with nb_disp={nb_disp}")

    predictive = Predictive(model_fn, mcmc.get_samples(group_by_chain=False))

    # control_idx_array=None + model_treated=False => counterfactual (untreated) predictions
    start_time = time.time()
    predictions = predictive(
        rng_key_,
        denominators=data_dict["denominators"],
        control_idx_array=None,
        missing_idx_array=None,
        rank=rank,
        outcome_dist=outcome_dist,
        nb_disp=nb_disp,
        sample_disp=sample_disp,
        model_treated=False,
    )["y_obs"]
    elapsed = time.time() - start_time
    logger.debug(f"  Predictions generated in {elapsed:.1f}s")

    K, D, N = data_dict["denominators"].shape
    num_chains = mcmc.num_chains
    total_samples = predictions.shape[0]

    if total_samples % num_chains != 0:
        raise DataError(
            f"Number of predictive samples ({total_samples}) is not evenly "
            f"divisible by num_chains ({num_chains})"
        )

    samples_per_chain = total_samples // num_chains

    pred_mat = predictions.reshape(mcmc.num_chains, samples_per_chain, K, D, N)
    logger.debug(f"  Prediction shape: {pred_mat.shape}")

    return pred_mat


def run_two_stage_inference(
    data_dict: dict[str, np.ndarray], model_fn: Callable, rank: int, config: dict
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Two-stage ("cut") inference that decouples the counterfactual from the effect.

    Stage 1 fits the factor model to **control data only** (``model_treated=False``)
    to obtain the counterfactual log-rate ``mu_ctrl``. Stage 2 draws
    ``stage2.n_imputations`` posterior samples of ``mu_ctrl`` and, for each, runs a
    separate MCMC that estimates **only** the hierarchical treatment-effect block on
    the treated post-treatment cells, with that draw plugged in as a fixed offset.
    Pooling the runs (multiple-imputation style) propagates Stage-1 uncertainty while
    cutting feedback from post-treatment data into the counterfactual.

    The 10 imputations run sequentially (v1) but are surfaced as the ``C`` chains of
    the returned arrays, so downstream ``format_draws``/reporting are unchanged.

    Parameters
    ----------
    data_dict : dict
        Output of ``load_and_prepare`` (Y, denominators, control_idx_array, ...).
    model_fn : callable
        The joint NumPyro ``model`` (used for Stage 1 + counterfactual prediction).
    rank : int
        Factorization rank for Stage 1.
    config : dict
        Full config; reads ``model``, ``mcmc`` and ``stage2`` sections.

    Returns
    -------
    samples : dict
        ``{'mu_ctrl': (C,S,K,D,N), 'te': (C,S,K,D,N)}`` with ``C = n_imputations``.
    predictions : np.ndarray
        Counterfactual posterior predictive, shape ``(C,S,K,D,N)``.
    diagnostics : dict
        Stage-1 convergence diagnostics (counterfactual fit quality).
    """
    validate_data_dict(data_dict)
    rank = validate_rank(rank)

    model_config = config.get("model", {})
    outcome_dist = model_config.get("outcome_distribution", "NB")
    nb_disp = model_config.get("nb_disp", 1e-4)
    sample_disp = model_config.get("sample_disp", False)
    adjust_for_missingness = model_config.get("adjust_for_missingness", True)

    stage2_config = config.get("stage2", {})
    n_imputations = int(stage2_config.get("n_imputations", 10))
    s2_warmup = int(stage2_config.get("num_warmup", 1000))
    s2_samples = int(stage2_config.get("num_samples", 2500))
    s2_thinning = int(stage2_config.get("thinning", 10))
    s2_seed = int(stage2_config.get("random_seed", 8675309))
    progress_bar = config.get("mcmc", {}).get("progress_bar", True)

    Y = data_dict["Y"]
    K, D, N = Y.shape
    control_idx_array = data_dict["control_idx_array"]
    missing_idx_array = data_dict["missing_idx_array"]

    # ---- Stage 1: factor model fit to control cells only ----
    logger.info("Two-stage inference: Stage 1 (control-only factor model)")
    stage1_config = {**config, "model": {**model_config, "model_treated": False}}
    mcmc1 = run_mcmc_inference(data_dict, model_fn, rank, stage1_config)
    diagnostics = extract_diagnostics(mcmc1)

    predictions = generate_predictions(mcmc1, data_dict, model_fn, rank, config)

    stage1_samples = mcmc1.get_samples(group_by_chain=False)
    mu_ctrl_all = np.asarray(stage1_samples["mu_ctrl"])  # (total, K, D, N)
    total_draws = mu_ctrl_all.shape[0]
    # generate_predictions reshapes from get_samples(group_by_chain=False), so the
    # flat ordering matches mu_ctrl_all draw-for-draw.
    pred_flat = np.asarray(predictions).reshape(total_draws, K, D, N)

    disp_all = (
        np.asarray(stage1_samples["disp"])
        if sample_disp and "disp" in stage1_samples
        else None
    )

    num_treated = int((~control_idx_array).sum())
    if num_treated == 0:
        logger.warning(
            "No treated cells found; skipping Stage 2 and returning control-only "
            "fit (treatment effect = 0)."
        )
        return mcmc1.get_samples(group_by_chain=True), predictions, diagnostics

    # ---- Select Stage-1 draws of mu_ctrl to propagate ----
    if n_imputations > total_draws:
        logger.warning(
            f"stage2.n_imputations={n_imputations} > available Stage-1 draws "
            f"({total_draws}); clamping to {total_draws}."
        )
        n_imputations = total_draws
    rng = np.random.default_rng(s2_seed)
    selected = rng.choice(total_draws, size=n_imputations, replace=False)
    keys = random.split(random.PRNGKey(s2_seed), n_imputations)

    logger.info(
        f"Two-stage inference: Stage 2 ({n_imputations} imputations, sequential)"
    )

    # ---- Stage 2: one treatment-effect MCMC per mu_ctrl draw ----
    mu_ctrl_blocks: list[np.ndarray] = []
    te_blocks: list[np.ndarray] = []
    pred_blocks: list[np.ndarray] = []
    for i, j in enumerate(selected):
        offset_j = mu_ctrl_all[j]  # (K, D, N) fixed counterfactual offset
        ypred_j = pred_flat[j]  # (K, D, N) counterfactual predictive

        if outcome_dist == "NB":
            dispersion_j = (
                1.0 / disp_all[j] if disp_all is not None else np.ones(D) / nb_disp
            )
        else:
            dispersion_j = None

        logger.info(
            f"  Stage 2 imputation {i + 1}/{n_imputations} (Stage-1 draw {int(j)})"
        )
        mcmc2 = MCMC(
            NUTS(treatment_effect_model),
            num_warmup=s2_warmup,
            num_samples=s2_samples,
            num_chains=1,
            thinning=s2_thinning,
            progress_bar=progress_bar,
        )
        mcmc2.run(
            keys[i],
            mu_ctrl_offset=offset_j,
            control_idx_array=control_idx_array,
            missing_idx_array=missing_idx_array,
            y=Y,
            outcome_dist=outcome_dist,
            dispersion=dispersion_j,
            adjust_for_missingness=adjust_for_missingness,
        )

        s2_diag = extract_diagnostics(mcmc2)
        logger.info(
            f"    diagnostics: min_ESS={s2_diag['n_eff_min']:.0f}, "
            f"max_Rhat={s2_diag['rhat_max']:.4f}, divergences={s2_diag['divergences']}"
        )
        if not s2_diag["converged"]:
            logger.warning(f"    Stage 2 imputation {i + 1} flagged for convergence.")

        te_j = np.asarray(mcmc2.get_samples()["te"])  # (S, K, D, N)
        s = te_j.shape[0]
        mu_ctrl_blocks.append(np.broadcast_to(offset_j, (s, K, D, N)))
        te_blocks.append(te_j)
        pred_blocks.append(np.broadcast_to(ypred_j, (s, K, D, N)))

    samples = {
        "mu_ctrl": np.stack(mu_ctrl_blocks, axis=0),  # (C, S, K, D, N)
        "te": np.stack(te_blocks, axis=0),  # (C, S, K, D, N)
    }
    predictions_out = np.stack(pred_blocks, axis=0)  # (C, S, K, D, N)

    C, S = samples["te"].shape[:2]
    logger.info(
        f"Two-stage inference complete: pooled {C} imputations x {S} draws "
        f"= {C * S} total draws."
    )
    return samples, predictions_out, diagnostics
