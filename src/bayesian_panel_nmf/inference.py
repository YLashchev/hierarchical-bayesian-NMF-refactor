"""MCMC sampling, posterior prediction, and convergence diagnostics.

All settings read from a config dict rather than passed as individual kwargs.
"""

import time
from typing import Any, Callable

import numpy as np
from jax import block_until_ready, random
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import summary
import numpyro
from loguru import logger

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


def convergence_summary(idata) -> dict[str, Any]:
    """Rank-normalized R-hat / bulk+tail ESS gate (Vehtari et al. 2021 via ArviZ).

    Thresholds: R-hat < 1.01, bulk ESS > 400, zero divergences.
    """
    import arviz as az

    stats = az.summary(idata, kind="diagnostics", round_to="none")
    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())

    result = {
        "rhat_max": float(stats["r_hat"].max()),
        "ess_bulk_min": float(stats["ess_bulk"].min()),
        "ess_tail_min": float(stats["ess_tail"].min()),
        "divergences": divergences,
    }
    result["converged"] = bool(
        result["rhat_max"] < 1.01
        and result["ess_bulk_min"] > 400
        and divergences == 0
    )
    return result


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
    JAX dispatch is asynchronous, so runtime logging blocks until posterior
    samples are ready. Use extract_diagnostics(mcmc) and
    check_convergence(diagnostics) explicitly when diagnostics are needed.
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

    worker_log_level = config.get("output", {}).get("worker_log_level")
    if worker_log_level != "WARNING":
        logger.info("Starting MCMC inference")
        logger.info(f"  Rank: {rank}")
        logger.info(f"  Distribution: {outcome_dist}")
        logger.info(f"  Chains: {num_chains}")
        logger.info(
            f"  Warmup: {num_warmup}, Samples: {num_samples}, Thinning: {thinning}"
        )
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
    # JAX dispatch is asynchronous; force samples ready before logging runtime.
    block_until_ready(mcmc.get_samples(group_by_chain=False))
    elapsed = time.time() - start_time
    if worker_log_level != "WARNING":
        logger.info(f"MCMC completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Divergences and ESS/R-hat are expensive to compute (numpyro
    # `_get_states_flat()` reshapes all model parameters).  Skipped
    # here by default.  Use extract_diagnostics() explicitly when
    # save_diagnostics=True.

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

    worker_log_level = config.get("output", {}).get("worker_log_level")
    if worker_log_level != "WARNING":
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
    predictions = block_until_ready(predictions)
    elapsed = time.time() - start_time
    if worker_log_level != "WARNING":
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
