"""MCMC sampling, posterior prediction, and convergence diagnostics.

All settings read from a config dict rather than passed as individual kwargs.
"""

import time
from collections.abc import Callable
from typing import Any

import jax
import numpy as np
import numpyro.infer.initialization  # noqa: F401 — binds attr arviz's from_numpyro needs
from jax import block_until_ready, random
from loguru import logger
from numpyro.infer import MCMC, NUTS, Predictive

from bayesian_panel_nmf.mcmc_utils import choose_mcmc_parallelism
from bayesian_panel_nmf.validation import (
    DataError,
    validate_data_dict,
    validate_rank,
)

# Identifiable deterministic quantities the convergence gate assesses.
# The raw low-rank factors (time_fac, unit_weight) and fixed effects
# (state_fe, time_fe) are NON-identifiable: the factorization
#   log(sum_k exp(time_fac[k] + unit_weight[k]))
# is rotation/permutation-invariant, so R-hat/ESS on the latent factors is
# meaningless and can flag "non-convergence" while the identifiable log-rate
# surface (mu_ctrl, mu) and treatment effect (te) are well mixed. te is only
# present when model_treated=True; mu_ctrl and mu are always present.
IDENTIFIABLE_VARS: tuple[str, ...] = ("mu", "mu_ctrl", "te")


def convergence_summary(
    idata,
    var_names: tuple[str, ...] = IDENTIFIABLE_VARS,
) -> dict[str, Any]:
    """Rank-normalized R-hat / bulk+tail ESS gate (Vehtari et al. 2021 via ArviZ).

    Diagnostics are computed only on identifiable deterministic variables
    (``mu``, ``mu_ctrl``, ``te`` by default) for the reason documented on
    ``IDENTIFIABLE_VARS``. If none of the requested ``var_names`` are present
    in ``idata`` (e.g. a model variant without these sites, or a unit test with
    a placeholder variable), fall back to summarizing every variable so the
    gate still produces a useful diagnostic rather than an empty one.

    Thresholds: R-hat < 1.01, bulk ESS > 400, zero divergences.
    """
    import arviz as az

    stats = az.summary(idata, kind="diagnostics", round_to="none")
    # ArviZ flattens multi-dim sites as "mu[0,0,0]"; split off the base name.
    base_names = stats.index.map(lambda name: name.split("[", 1)[0])
    mask = base_names.isin(var_names)
    if not mask.any():
        logger.warning(
            f"convergence_summary: none of {var_names} present in posterior; "
            "falling back to diagnostics over all variables."
        )
        identifiable_stats = stats
    else:
        identifiable_stats = stats[mask]

    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())

    result = {
        "rhat_max": float(identifiable_stats["r_hat"].max()),
        "ess_bulk_min": float(identifiable_stats["ess_bulk"].min()),
        "ess_tail_min": float(identifiable_stats["ess_tail"].min()),
        "divergences": divergences,
    }
    result["converged"] = bool(
        result["rhat_max"] < 1.01 and result["ess_bulk_min"] > 400 and divergences == 0
    )
    return result


def _resolve_model_settings(config: dict) -> dict:
    """Resolve the outcome-distribution settings shared by
    run_mcmc_inference and generate_predictions, with defaults matching
    the model's own signature defaults (outcome_dist='NB', nb_disp=1e-4,
    sample_disp=False)."""
    model_config = config.get("model", {})
    return {
        "outcome_dist": model_config.get("outcome_distribution", "NB"),
        "nb_disp": model_config.get("nb_disp", 1e-4),
        "sample_disp": model_config.get("sample_disp", False),
    }


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
        - config['mcmc']['auto_parallelism']: pick num_chains/chain_method
          from visible JAX devices via choose_mcmc_parallelism (default True)
        - config['mcmc']['max_chains']: upper bound on chain count, used
          only when auto_parallelism is true
        - config['mcmc']['num_chains']: literal chain count, used only
          when auto_parallelism is false
        - config['mcmc']['chain_method']: literal chain_method, used only
          when auto_parallelism is false (defaults to "sequential")
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
    samples are ready. Use convergence_summary(idata) explicitly when
    diagnostics are needed.
    """
    validate_data_dict(data_dict)
    rank = validate_rank(rank)

    mcmc_config = config.get("mcmc", {})
    auto_parallelism = mcmc_config.get("auto_parallelism", True)
    if auto_parallelism:
        max_chains = mcmc_config.get("max_chains", 4)
        num_chains, chain_method = choose_mcmc_parallelism(max_chains=max_chains)
        logger.info(
            f"auto_parallelism: max_chains={max_chains} -> "
            f"num_chains={num_chains}, chain_method={chain_method!r} "
            f"(jax.local_device_count()={jax.local_device_count()})"
        )
    else:
        num_chains = mcmc_config.get("num_chains", 4)
        chain_method = mcmc_config.get("chain_method", "sequential")
        logger.info(
            f"auto_parallelism=false -> num_chains={num_chains}, "
            f"chain_method={chain_method!r}"
        )
    # Guard: a literal 'parallel' on a 1-device host silently runs sequential.
    if chain_method == "parallel" and jax.local_device_count() < 2:
        logger.warning(
            "chain_method='parallel' requested but "
            f"jax.local_device_count()={jax.local_device_count()}; "
            "NumPyro will silently fall back to sequential execution. "
            "Verify numpyro.set_host_device_count() ran before JAX import."
        )
    num_warmup = mcmc_config.get("num_warmup", 1000)
    num_samples = mcmc_config.get("num_samples", 2500)
    thinning = mcmc_config.get("thinning", 10)
    random_seed = mcmc_config.get("random_seed", 8675309)
    progress_bar = mcmc_config.get("progress_bar", True)

    model_settings = _resolve_model_settings(config)
    outcome_dist = model_settings["outcome_dist"]
    nb_disp = model_settings["nb_disp"]
    sample_disp = model_settings["sample_disp"]

    model_config = config.get("model", {})
    adjust_for_missingness = model_config.get("adjust_for_missingness", True)
    model_treated = model_config.get("model_treated", True)

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
        chain_method=chain_method,
    )

    start_time = time.time()
    mcmc.run(
        rng_key_,
        extra_fields=("diverging",),
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
    logger.info(f"MCMC completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

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

    model_settings = _resolve_model_settings(config)
    outcome_dist = model_settings["outcome_dist"]
    nb_disp = model_settings["nb_disp"]
    sample_disp = model_settings["sample_disp"]

    # Offset seed from inference key so posterior predictive uses a different stream
    rng_key = random.PRNGKey(random_seed + 1)
    rng_key, rng_key_ = random.split(rng_key)

    predictive = Predictive(model_fn, mcmc.get_samples(group_by_chain=False))

    # control_idx_array=None + model_treated=False => counterfactual (untreated) predictions
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

    return pred_mat
