"""MCMC sampling and posterior prediction.

All settings read from a typed ``Config`` object (see ``config.py``) via
attribute access rather than an untyped dict. The convergence gate lives in
``diagnostics.py``.
"""

import time
from collections.abc import Callable

import jax
import numpy as np
import numpyro.infer.initialization  # noqa: F401 — binds attr arviz's from_numpyro needs
from jax import block_until_ready, random
from loguru import logger
from numpyro.infer import MCMC, NUTS, Predictive

from bayesian_panel_nmf.checks import validate_data_dict, validate_rank
from bayesian_panel_nmf.config import Config
from bayesian_panel_nmf.parallelism import choose_mcmc_parallelism
from bayesian_panel_nmf.validation import DataError


def run_mcmc_inference(
    data_dict: dict[str, np.ndarray], model_fn: Callable, rank: int, config: Config
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
    config : Config
        Typed configuration (see ``config.py``). Reads ``config.model``
        (outcome_distribution, nb_disp, sample_disp, adjust_for_missingness,
        model_treated) and ``config.mcmc`` (auto_parallelism, max_chains,
        num_chains, chain_method, num_warmup, num_samples, thinning,
        random_seed, progress_bar).

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

    mcmc_cfg = config.mcmc
    auto_parallelism = mcmc_cfg.auto_parallelism
    if auto_parallelism:
        max_chains = mcmc_cfg.max_chains
        num_chains, chain_method = choose_mcmc_parallelism(max_chains=max_chains)
        logger.info(
            f"auto_parallelism: max_chains={max_chains} -> "
            f"num_chains={num_chains}, chain_method={chain_method!r} "
            f"(jax.local_device_count()={jax.local_device_count()})"
        )
    else:
        num_chains = mcmc_cfg.num_chains if mcmc_cfg.num_chains is not None else 4
        chain_method = (
            mcmc_cfg.chain_method if mcmc_cfg.chain_method is not None else "sequential"
        )
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
    num_warmup = mcmc_cfg.num_warmup
    num_samples = mcmc_cfg.num_samples
    thinning = mcmc_cfg.thinning
    random_seed = mcmc_cfg.random_seed
    progress_bar = mcmc_cfg.progress_bar

    model_cfg = config.model
    outcome_dist = model_cfg.outcome_distribution
    nb_disp = model_cfg.nb_disp
    sample_disp = model_cfg.sample_disp
    adjust_for_missingness = model_cfg.adjust_for_missingness
    model_treated = model_cfg.model_treated

    rng_key = random.PRNGKey(random_seed)
    rng_key, rng_key_ = random.split(rng_key)

    kernel = NUTS(model_fn, target_accept_prob=mcmc_cfg.target_accept)
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
    config: Config,
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
    config : Config
        Typed configuration. Reads ``config.model`` (outcome_distribution,
        nb_disp, sample_disp) and ``config.mcmc.random_seed``.

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

    random_seed = config.mcmc.random_seed

    outcome_dist = config.model.outcome_distribution
    nb_disp = config.model.nb_disp
    sample_disp = config.model.sample_disp

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
