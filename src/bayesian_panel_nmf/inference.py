"""
Simplified MCMC inference module for bayesian_panel_nmf.

This module provides MCMC sampling functions that extract all settings
from a config dict, removing the need for numerous individual parameters.
Also includes MCMC diagnostic extraction and convergence checking.
"""

import time
from typing import Dict, Callable, Any

import numpy as np
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import summary
import numpyro
from loguru import logger

from bayesian_panel_nmf.validation import (
    validate_data_dict,
    validate_rank,
    DataError,
)


def extract_diagnostics(mcmc: MCMC) -> Dict[str, Any]:
    """
    Extract MCMC diagnostics including ESS and R-hat.
    
    Parameters
    ----------
    mcmc : MCMC
        Fitted MCMC object
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'summary': dict of parameter summaries (mean, std, ESS, Rhat per parameter)
        - 'n_eff_min': minimum effective sample size across all parameters
        - 'n_eff_mean': mean effective sample size across all parameters
        - 'rhat_max': maximum R-hat across all parameters
        - 'rhat_mean': mean R-hat across all parameters
        - 'divergences': number of divergent transitions
        - 'converged': bool, True if rhat_max < 1.01 and n_eff_min > 400 and no divergences
        - 'num_chains': number of chains
        - 'num_samples': number of samples per chain (post-thinning)
        - 'num_warmup': number of warmup iterations
        - 'thinning': thinning interval
        
    Notes
    -----
    Convergence is assessed using standard thresholds:
    - R-hat < 1.01 indicates chains have mixed well
    - ESS > 400 indicates sufficient effective samples
    - Zero divergent transitions indicates no sampling issues
    
    Examples
    --------
    >>> mcmc = run_mcmc_inference(data_dict, model, rank=10, config=config)
    >>> diagnostics = extract_diagnostics(mcmc)
    >>> print(f"Max R-hat: {diagnostics['rhat_max']:.4f}")
    >>> print(f"Min ESS: {diagnostics['n_eff_min']:.0f}")
    """
    # Get samples grouped by chain for R-hat calculation
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    summary_dict = summary(samples_by_chain)
    
    # Extract ESS and R-hat values across all parameters
    all_ess = []
    all_rhat = []
    param_summaries = {}
    
    for param_name, stats in summary_dict.items():
        # stats contains: mean, std, median, 5%, 95%, n_eff, r_hat
        # Each is an array with shape matching the parameter
        if 'n_eff' in stats:
            ess_values = np.asarray(stats['n_eff']).flatten()
            all_ess.extend(ess_values[~np.isnan(ess_values)])
        
        if 'r_hat' in stats:
            rhat_values = np.asarray(stats['r_hat']).flatten()
            all_rhat.extend(rhat_values[~np.isnan(rhat_values)])
        
        # Store summary for this parameter
        param_summaries[param_name] = {
            'mean': float(np.nanmean(stats.get('mean', np.nan))),
            'std': float(np.nanmean(stats.get('std', np.nan))),
            'n_eff_min': float(np.nanmin(stats.get('n_eff', np.nan))),
            'n_eff_mean': float(np.nanmean(stats.get('n_eff', np.nan))),
            'r_hat_max': float(np.nanmax(stats.get('r_hat', np.nan))),
            'r_hat_mean': float(np.nanmean(stats.get('r_hat', np.nan))),
        }
    
    # Compute global diagnostics
    n_eff_min = float(np.min(all_ess)) if all_ess else np.nan
    n_eff_mean = float(np.mean(all_ess)) if all_ess else np.nan
    rhat_max = float(np.max(all_rhat)) if all_rhat else np.nan
    rhat_mean = float(np.mean(all_rhat)) if all_rhat else np.nan
    
    # Count divergent transitions
    # MCMC stores extra fields including diverging transitions
    extra_fields = mcmc.get_extra_fields()
    divergences = 0
    if 'diverging' in extra_fields:
        divergences = int(np.sum(extra_fields['diverging']))
    
    # Determine convergence
    rhat_ok = rhat_max < 1.01 if not np.isnan(rhat_max) else False
    ess_ok = n_eff_min > 400 if not np.isnan(n_eff_min) else False
    no_divergences = divergences == 0
    converged = rhat_ok and ess_ok and no_divergences
    
    return {
        'summary': param_summaries,
        'n_eff_min': n_eff_min,
        'n_eff_mean': n_eff_mean,
        'rhat_max': rhat_max,
        'rhat_mean': rhat_mean,
        'divergences': divergences,
        'converged': converged,
        'num_chains': mcmc.num_chains,
        'num_samples': mcmc.num_samples,
        'num_warmup': mcmc.num_warmup,
        'thinning': mcmc.thinning,
    }


def check_convergence(diagnostics: Dict[str, Any]) -> bool:
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
    rhat_max = diagnostics['rhat_max']
    n_eff_min = diagnostics['n_eff_min']
    divergences = diagnostics['divergences']
    
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
        total_samples = diagnostics['num_chains'] * diagnostics['num_samples']
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
    data_dict: Dict[str, np.ndarray],
    model_fn: Callable,
    rank: int,
    config: dict
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
    # =========================================================================
    # Input validation (minimal - trust users)
    # =========================================================================
    validate_data_dict(data_dict)
    rank = validate_rank(rank)
    
    K, D, N = data_dict['Y'].shape
    logger.debug(f"Running inference: K={K}, D={D}, N={N}, rank={rank}")
    
    # Extract MCMC settings
    mcmc_config = config.get('mcmc', {})
    num_chains = mcmc_config.get('num_chains', 4)
    num_warmup = mcmc_config.get('num_warmup', 1000)
    num_samples = mcmc_config.get('num_samples', 2500)
    thinning = mcmc_config.get('thinning', 10)
    random_seed = mcmc_config.get('random_seed', 8675309)
    progress_bar = mcmc_config.get('progress_bar', True)
    
    # Extract model settings
    model_config = config.get('model', {})
    outcome_dist = model_config.get('outcome_distribution', 'NB')
    nb_disp = model_config.get('nb_disp', 1e-4)
    sample_disp = model_config.get('sample_disp', False)
    adjust_for_missingness = model_config.get('adjust_for_missingness', True)
    model_treated = model_config.get('model_treated', True)
    
    # Set host device count for chain parallelization
    numpyro.set_host_device_count(num_chains)
    
    # Create PRNG key
    rng_key = random.PRNGKey(random_seed)
    rng_key, rng_key_ = random.split(rng_key)
    
    # Setup NUTS kernel and MCMC
    kernel = NUTS(model_fn)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
        thinning=thinning
    )
    
    # Log run configuration
    logger.info("Starting MCMC inference")
    logger.info(f"  Rank: {rank}")
    logger.info(f"  Distribution: {outcome_dist}")
    logger.info(f"  Chains: {num_chains}")
    logger.info(f"  Warmup: {num_warmup}, Samples: {num_samples}, Thinning: {thinning}")
    logger.debug(f"  Data shape: {data_dict['Y'].shape}")
    logger.debug(f"  Model options: adjust_missingness={adjust_for_missingness}, "
                 f"model_treated={model_treated}, nb_disp={nb_disp}, sample_disp={sample_disp}")
    
    # Run MCMC with timing
    start_time = time.time()
    mcmc.run(
        rng_key_,
        y=data_dict['Y'],
        denominators=data_dict['denominators'],
        control_idx_array=data_dict['control_idx_array'],
        missing_idx_array=data_dict['missing_idx_array'],
        rank=rank,
        outcome_dist=outcome_dist,
        adjust_for_missingness=adjust_for_missingness,
        nb_disp=nb_disp,
        sample_disp=sample_disp,
        model_treated=model_treated
    )
    elapsed = time.time() - start_time
    logger.info(f"MCMC completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # Print NumPyro's built-in summary
    mcmc.print_summary()
    
    # Extract and log diagnostics
    diagnostics = extract_diagnostics(mcmc)
    
    logger.info(
        f"Diagnostics: min_ESS={diagnostics['n_eff_min']:.0f}, "
        f"max_Rhat={diagnostics['rhat_max']:.4f}, "
        f"divergences={diagnostics['divergences']}"
    )
    
    # Check convergence and log warnings
    if not check_convergence(diagnostics):
        logger.warning(
            "MCMC may not have converged. Consider increasing num_samples or num_warmup."
        )
    else:
        logger.success("All convergence checks passed.")
    
    return mcmc


def generate_predictions(
    mcmc: MCMC,
    data_dict: Dict[str, np.ndarray],
    model_fn: Callable,
    rank: int,
    config: dict
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
    # =========================================================================
    # Input validation
    # =========================================================================
    # Minimal validation
    rank = validate_rank(rank)
    
    if 'denominators' not in data_dict:
        raise DataError("data_dict missing 'denominators'")
    
    # Extract settings from config
    mcmc_config = config.get('mcmc', {})
    random_seed = mcmc_config.get('random_seed', 8675309)
    
    model_config = config.get('model', {})
    outcome_dist = model_config.get('outcome_distribution', 'NB')
    nb_disp = model_config.get('nb_disp', 1e-4)
    sample_disp = model_config.get('sample_disp', False)
    
    # Create PRNG key (offset from inference key)
    rng_key = random.PRNGKey(random_seed + 1)
    rng_key, rng_key_ = random.split(rng_key)
    
    logger.info("Generating posterior predictive samples")
    logger.debug(f"  Using {outcome_dist} distribution with nb_disp={nb_disp}")
    
    # Create Predictive object with posterior samples
    predictive = Predictive(model_fn, mcmc.get_samples(group_by_chain=False))
    
    # Generate predictions with control_idx_array=None for counterfactual
    start_time = time.time()
    predictions = predictive(
        rng_key_,
        denominators=data_dict['denominators'],
        control_idx_array=None,  # Generate predictions for all periods
        missing_idx_array=None,
        rank=rank,
        outcome_dist=outcome_dist,
        nb_disp=nb_disp,
        sample_disp=sample_disp,
        model_treated=False  # Counterfactual without treatment
    )['y_obs']
    elapsed = time.time() - start_time
    logger.debug(f"  Predictions generated in {elapsed:.1f}s")
    
    # Reshape predictions to (chains, samples, K, D, N)
    K, D, N = data_dict['denominators'].shape
    num_total_samples = mcmc.num_chains * (mcmc.num_samples // mcmc.thinning)
    samples_per_chain = num_total_samples // mcmc.num_chains
    
    pred_mat = predictions.reshape(mcmc.num_chains, samples_per_chain, K, D, N)
    logger.debug(f"  Prediction shape: {pred_mat.shape}")
    
    return pred_mat
