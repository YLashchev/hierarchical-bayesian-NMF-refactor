"""
Simplified MCMC inference module for bayesian_panel_nmf.

This module provides MCMC sampling functions that extract all settings
from a config dict, removing the need for numerous individual parameters.
"""

import numpy as np
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro
from typing import Dict, Callable


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
    """
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
    
    # Print run configuration
    print(f"\nRunning MCMC inference:")
    print(f"  - Rank: {rank}")
    print(f"  - Distribution: {outcome_dist}")
    print(f"  - Chains: {num_chains}")
    print(f"  - Warmup: {num_warmup}, Samples: {num_samples}, Thinning: {thinning}")
    print(f"  - Data shape: {data_dict['Y'].shape}")
    
    # Run MCMC
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
    
    # Print summary diagnostics
    mcmc.print_summary()
    
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
    """
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
    
    # Create Predictive object with posterior samples
    predictive = Predictive(model_fn, mcmc.get_samples(group_by_chain=False))
    
    # Generate predictions with control_idx_array=None for counterfactual
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
    
    # Reshape predictions to (chains, samples, K, D, N)
    K, D, N = data_dict['denominators'].shape
    num_total_samples = mcmc.num_chains * (mcmc.num_samples // mcmc.thinning)
    samples_per_chain = num_total_samples // mcmc.num_chains
    
    pred_mat = predictions.reshape(mcmc.num_chains, samples_per_chain, K, D, N)
    
    return pred_mat
