"""
MCMC sampling utilities for Bayesian inference.

This module provides a clean interface for running MCMC sampling
on the panel NMF model.
"""

import numpy as np
from jax import random
from numpyro.infer import MCMC, NUTS
import numpyro
from typing import Dict, Optional

from ..models.panel_nmf_model import model


def run_mcmc_inference(
    data_dict: Dict[str, np.ndarray],
    rank: int = 10,
    outcome_dist: str = "NB",
    nb_disp: float = 1e-4,
    sample_disp: bool = False,
    adjust_for_missingness: bool = True,
    model_treated: bool = True,
    num_chains: int = 4,
    num_warmup: int = 1000,
    num_samples: int = 2500,
    thinning: int = 10,
    random_seed: int = 8675309,
    progress_bar: bool = True
) -> MCMC:
    """
    Run MCMC inference on the panel NMF model.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing model data with keys:
        - Y: outcome array (K x D x N)
        - denominators: population array (K x D x N)
        - control_idx_array: boolean control array (K x D x N)
        - missing_idx_array: boolean missing data array (K x D x N)
    rank : int, default=10
        Rank for matrix factorization
    outcome_dist : str, default="NB"
        Distribution for outcomes ("NB" or "Poisson")
    nb_disp : float, default=1e-4
        Negative binomial dispersion parameter
    sample_disp : bool, default=False
        Whether to sample dispersion parameter
    adjust_for_missingness : bool, default=True
        Whether to adjust for missing data
    model_treated : bool, default=True
        Whether to model treatment effects
    num_chains : int, default=4
        Number of MCMC chains
    num_warmup : int, default=1000
        Number of warmup iterations
    num_samples : int, default=2500
        Number of sampling iterations
    thinning : int, default=10
        Thinning interval
    random_seed : int, default=8675309
        Random seed for reproducibility
    progress_bar : bool, default=True
        Whether to show progress bar
        
    Returns
    -------
    MCMC
        Fitted MCMC object containing posterior samples
    """
    # Set host device count for parallelization
    numpyro.set_host_device_count(num_chains)
    
    # Set random seed
    rng_key = random.PRNGKey(random_seed)
    rng_key, rng_key_ = random.split(rng_key)
    
    # Setup the sampler
    kernel = NUTS(model)
    
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
        thinning=thinning
    )
    
    # Run MCMC
    print(f"\nRunning MCMC inference:")
    print(f"  - Rank: {rank}")
    print(f"  - Distribution: {outcome_dist}")
    print(f"  - Chains: {num_chains}")
    print(f"  - Warmup: {num_warmup}, Samples: {num_samples}, Thinning: {thinning}")
    print(f"  - Data shape: {data_dict['Y'].shape}")
    
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
    
    # Print diagnostics
    mcmc.print_summary()
    
    return mcmc


def get_posterior_samples(mcmc: MCMC, group_by_chain: bool = True) -> Dict:
    """
    Extract posterior samples from fitted MCMC object.
    
    Parameters
    ----------
    mcmc : MCMC
        Fitted MCMC object
    group_by_chain : bool, default=True
        Whether to group samples by chain
        
    Returns
    -------
    dict
        Dictionary of posterior samples
    """
    return mcmc.get_samples(group_by_chain=group_by_chain)


def generate_predictions(
    mcmc: MCMC,
    data_dict: Dict[str, np.ndarray],
    rank: int,
    outcome_dist: str = "NB",
    nb_disp: float = 1e-4,
    sample_disp: bool = False,
    random_seed: int = 8675309
) -> np.ndarray:
    """
    Generate posterior predictive samples (counterfactual predictions).
    
    Parameters
    ----------
    mcmc : MCMC
        Fitted MCMC object
    data_dict : dict
        Dictionary containing model data
    rank : int
        Rank used in model fitting
    outcome_dist : str, default="NB"
        Distribution for outcomes
    nb_disp : float, default=1e-4
        Negative binomial dispersion parameter
    sample_disp : bool, default=False
        Whether dispersion was sampled
    random_seed : int, default=8675309
        Random seed
        
    Returns
    -------
    np.ndarray
        Posterior predictive samples with shape (num_chains, num_samples, K, D, N)
    """
    from numpyro.infer import Predictive
    
    rng_key = random.PRNGKey(random_seed + 1)
    rng_key, rng_key_ = random.split(rng_key)
    
    predictive = Predictive(model, mcmc.get_samples(group_by_chain=False))
    
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
    
    # Reshape predictions
    K, D, N = data_dict['denominators'].shape
    num_total_samples = mcmc.num_chains * (mcmc.num_samples // mcmc.thinning)
    pred_mat = predictions.reshape(mcmc.num_chains, num_total_samples // mcmc.num_chains, K, D, N)
    
    return pred_mat
