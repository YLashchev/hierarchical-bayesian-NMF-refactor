"""
MCMC sampling utilities for bayesian_panel_nmf.

This module provides MCMC sampling with optional parallel execution
for multiple analyses via joblib.
"""

import numpy as np
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro
from typing import Dict, Optional, List, Any, Callable
from joblib import Parallel, delayed


def run_mcmc_inference(
    data_dict: Dict[str, np.ndarray],
    model_fn: Callable,
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
    model_fn : callable
        NumPyro model function
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
    # Set host device count for chain parallelization
    numpyro.set_host_device_count(num_chains)
    
    # Set random seed
    rng_key = random.PRNGKey(random_seed)
    rng_key, rng_key_ = random.split(rng_key)
    
    # Setup the sampler
    kernel = NUTS(model_fn)
    
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


def generate_predictions(
    mcmc: MCMC,
    model_fn: Callable,
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
    model_fn : callable
        NumPyro model function
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
    rng_key = random.PRNGKey(random_seed + 1)
    rng_key, rng_key_ = random.split(rng_key)
    
    predictive = Predictive(model_fn, mcmc.get_samples(group_by_chain=False))
    
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


def run_single_analysis(
    analysis_config: Dict[str, Any],
    data_dict: Dict[str, np.ndarray],
    model_fn: Callable,
    mcmc_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a single analysis (one model type + rank combination).
    
    Used internally by run_parallel_analyses.
    
    Parameters
    ----------
    analysis_config : dict
        Configuration for this analysis:
        - rank: int
        - model_type: str
        - groups: list
    data_dict : dict
        Model data
    model_fn : callable
        Model function
    mcmc_config : dict
        MCMC settings
        
    Returns
    -------
    dict
        Results including mcmc object and predictions
    """
    rank = analysis_config['rank']
    model_type = analysis_config.get('model_type', 'default')
    
    print(f"\n{'='*60}")
    print(f"Running: {model_type} with rank {rank}")
    print(f"{'='*60}")
    
    # Run MCMC
    mcmc = run_mcmc_inference(
        data_dict=data_dict,
        model_fn=model_fn,
        rank=rank,
        **mcmc_config
    )
    
    # Generate predictions
    predictions = generate_predictions(
        mcmc=mcmc,
        model_fn=model_fn,
        data_dict=data_dict,
        rank=rank,
        outcome_dist=mcmc_config.get('outcome_dist', 'NB'),
        nb_disp=mcmc_config.get('nb_disp', 1e-4),
        sample_disp=mcmc_config.get('sample_disp', False),
        random_seed=mcmc_config.get('random_seed', 8675309)
    )
    
    return {
        'mcmc': mcmc,
        'predictions': predictions,
        'config': analysis_config
    }


def run_parallel_analyses(
    analysis_configs: List[Dict[str, Any]],
    data_dicts: List[Dict[str, np.ndarray]],
    model_fn: Callable,
    mcmc_config: Dict[str, Any],
    num_workers: int = 1
) -> List[Dict[str, Any]]:
    """
    Run multiple analyses in parallel using joblib.
    
    Parameters
    ----------
    analysis_configs : list of dict
        Each dict specifies a single analysis (type, rank, etc.)
    data_dicts : list of dict
        Corresponding data dictionaries (same length as analysis_configs)
    model_fn : callable
        Model function
    mcmc_config : dict
        MCMC settings (common to all analyses)
    num_workers : int, default=1
        Number of parallel workers
        - 1: Sequential execution
        - -1: Use all available cores
        - N: Use N workers
        
    Returns
    -------
    list of dict
        Results for each analysis
    """
    if num_workers == 1:
        # Sequential execution
        print(f"Running {len(analysis_configs)} analyses sequentially...")
        results = []
        for config, data in zip(analysis_configs, data_dicts):
            result = run_single_analysis(config, data, model_fn, mcmc_config)
            results.append(result)
    else:
        # Parallel execution via joblib
        print(f"Running {len(analysis_configs)} analyses in parallel "
              f"with {num_workers} workers...")
        results = Parallel(n_jobs=num_workers)(
            delayed(run_single_analysis)(config, data, model_fn, mcmc_config)
            for config, data in zip(analysis_configs, data_dicts)
        )
    
    return results


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
