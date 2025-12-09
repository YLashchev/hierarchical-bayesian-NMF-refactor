"""
Configurable prior distributions for bayesian_panel_nmf.

This module allows researchers to modify priors via YAML configuration
without changing model code.
"""

import numpyro.distributions as dist
from typing import Dict, Any, Optional


# Default prior specifications
DEFAULT_PRIORS = {
    'time_factor': {
        'distribution': 'Gamma',
        'alpha': 20.0,
        'beta': 20.0
    },
    'time_fe': {
        'distribution': 'Gamma',
        'alpha': 1.0,
        'beta': 1.0
    },
    'treatment_it_scale': {
        'distribution': 'HalfNormal',
        'scale': 0.1
    },
    'treatment_state_scale': {
        'distribution': 'HalfNormal',
        'scale': 1.0
    },
    'treatment_category_scale': {
        'distribution': 'HalfNormal',
        'scale': 1.0
    },
    'state_category_scale': {
        'distribution': 'HalfNormal',
        'scale': 1.0
    }
}


def get_distribution(prior_config: Dict[str, Any]) -> dist.Distribution:
    """
    Create a NumPyro distribution from configuration specification.
    
    Parameters
    ----------
    prior_config : dict
        Configuration with 'distribution' key and distribution-specific parameters
        
    Returns
    -------
    numpyro.distributions.Distribution
        The configured distribution
        
    Examples
    --------
    >>> config = {'distribution': 'Gamma', 'alpha': 1.0, 'beta': 1.0}
    >>> get_distribution(config)
    Gamma(concentration=1.0, rate=1.0)
    
    >>> config = {'distribution': 'HalfNormal', 'scale': 0.5}
    >>> get_distribution(config)
    HalfNormal(scale=0.5)
    """
    dist_name = prior_config['distribution']
    
    if dist_name == 'Gamma':
        return dist.Gamma(
            concentration=prior_config['alpha'],
            rate=prior_config['beta']
        )
    elif dist_name == 'HalfNormal':
        return dist.HalfNormal(scale=prior_config['scale'])
    elif dist_name == 'Normal':
        return dist.Normal(
            loc=prior_config.get('loc', 0.0),
            scale=prior_config['scale']
        )
    elif dist_name == 'HalfCauchy':
        return dist.HalfCauchy(scale=prior_config['scale'])
    elif dist_name == 'Exponential':
        return dist.Exponential(rate=prior_config['rate'])
    elif dist_name == 'InverseGamma':
        return dist.InverseGamma(
            concentration=prior_config['alpha'],
            rate=prior_config['beta']
        )
    elif dist_name == 'Uniform':
        return dist.Uniform(
            low=prior_config.get('low', 0.0),
            high=prior_config.get('high', 1.0)
        )
    elif dist_name == 'Beta':
        return dist.Beta(
            concentration1=prior_config['alpha'],
            concentration0=prior_config['beta']
        )
    elif dist_name == 'StudentT':
        return dist.StudentT(
            df=prior_config['df'],
            loc=prior_config.get('loc', 0.0),
            scale=prior_config.get('scale', 1.0)
        )
    elif dist_name == 'Laplace':
        return dist.Laplace(
            loc=prior_config.get('loc', 0.0),
            scale=prior_config['scale']
        )
    else:
        raise ValueError(
            f"Unknown distribution: {dist_name}\n"
            f"Supported: Gamma, HalfNormal, Normal, HalfCauchy, Exponential, "
            f"InverseGamma, Uniform, Beta, StudentT, Laplace"
        )


def load_priors(config: Dict[str, Any]) -> Dict[str, dist.Distribution]:
    """
    Load all priors from configuration, falling back to defaults.
    
    Parameters
    ----------
    config : dict
        Full configuration dictionary. Priors should be in config['priors']
        
    Returns
    -------
    dict
        Dictionary mapping prior names to Distribution objects
        
    Examples
    --------
    >>> config = {'priors': {'time_factor': {'distribution': 'Gamma', 'alpha': 10, 'beta': 10}}}
    >>> priors = load_priors(config)
    >>> priors['time_factor']
    Gamma(concentration=10, rate=10)
    """
    # Start with defaults
    priors_config = DEFAULT_PRIORS.copy()
    
    # Override with user-specified priors
    if 'priors' in config:
        for name, spec in config['priors'].items():
            if name in priors_config:
                # Merge: keep defaults for unspecified params
                priors_config[name] = {**priors_config[name], **spec}
            else:
                # New prior not in defaults
                priors_config[name] = spec
    
    # Create distributions
    return {name: get_distribution(spec) for name, spec in priors_config.items()}


def get_prior_value(
    config: Dict[str, Any],
    prior_name: str,
    param_name: str,
    default: Optional[float] = None
) -> float:
    """
    Get a specific parameter value for a prior.
    
    This is useful when you just need the parameter value, not the full distribution.
    
    Parameters
    ----------
    config : dict
        Full configuration dictionary
    prior_name : str
        Name of the prior (e.g., 'time_factor')
    param_name : str
        Name of the parameter (e.g., 'alpha')
    default : float, optional
        Default value if not found
        
    Returns
    -------
    float
        The parameter value
    """
    # Check user config first
    if 'priors' in config and prior_name in config['priors']:
        if param_name in config['priors'][prior_name]:
            return config['priors'][prior_name][param_name]
    
    # Fall back to defaults
    if prior_name in DEFAULT_PRIORS and param_name in DEFAULT_PRIORS[prior_name]:
        return DEFAULT_PRIORS[prior_name][param_name]
    
    if default is not None:
        return default
    
    raise KeyError(f"Prior '{prior_name}' parameter '{param_name}' not found")


def validate_priors_config(config: Dict[str, Any]) -> None:
    """
    Validate the priors section of a configuration.
    
    Parameters
    ----------
    config : dict
        Configuration to validate
        
    Raises
    ------
    ValueError
        If any prior specification is invalid
    """
    if 'priors' not in config:
        return  # No custom priors, will use defaults
    
    for name, spec in config['priors'].items():
        try:
            # Try to create the distribution to validate
            get_distribution({**DEFAULT_PRIORS.get(name, {}), **spec})
        except Exception as e:
            raise ValueError(f"Invalid prior specification for '{name}': {e}")
