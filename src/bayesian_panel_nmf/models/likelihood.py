import numpyro
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from numpyro import distributions as dist


def missingness_adjustment(
    log_rate, missing_idx, control_idx, missing_values, dist_type, dispersion=None
):
    """Add log-likelihood factors for deterministic censoring of small counts.

    For Poisson/NegBin outcomes where values in ``missing_values`` (e.g. 1..9)
    are suppressed in the data: adds ``log P(y in missing_values)`` on masked
    cells and ``log P(y not in missing_values)`` on observed control cells.

    Parameters
    ----------
    log_rate : jnp.ndarray
        Per-cell log rate (flattened externally).
    missing_idx, control_idx : jnp.ndarray
        Boolean masks, same shape as ``log_rate`` after reshape(-1).
    missing_values : jnp.ndarray
        Integer outcome values assumed censored (e.g. jnp.array([1..9])).
    dist_type : {'Poisson', 'NB'}
    dispersion : jnp.ndarray, optional
        Required when ``dist_type == 'NB'``.
    """
    if dist_type == "Poisson":
        probs = dist.Poisson(jnp.exp(log_rate).reshape(-1)[:, None]).log_prob(
            missing_values[None, :]
        )
    elif dist_type == "NB":
        if dispersion is None:
            raise ValueError("Negative Binomial requires a dispersion parameter")
        probs = dist.NegativeBinomial2(
            jnp.exp(log_rate).reshape(-1)[:, None], dispersion.reshape(-1)[:, None]
        ).log_prob(missing_values[None, :])
    else:
        raise ValueError(f"Unsupported dist_type: {dist_type!r}")

    log_probs_summed = logsumexp(probs, axis=1)

    numpyro.factor("missing_factors", log_probs_summed[missing_idx & control_idx].sum())
    numpyro.factor(
        "nonmissing_factors",
        jnp.log(1 - jnp.exp(log_probs_summed[~missing_idx & control_idx])).sum(),
    )
