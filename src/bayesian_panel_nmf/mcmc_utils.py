"""Device-aware MCMC chain parallelism selection.

Inspects the visible JAX devices and picks (num_chains, chain_method)
following NumPyro's own documented chain_method semantics. This module
has zero side effects -- it never calls numpyro.set_host_device_count()
or otherwise mutates JAX/XLA state. That call, when needed, is the
caller's responsibility and must run before any jax/numpyro import (see
scripts/run_analysis.py's module-level call and comment for why).
"""

import jax


def choose_mcmc_parallelism(max_chains: int = 4) -> tuple[int, str]:
    """Pick (num_chains, chain_method) from the visible JAX devices.

    Rules (matching NumPyro's documented chain_method semantics):

    - CPU, 1 device (the common case: a single-CPU host with no prior
      numpyro.set_host_device_count() call exposing more logical
      devices): ``max_chains`` chains, ``chain_method="sequential"``.
      Chains run one after another but remain independent draws usable
      for R-hat/ESS diagnostics. XLA's CPU backend shares one
      client-level thread pool across all logical devices regardless of
      how many are exposed, so forcing multiple logical CPU devices via
      set_host_device_count for chain_method="parallel" does not give
      clean N-way speedup -- sequential is the honest default here.
    - CPU, >1 device (a host where the CALLER already exposed multiple
      logical CPU devices, e.g. via numpyro.set_host_device_count()
      before this function runs -- this function never calls it itself):
      ``min(max_chains, n_devices)`` chains, ``chain_method="parallel"``.
    - GPU, 1 device: ``max_chains`` chains, ``chain_method="vectorized"``
      (vmap batches chains on the single accelerator; NumPyro marks this
      experimental but it is the documented fit for one GPU).
    - GPU or TPU, >1 device: ``min(max_chains, n_devices)`` chains,
      ``chain_method="parallel"``.
    - Any other/unrecognized platform: ``max_chains`` chains,
      ``chain_method="sequential"`` (safe fallback).

    Parameters
    ----------
    max_chains : int
        Upper bound on the number of chains. Used directly as the chain
        count on single-device hosts; capped by the visible device count
        on multi-device hosts.

    Returns
    -------
    tuple[int, str]
        ``(num_chains, chain_method)`` ready to pass to
        ``numpyro.infer.MCMC(..., num_chains=num_chains, chain_method=chain_method)``.
    """
    devices = jax.devices()
    platform = devices[0].platform if devices else "cpu"
    n_devices = jax.local_device_count()

    if platform == "cpu":
        if n_devices <= 1:
            return max_chains, "sequential"
        return min(max_chains, n_devices), "parallel"

    if platform == "gpu":
        if n_devices <= 1:
            return max_chains, "vectorized"
        return min(max_chains, n_devices), "parallel"

    if platform == "tpu":
        return min(max_chains, n_devices), "parallel"

    return max_chains, "sequential"
