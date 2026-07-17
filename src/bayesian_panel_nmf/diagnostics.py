"""MCMC convergence diagnostics.

The single job here is the numeric convergence gate — rank-normalized R-hat,
bulk/tail ESS, and divergence count — computed from an ArviZ InferenceData /
DataTree. Sampling lives in inference.py; trace *plotting* lives with the
figures layer, not here.
"""

from typing import Any

import numpy as np


def convergence_summary(idata, params: list[str] | None = None) -> dict[str, Any]:
    """Rank-normalized R-hat / bulk+tail ESS gate (Vehtari et al. 2021 via ArviZ).

    Thresholds: R-hat < 1.01, bulk ESS > 400, zero divergences.

    ``params`` restricts which posterior variables feed R-hat/ESS (prefix
    match on the variable name, so "mu" covers "mu" and scoped variants).
    Use it to exclude non-identifiable sites (fixed effects, unit_weight)
    whose R-hat legitimately fails while the quantities of interest mix.
    Divergences are ALWAYS counted over the full run regardless of ``params``.
    Default None = every parameter (the historical gate).
    """
    import arviz as az

    if params is not None:
        all_vars = list(idata.posterior.data_vars)
        keep = [v for v in all_vars if any(v.startswith(p) for p in params)]
        if not keep:
            raise ValueError(
                f"gate_params matched no posterior variables: {params} "
                f"(available: {all_vars})"
            )
        stats = az.summary(
            idata, var_names=keep, kind="diagnostics", round_to="none"
        )
    else:
        stats = az.summary(idata, kind="diagnostics", round_to="none")
    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())

    result: dict[str, Any] = {
        "rhat_max": float(stats["r_hat"].max()),
        "ess_bulk_min": float(stats["ess_bulk"].min()),
        "ess_tail_min": float(stats["ess_tail"].min()),
        "divergences": divergences,
    }
    result["converged"] = bool(
        result["rhat_max"] < 1.01 and result["ess_bulk_min"] > 400 and divergences == 0
    )
    if params is not None:
        result["gate_params"] = list(params)
    return result
