"""MCMC convergence diagnostics.

The single job here is the numeric convergence gate — rank-normalized R-hat,
bulk/tail ESS, and divergence count — computed from an ArviZ InferenceData /
DataTree. Sampling lives in inference.py; trace *plotting* lives with the
figures layer, not here.
"""

from typing import Any

import numpy as np


def convergence_summary(idata) -> dict[str, Any]:
    """Rank-normalized R-hat / bulk+tail ESS gate (Vehtari et al. 2021 via ArviZ).

    Thresholds: R-hat < 1.01, bulk ESS > 400, zero divergences.
    """
    import arviz as az

    stats = az.summary(idata, kind="diagnostics", round_to="none")
    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())

    result = {
        "rhat_max": float(stats["r_hat"].max()),
        "ess_bulk_min": float(stats["ess_bulk"].min()),
        "ess_tail_min": float(stats["ess_tail"].min()),
        "divergences": divergences,
    }
    result["converged"] = bool(
        result["rhat_max"] < 1.01 and result["ess_bulk_min"] > 400 and divergences == 0
    )
    return result
