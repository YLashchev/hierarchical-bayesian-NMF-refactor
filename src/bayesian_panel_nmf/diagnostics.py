"""MCMC convergence diagnostics.

The single job here is the numeric convergence gate — rank-normalized R-hat,
bulk/tail ESS, and divergence count — computed from an ArviZ InferenceData /
DataTree. Sampling lives in inference.py; trace *plotting* lives with the
figures layer, not here.
"""

from typing import Any

import numpy as np

# Gate thresholds (single source of truth; see convergence_summary).
_RHAT_MAX = 1.01
_ESS_BULK_MIN = 400
_ESS_BULK_FAIL = 100


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
        result["rhat_max"] < _RHAT_MAX
        and result["ess_bulk_min"] > _ESS_BULK_MIN
        and divergences == 0
    )
    if params is not None:
        result["gate_params"] = list(params)
    return result


def parameter_diagnostics(idata, params: list[str] | None = None) -> list[dict]:
    """Per-parameter R-hat/ESS rows, worst first; constant sites -> 'fixed'.

    A site is 'fixed' iff it has zero variance across (chain, draw) in THIS
    idata -- empirical, so ``disp`` under ``sample_disp: false`` and cut
    Stage-2's ``mu_ctrl`` are labeled without hard-coded model knowledge.
    Fixed sites get no R-hat/ESS (undefined) and never count as FAIL.
    ``params`` prefix-filters the same way as the convergence gate.

    Display helper for the run-summary panel and ``bpnmf traces``; the pass/
    fail gate itself is ``convergence_summary``.
    """
    import arviz as az

    names = list(idata.posterior.data_vars)
    if params:
        names = [v for v in names if any(v.startswith(p) for p in params)]
        if not names:
            raise ValueError(
                f"no parameters matched {params} "
                f"(available: {list(idata.posterior.data_vars)})"
            )

    fixed: list[str] = []
    varying: list[str] = []
    for v in names:
        vals = np.asarray(idata.posterior[v].values)
        span = vals.max(axis=(0, 1)) - vals.min(axis=(0, 1))
        (fixed if np.all(span == 0) else varying).append(v)

    rows: list[dict] = []
    if varying:
        # round_to="none": arviz's default 2dp rounding can push 1.005 up to
        # the 1.01 threshold and fake a FAIL.
        stats = az.summary(
            idata, var_names=varying, kind="diagnostics", round_to="none"
        )
        stats["base"] = stats.index.to_series().apply(lambda x: x.split("[")[0])
        for name, group in stats.groupby("base"):
            rhat = float(group["r_hat"].max())
            ess_bulk = float(group["ess_bulk"].min())
            ess_tail = float(group["ess_tail"].min())
            status = "PASS" if rhat < _RHAT_MAX and ess_bulk > _ESS_BULK_MIN else "WARN"
            if rhat >= _RHAT_MAX or ess_bulk < _ESS_BULK_FAIL:
                status = "FAIL"
            rows.append(
                {
                    "param": name,
                    "rhat": rhat,
                    "ess_bulk": ess_bulk,
                    "ess_tail": ess_tail,
                    "status": status,
                }
            )
    rows.sort(key=lambda r: r["rhat"], reverse=True)
    for v in sorted(fixed):
        rows.append(
            {"param": v, "rhat": None, "ess_bulk": None, "ess_tail": None,
             "status": "fixed"}
        )
    return rows
