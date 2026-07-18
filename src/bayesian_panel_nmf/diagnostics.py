"""MCMC convergence diagnostics.

The single job here is the numeric convergence gate — rank-normalized R-hat,
bulk/tail ESS, and divergence count — computed from an ArviZ InferenceData /
DataTree. Sampling lives in inference.py; trace *plotting* lives with the
figures layer, not here.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConvergenceThresholds:
    """Pass/warn/fail bands for the convergence gate.

    Defaults reproduce the historical gate (R-hat < 1.01, ESS > 400, and the
    old hard-FAIL at ESS < 100 via ``ess_min`` being the warn line with
    ``rhat_fail`` the R-hat hard line). A parameter is:

    - PASS  when rhat < ``rhat_warn`` and ess >= ``ess_min``
    - FAIL  when rhat >= ``rhat_fail`` or ess < ``ess_min`` * (fail fraction)
    - WARN  otherwise (marginal mixing)

    ESS is a single number per parameter: the caller passes
    ``min(bulk_ess, tail_ess)`` so one threshold covers both.
    """

    rhat_warn: float = 1.01
    rhat_fail: float = 1.05
    ess_min: float = 400.0
    # ESS below this fraction of ess_min is a hard FAIL (else WARN). 0.25 keeps
    # the historical FAIL-at-100 when ess_min=400.
    ess_fail_fraction: float = 0.25

    def status(self, rhat: float, ess: float) -> str:
        """Classify one parameter's worst R-hat and min ESS."""
        if rhat >= self.rhat_fail or ess < self.ess_min * self.ess_fail_fraction:
            return "FAIL"
        if rhat >= self.rhat_warn or ess < self.ess_min:
            return "WARN"
        return "PASS"


_DEFAULT_THRESHOLDS = ConvergenceThresholds()


def _param_span_is_zero(idata, var: str) -> bool:
    """True if a posterior variable is constant across all (chain, draw)."""
    vals = np.asarray(idata.posterior[var].values)
    span = vals.max(axis=(0, 1)) - vals.min(axis=(0, 1))
    return bool(np.all(span == 0))


def convergence_summary(
    idata,
    params: list[str] | None = None,
    thresholds: ConvergenceThresholds | None = None,
) -> dict[str, Any]:
    """Rank-normalized R-hat / ESS gate (Vehtari et al. 2021 via ArviZ).

    The verdict and the reported rhat_max / ess_*_min are computed over
    ``params`` only (prefix match on variable name, so "mu" covers "mu" and
    scoped variants) — so non-identifiable sites (state_fe_z, unit_weight, ...)
    excluded from ``params`` don't fail the gate. Default None = every
    parameter (the historical gate). Divergences are ALWAYS counted over the
    full run regardless of ``params``. ``thresholds`` defaults to the
    historical R-hat<1.01 / ESS>400 bands.
    """
    import arviz as az

    t = thresholds or _DEFAULT_THRESHOLDS

    if params is not None:
        all_vars = list(idata.posterior.data_vars)
        keep = [v for v in all_vars if any(v.startswith(p) for p in params)]
        if not keep:
            raise ValueError(
                f"gate_params matched no posterior variables: {params} "
                f"(available: {all_vars})"
            )
        stats = az.summary(idata, var_names=keep, kind="diagnostics", round_to="none")
    else:
        stats = az.summary(idata, kind="diagnostics", round_to="none")

    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())

    rhat_max = float(stats["r_hat"].max())
    ess_bulk_min = float(stats["ess_bulk"].min())
    ess_tail_min = float(stats["ess_tail"].min())
    result: dict[str, Any] = {
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_bulk_min,
        "ess_tail_min": ess_tail_min,
        "divergences": divergences,
    }
    # Verdict from the worst gated param (min ESS across bulk/tail) + zero
    # divergences. status() != FAIL/WARN means PASS-level R-hat and ESS.
    worst_status = t.status(rhat=rhat_max, ess=min(ess_bulk_min, ess_tail_min))
    result["converged"] = bool(worst_status == "PASS" and divergences == 0)
    if params is not None:
        result["gate_params"] = list(params)
    return result


def parameter_diagnostics(
    idata,
    gate_params: list[str] | None = None,
    thresholds: ConvergenceThresholds | None = None,
) -> list[dict]:
    """Per-parameter R-hat/ESS rows for display, worst first.

    Shows EVERY posterior variable (so fixed/non-identifiable sites stay
    visible), tagging each with ``gated`` — whether it is in ``gate_params``
    (None = all gated). Only gated params' FAIL/WARN reflect on the run; the
    caller decides the overall verdict from the gated subset.

    A site is 'fixed' iff it has zero variance across (chain, draw) in THIS
    idata — empirical, so ``disp`` under ``sample_disp: false`` and cut
    Stage-2's ``mu_ctrl`` are labeled without hard-coded model knowledge.
    Fixed sites get no R-hat/ESS (undefined) and never FAIL. ``ess`` is
    ``min(bulk, tail)``; per-parameter divergences are meaningless (a
    divergence is a trajectory property, reported run-level elsewhere).
    """
    import arviz as az

    t = thresholds or _DEFAULT_THRESHOLDS
    all_vars = list(idata.posterior.data_vars)

    def _is_gated(name: str) -> bool:
        return gate_params is None or any(name.startswith(p) for p in gate_params)

    fixed: list[str] = []
    varying: list[str] = []
    for v in all_vars:
        (fixed if _param_span_is_zero(idata, v) else varying).append(v)

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
            ess = min(ess_bulk, ess_tail)
            rows.append(
                {
                    "param": name,
                    "rhat": rhat,
                    "ess": ess,
                    "ess_bulk": ess_bulk,
                    "ess_tail": ess_tail,
                    "status": t.status(rhat=rhat, ess=ess),
                    "gated": _is_gated(str(name)),
                }
            )
    rows.sort(key=lambda r: r["rhat"], reverse=True)
    for v in sorted(fixed):
        rows.append(
            {
                "param": v,
                "rhat": None,
                "ess": None,
                "ess_bulk": None,
                "ess_tail": None,
                "status": "fixed",
                "gated": _is_gated(v),
            }
        )
    return rows
