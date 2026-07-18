"""Configurable convergence thresholds + gate-scoped verdict.

Contract:
- ConvergenceThresholds.status(rhat, ess) -> PASS/WARN/FAIL from
  rhat_warn/rhat_fail/ess_min; ESS is a single value (caller passes
  min(bulk, tail)).
- parameter_diagnostics shows ALL params (incl. fixed/non-gated), each tagged
  gated=True/False; only gated params' FAILs drive the overall verdict.
- convergence_summary verdict + rhat_max/ess_*_min are computed over
  gate_params only (unchanged); defaults reproduce the historical gate.
- divergences run-level only, always counted.
- fixed (zero-variance) sites never fail and are always shown.
"""

import arviz as az
import numpy as np

from bayesian_panel_nmf.diagnostics import (
    ConvergenceThresholds,
    convergence_summary,
    parameter_diagnostics,
)


def _idata(fixed=False, diverge=0, seed=0):
    rng = np.random.default_rng(seed)
    post = {
        "mu": rng.normal(size=(2, 400)),  # healthy
        "time_fac": np.stack([rng.normal(0, 1, 400), rng.normal(9, 1, 400)]),  # bad
    }
    if fixed:
        post["mu_ctrl"] = np.full((2, 400), 3.14)
    div = np.zeros((2, 400), bool)
    if diverge:
        div.flat[:diverge] = True
    return az.from_dict({"posterior": post, "sample_stats": {"diverging": div}})


# ---- thresholds ------------------------------------------------------------


def test_status_bands():
    t = ConvergenceThresholds(rhat_warn=1.01, rhat_fail=1.05, ess_min=400)
    assert t.status(rhat=1.001, ess=1000) == "PASS"
    assert t.status(rhat=1.03, ess=1000) == "WARN"  # rhat in warn band
    assert t.status(rhat=1.07, ess=1000) == "FAIL"  # rhat past fail
    assert t.status(rhat=1.0, ess=200) == "WARN"    # ess below min


def test_defaults_match_history():
    t = ConvergenceThresholds()
    assert (t.rhat_warn, t.ess_min) == (1.01, 400)
    # historical FAIL was rhat>=1.01 or ess_bulk<100; default rhat_fail keeps
    # a param with rhat just over warn as WARN, and <100 ess as FAIL.
    assert t.status(rhat=1.0, ess=99) == "FAIL"


# ---- parameter_diagnostics: show all, gate on subset -----------------------


def test_shows_all_params_tags_gated():
    rows = parameter_diagnostics(_idata(fixed=True), gate_params=["mu"])
    names = {r["param"] for r in rows}
    assert names == {"mu", "mu_ctrl", "time_fac"}  # ALL shown
    gated = {r["param"]: r["gated"] for r in rows}
    assert gated["mu"] is True
    assert gated["mu_ctrl"] is True     # "mu" prefix also matches mu_ctrl
    assert gated["time_fac"] is False   # not in gate_params


def test_ess_column_is_min_bulk_tail():
    rows = parameter_diagnostics(_idata())
    mu = next(r for r in rows if r["param"] == "mu")
    assert mu["ess"] == min(mu["ess_bulk"], mu["ess_tail"])


def test_fixed_shown_never_fails():
    rows = parameter_diagnostics(_idata(fixed=True), gate_params=["mu", "mu_ctrl"])
    mc = next(r for r in rows if r["param"] == "mu_ctrl")
    assert mc["status"] == "fixed" and mc["ess"] is None


# ---- convergence_summary: verdict over gate_params -------------------------


def test_verdict_scoped_to_gate_params():
    # time_fac fails, but if we only gate "mu", the run should converge
    gate = convergence_summary(_idata(), params=["mu"])
    assert gate["converged"] is True
    # ...whereas gating everything fails (time_fac)
    assert convergence_summary(_idata())["converged"] is False


def test_default_keys_and_divergences():
    gate = convergence_summary(_idata(diverge=2))
    assert set(gate) >= {"rhat_max", "ess_bulk_min", "ess_tail_min",
                         "divergences", "converged"}
    assert gate["divergences"] == 2
    assert gate["converged"] is False  # divergences fail the gate


def test_summary_accepts_custom_thresholds():
    lax = ConvergenceThresholds(rhat_warn=5.0, rhat_fail=10.0, ess_min=1)
    assert convergence_summary(_idata(), thresholds=lax)["converged"] is True
