"""Per-parameter / per-component diagnostic tables for the run summary.

The convergence GATE (thresholds, JSON) is unchanged; these cover only the
display helpers that replace the old 4-number block.
"""

import arviz as az
import numpy as np

from bayesian_panel_nmf.diagnostics import parameter_diagnostics


def _idata(fixed: bool):
    rng = np.random.default_rng(0)
    post = {
        "mu": rng.normal(size=(2, 300)),
        "time_fac": np.stack([rng.normal(0, 1, 300), rng.normal(9, 1, 300)]),
    }
    if fixed:
        post["mu_ctrl"] = np.full((2, 300), 3.14)
    return az.from_dict({"posterior": post})


def test_rows_worst_first_and_status():
    rows = parameter_diagnostics(_idata(fixed=False))
    assert rows[0]["param"] == "time_fac"
    assert rows[0]["status"] == "FAIL"
    assert {r["param"] for r in rows} == {"mu", "time_fac"}
    assert next(r for r in rows if r["param"] == "mu")["status"] == "PASS"


def test_constant_site_is_fixed_not_failed():
    rows = parameter_diagnostics(_idata(fixed=True))
    mc = next(r for r in rows if r["param"] == "mu_ctrl")
    assert mc["status"] == "fixed"
    assert mc["rhat"] is None


def test_prefix_filter_matches_gate_params():
    rows = parameter_diagnostics(_idata(fixed=True), params=["mu"])
    # "mu" prefix matches both mu and mu_ctrl, not time_fac
    assert {r["param"] for r in rows} == {"mu", "mu_ctrl"}


def test_render_returns_pass_flag():
    from bayesian_panel_nmf.tables import render_diagnostics_table

    ok = render_diagnostics_table(parameter_diagnostics(_idata(fixed=True)))
    assert ok is False  # time_fac fails


def test_component_table_lists_all_components():
    from bayesian_panel_nmf.tables import render_component_table

    fits = [
        {"component": 1, "stage1_chain": 1, "rhat_max": 1.004,
         "ess_bulk_min": 800, "divergences": 0, "converged": True},
        {"component": 2, "stage1_chain": 2, "rhat_max": 1.9,
         "ess_bulk_min": 4, "divergences": 0, "converged": False},
        {"component": 3, "stage1_chain": 1, "rhat_max": 1.001,
         "ess_bulk_min": 950, "divergences": 0, "converged": True},
    ]
    # returns overall-pass flag; renders one row per component (all 3)
    ok = render_component_table(fits, title="cut components")
    assert ok is False  # component 2 fails
