"""Default trace-plot variable selection (_select_variables_to_plot).

After the hierarchical state_fe prior, the sampled sites are state_fe_mu,
state_fe_sigma, state_fe_z (the old 'state_fe' name is gone). The default
trace set must include them + the full treatment block, and exclude the huge
derived surfaces mu/mu_ctrl (reachable via --param-filter, but a 20-of-2448
subsample isn't a useful default trace).
"""

from bayesian_panel_nmf.cli import _select_variables_to_plot

# The posterior sites a joint NB run emits (post prior change).
ALL_VARS = [
    "category_treatment_effect",
    "disp",
    "mu",
    "mu_ctrl",
    "state_category_scale",
    "state_category_te",
    "state_fe_mu",
    "state_fe_sigma",
    "state_fe_z",
    "state_treatment_effect",
    "te",
    "time_fac",
    "time_fe",
    "treatment_category_scale",
    "treatment_it_scale",
    "treatment_kt",
    "treatment_state_scale",
    "unit_weight",
]


def test_default_includes_new_state_fe_sites():
    got = _select_variables_to_plot(ALL_VARS, None)
    assert "state_fe_mu" in got
    assert "state_fe_sigma" in got
    assert "state_fe_z" in got
    assert "state_fe" not in got  # stale name must not appear


def test_default_includes_full_treatment_block():
    got = _select_variables_to_plot(ALL_VARS, None)
    for p in (
        "treatment_it_scale",
        "treatment_state_scale",
        "treatment_category_scale",
        "state_category_scale",
        "treatment_kt",
        "state_treatment_effect",
        "category_treatment_effect",
        "state_category_te",
        "te",
    ):
        assert p in got, p


def test_default_excludes_mu_surfaces():
    got = _select_variables_to_plot(ALL_VARS, None)
    assert "mu" not in got
    assert "mu_ctrl" not in got


def test_default_only_selects_present_vars():
    # a Poisson run has no 'disp'; selection must not invent absent names
    poisson_vars = [v for v in ALL_VARS if v != "disp"]
    got = _select_variables_to_plot(poisson_vars, None)
    assert "disp" not in got
    assert set(got).issubset(set(poisson_vars))


def test_param_filter_still_reaches_mu():
    got = _select_variables_to_plot(ALL_VARS, ["mu"])
    # prefix match -> mu and mu_ctrl both reachable explicitly
    assert "mu" in got and "mu_ctrl" in got
