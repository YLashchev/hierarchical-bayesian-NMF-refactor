"""Guards for the Stage-2 conditional treatment-effect model."""

import jax
import numpy as np
import pytest
from numpyro.handlers import seed, trace

from bayesian_panel_nmf.models.cut_treatment import stage2_model

K, D, N = 2, 3, 4


def _tiny():
    rng = np.random.default_rng(0)
    Y = rng.poisson(50, size=(K, D, N)).astype(float)
    control = np.ones((K, D, N), dtype=bool)
    control[:, 0, 2:] = False
    missing = np.zeros((K, D, N), dtype=bool)
    missing[0, 0, 3] = True  # suppressed EXPOSED cell
    Y[missing] = 0.0
    return Y, control, missing


def _kwargs(Y, control, missing, **over):
    base = {
        "mu_ctrl": np.zeros((K, D, N)),
        "control_idx_array": control,
        "missing_idx_array": missing,
        "y": Y,
        "outcome_dist": "NB",
        "nb_concentration": np.ones(D) / 1e-4,
        "adjust_for_missingness": True,
    }
    base.update(over)
    return base


def test_nb_without_concentration_raises():
    Y, control, missing = _tiny()
    with pytest.raises(ValueError, match="nb_concentration"):
        stage2_model(**_kwargs(Y, control, missing, nb_concentration=None))


def test_none_control_mask_raises():
    Y, control, missing = _tiny()
    with pytest.raises(ValueError, match="control_idx_array"):
        stage2_model(**_kwargs(Y, control, missing, control_idx_array=None))


def test_poisson_never_constructs_concentration():
    Y, control, missing = _tiny()
    t = trace(seed(stage2_model, jax.random.PRNGKey(0))).get_trace(
        **_kwargs(Y, control, missing, outcome_dist="Poisson", nb_concentration=None)
    )
    assert "disp" not in t
    assert t["y_obs"]["fn"].__class__.__name__ == "Poisson"


def test_mu_equals_mu_ctrl_on_unexposed_cells():
    Y, control, missing = _tiny()
    mu_ctrl = np.arange(K * D * N, dtype=float).reshape(K, D, N)
    t = trace(seed(stage2_model, jax.random.PRNGKey(0))).get_trace(
        **_kwargs(Y, control, missing, mu_ctrl=mu_ctrl)
    )
    mu = np.asarray(t["mu"]["value"])
    np.testing.assert_array_equal(mu[control], mu_ctrl[control])
    te = np.asarray(t["te"]["value"])
    np.testing.assert_array_equal(te[control], np.zeros_like(te[control]))


def test_censored_exposed_factors_present_iff_adjusting():
    Y, control, missing = _tiny()
    t_on = trace(seed(stage2_model, jax.random.PRNGKey(0))).get_trace(
        **_kwargs(Y, control, missing)
    )
    assert "suppressed_counts/missing_factors" in t_on
    t_off = trace(seed(stage2_model, jax.random.PRNGKey(0))).get_trace(
        **_kwargs(Y, control, missing, adjust_for_missingness=False)
    )
    assert "suppressed_counts/missing_factors" not in t_off
