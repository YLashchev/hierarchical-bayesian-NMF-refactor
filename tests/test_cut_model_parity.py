"""Parity guards: the cut stage models must reproduce the joint model's science.

The cut models deliberately duplicate the joint model's untreated branch and
treatment block (approved independence decision -- no imports of private
panel_nmf_model helpers). These tests pin site names, shapes, sampled values
(identical PRNG consumption), and log-densities so any drift between the
copies fails loudly.
"""

import jax
import numpy as np
from numpyro.handlers import seed, substitute, trace
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.util import log_density

from bayesian_panel_nmf.models.cut_stage1_model import stage1_model
from bayesian_panel_nmf.models.panel_nmf_model import model as joint_model


def _traceable(model, seed_val):
    """Wrap a model for direct trace(...).get_trace(...) calls.

    ``state_fe`` uses ``dist.ImproperUniform``, whose ``.sample()`` is
    unconditionally unimplemented in numpyro (by design -- improper-prior
    sites are only ever initialized via an init strategy such as
    ``init_to_uniform``, e.g. inside NUTS/MCMC). Applied identically to the
    joint and cut models so PRNG/init treatment stays symmetric and every
    downstream assertion (bit-identity, log-density, site names) is
    unaffected.
    """
    return substitute(
        seed(model, jax.random.PRNGKey(seed_val)), substitute_fn=init_to_uniform
    )


K, D, N, RANK = 2, 3, 4, 2


def _tiny_data(seed_val: int = 0) -> dict:
    rng = np.random.default_rng(seed_val)
    Y = rng.poisson(50, size=(K, D, N)).astype(float)
    control = np.ones((K, D, N), dtype=bool)
    control[:, 0, 2:] = False  # unit 0 exposed from t=2 onward
    missing = np.zeros((K, D, N), dtype=bool)
    missing[0, 1, 1] = True  # suppressed control cell
    missing[0, 0, 3] = True  # suppressed exposed cell
    Y[missing] = 0.0
    return {
        "Y": Y,
        "denominators": np.full((K, D, N), 2.0),
        "control_idx_array": control,
        "missing_idx_array": missing,
    }


def _stage1_kwargs(d: dict) -> dict:
    return {
        "denominators": d["denominators"],
        "control_idx_array": d["control_idx_array"],
        "missing_idx_array": d["missing_idx_array"],
        "y": d["Y"],
        "rank": RANK,
        "outcome_dist": "NB",
        "adjust_for_missingness": True,
        "nb_disp": 1e-4,
        "sample_disp": False,
    }


def test_stage1_trace_bit_identical_to_joint_untreated_branch():
    d = _tiny_data()
    kwargs = _stage1_kwargs(d)
    t_joint = trace(_traceable(joint_model, 0)).get_trace(model_treated=False, **kwargs)
    t_cut = trace(_traceable(stage1_model, 0)).get_trace(**kwargs)
    assert set(t_joint) == set(t_cut)
    for name, site in t_joint.items():
        if site["type"] in ("sample", "deterministic"):
            np.testing.assert_array_equal(
                np.asarray(site["value"]),
                np.asarray(t_cut[name]["value"]),
                err_msg=name,
            )


def test_stage1_log_density_matches_joint():
    d = _tiny_data()
    kwargs = _stage1_kwargs(d)
    t_cut = trace(_traceable(stage1_model, 0)).get_trace(**kwargs)
    params = {
        name: site["value"]
        for name, site in t_cut.items()
        if site["type"] == "sample" and not site["is_observed"]
    }
    ld_cut, _ = log_density(stage1_model, (), kwargs, params)
    ld_joint, _ = log_density(
        joint_model, (), {**kwargs, "model_treated": False}, params
    )
    np.testing.assert_allclose(float(ld_cut), float(ld_joint), rtol=0, atol=0)


def test_stage1_sample_disp_paths_match_joint():
    d = _tiny_data()
    kwargs = {**_stage1_kwargs(d), "sample_disp": True}
    t_joint = trace(_traceable(joint_model, 3)).get_trace(model_treated=False, **kwargs)
    t_cut = trace(_traceable(stage1_model, 3)).get_trace(**kwargs)
    np.testing.assert_array_equal(
        np.asarray(t_joint["disp"]["value"]), np.asarray(t_cut["disp"]["value"])
    )


def test_stage1_poisson_has_no_disp_site():
    d = _tiny_data()
    kwargs = {**_stage1_kwargs(d), "outcome_dist": "Poisson"}
    t_cut = trace(_traceable(stage1_model, 0)).get_trace(**kwargs)
    assert "disp" not in t_cut
