"""Chain-stratified Stage-1 draw selection and output subsampling."""

import numpy as np
import pytest

from bayesian_panel_nmf.config import ModelConfig
from bayesian_panel_nmf.cut import (
    CutSettings,
    select_stage1_draws,
    subsample_component_draws,
)
from bayesian_panel_nmf.validation import ConfigError, DataError

K, D, N = 2, 3, 2


def _settings(**over):
    base = {
        "num_stage1_draws": 5,
        "selection_seed": 11,
        "stage2_seed": 12,
        "stage2_draws_per_component": None,
        "stage2_mcmc": {},
    }
    base.update(over)
    return CutSettings(**base)


def _samples(C=2, S=10, with_disp=False):
    # value encodes provenance: mu_ctrl[c, s] == c * 1000 + s everywhere
    base = (np.arange(C) * 1000)[:, None] + np.arange(S)[None, :]
    mu = np.broadcast_to(base[:, :, None, None, None], (C, S, K, D, N)).astype(float)
    samples = {"mu_ctrl": np.array(mu)}
    if with_disp:
        samples["disp"] = np.full((C, S, D), 0.5)
    return samples


MODEL_NB_FIXED = ModelConfig(
    outcome_distribution="NB", nb_disp=1e-4, sample_disp=False
)


def test_quota_split_and_component_order():
    refs = select_stage1_draws(_samples(C=2, S=10), _settings(), MODEL_NB_FIXED)
    assert [r.component for r in refs] == [1, 2, 3, 4, 5]
    assert sum(r.stage1_chain == 1 for r in refs) == 3  # first chain gets extra
    assert sum(r.stage1_chain == 2 for r in refs) == 2
    ordered = [(r.stage1_chain, r.stage1_iteration) for r in refs]
    assert ordered == sorted(ordered)


def test_selection_is_deterministic():
    a = select_stage1_draws(_samples(), _settings(), MODEL_NB_FIXED)
    b = select_stage1_draws(_samples(), _settings(), MODEL_NB_FIXED)
    assert [(r.stage1_chain, r.stage1_iteration) for r in a] == [
        (r.stage1_chain, r.stage1_iteration) for r in b
    ]


def test_ref_provenance_and_values():
    refs = select_stage1_draws(_samples(C=2, S=10), _settings(), MODEL_NB_FIXED)
    for r in refs:
        expected = (r.stage1_chain - 1) * 1000 + (r.stage1_iteration - 1)
        assert float(r.mu_ctrl[0, 0, 0]) == float(expected)
        assert r.stage1_draw == (r.stage1_chain - 1) * 10 + r.stage1_iteration
        np.testing.assert_allclose(r.nb_concentration, np.ones(D) / 1e-4)


def test_matched_sampled_dispersion():
    model = MODEL_NB_FIXED.model_copy(update={"sample_disp": True})
    refs = select_stage1_draws(_samples(with_disp=True), _settings(), model)
    for r in refs:
        np.testing.assert_allclose(r.nb_concentration, np.full(D, 2.0))  # 1/0.5


def test_poisson_has_no_concentration():
    model = ModelConfig(outcome_distribution="Poisson")
    refs = select_stage1_draws(_samples(), _settings(), model)
    assert all(r.nb_concentration is None for r in refs)


def test_too_many_draws_errors():
    with pytest.raises(ConfigError, match="num_stage1_draws"):
        select_stage1_draws(
            _samples(C=2, S=3), _settings(num_stage1_draws=7), MODEL_NB_FIXED
        )


def test_fewer_draws_than_chains_uses_first_chains():
    refs = select_stage1_draws(
        _samples(C=4, S=5), _settings(num_stage1_draws=2), MODEL_NB_FIXED
    )
    assert [r.stage1_chain for r in refs] == [1, 2]


def test_subsample_keep_all_when_none():
    idx = subsample_component_draws(2, 7, None)
    assert len(idx) == 2
    np.testing.assert_array_equal(idx[0], np.arange(7))


def test_subsample_even_stride():
    idx = subsample_component_draws(1, 5, 3)
    np.testing.assert_array_equal(idx[0], np.array([0, 2, 4]))


def test_subsample_quota_split_endpoints():
    idx = subsample_component_draws(2, 10, 4)
    np.testing.assert_array_equal(idx[0], np.array([0, 9]))
    np.testing.assert_array_equal(idx[1], np.array([0, 9]))


def test_subsample_exceeding_retained_errors():
    with pytest.raises(DataError, match="stage2_draws_per_component"):
        subsample_component_draws(2, 3, 8)
