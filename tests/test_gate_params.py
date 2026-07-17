"""Selectable convergence-gate parameters (mcmc.gate_params).

Non-identifiable sites (fixed effects, unit_weight) legitimately fail R-hat
while the quantities of interest (mu, te) mix fine. gate_params restricts
which sample sites feed the R-hat/ESS gate; thresholds are untouched and
divergences are always counted. Default (None) = all parameters, matching
the historical gate exactly.
"""

import arviz as az
import numpy as np
import pytest

from bayesian_panel_nmf.config import Config
from bayesian_panel_nmf.diagnostics import convergence_summary


def _idata(n_chains=2, n_draws=200, seed=0):
    """Two params: 'good' mixes across chains, 'bad' does not."""
    rng = np.random.default_rng(seed)
    good = rng.normal(size=(n_chains, n_draws))
    bad = np.stack([rng.normal(loc=0, size=n_draws), rng.normal(loc=9, size=n_draws)])
    return az.from_dict(
        {
            "posterior": {"good": good, "bad_fe": bad},
            "sample_stats": {"diverging": np.zeros((n_chains, n_draws), bool)},
        }
    )


def test_default_gates_all_params_and_fails_on_bad():
    gate = convergence_summary(_idata())
    assert gate["converged"] is False  # bad_fe wrecks rhat_max
    assert gate["rhat_max"] > 1.01
    assert "gate_params" not in gate  # default output schema unchanged


def test_selected_params_gate_only_those():
    gate = convergence_summary(_idata(), params=["good"])
    assert gate["converged"] is True
    assert gate["rhat_max"] < 1.01
    assert gate["gate_params"] == ["good"]


def test_prefix_matches_scoped_and_indexed_names():
    # cut-mode keys can be scoped ('suppressed_counts/...') or expanded
    # ('mu[0,1]'); selection is prefix-based on the var name.
    gate = convergence_summary(_idata(), params=["good", "nonexistent"])
    assert gate["converged"] is True


def test_no_matching_params_errors():
    with pytest.raises(ValueError, match="gate_params matched no"):
        convergence_summary(_idata(), params=["nope"])


def test_divergences_always_counted():
    idata = _idata()
    idata.sample_stats["diverging"][:] = True
    gate = convergence_summary(idata, params=["good"])
    assert gate["divergences"] > 0
    assert gate["converged"] is False


def test_mcmc_config_accepts_gate_params():
    cfg = Config.model_validate(
        {
            "data": {
                "input_file": "x.csv",
                "output_dir": "o",
                "schema": {
                    "unit_col": "s",
                    "time_col": "t",
                    "treatment_col": "e",
                    "outcomes": [{"outcome_col": "y", "label": "total"}],
                },
            },
            "mcmc": {"gate_params": ["mu", "te"]},
        }
    )
    assert cfg.mcmc.gate_params == ["mu", "te"]
