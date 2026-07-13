"""Tests for run_mcmc_inference's auto_parallelism wiring: confirms the
MCMC constructor receives (num_chains, chain_method) from
choose_mcmc_parallelism when auto_parallelism is true (default), and
from literal config values when explicitly false. No real MCMC sampling
-- MCMC/NUTS are mocked, matching the existing pattern in
tests/test_inference_timing.py."""

import numpy as np

from bayesian_panel_nmf import inference


class _DummyMCMC:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def run(self, *args, **kwargs):
        pass

    def get_samples(self, group_by_chain=False):
        return {"mu_ctrl": np.ones((1, 1, 1, 1, 1))}


def _minimal_data_dict():
    shape = (1, 1, 1)
    return {
        "Y": np.ones(shape),
        "denominators": np.ones(shape),
        "control_idx_array": np.ones(shape, dtype=bool),
        "missing_idx_array": np.zeros(shape, dtype=bool),
        "groups": ["g"],
        "units": ["u"],
        "times": [np.datetime64("2020-01-01")],
    }


def test_auto_parallelism_true_uses_choose_mcmc_parallelism(monkeypatch):
    monkeypatch.setattr(inference, "MCMC", _DummyMCMC)
    monkeypatch.setattr(inference, "NUTS", lambda model_fn: model_fn)
    monkeypatch.setattr(
        inference, "choose_mcmc_parallelism", lambda max_chains: (2, "vectorized")
    )
    monkeypatch.setattr(inference, "block_until_ready", lambda value: value)

    mcmc = inference.run_mcmc_inference(
        _minimal_data_dict(),
        model_fn=lambda **kwargs: None,
        rank=1,
        config={"model": {}, "mcmc": {"max_chains": 4}},
    )

    assert mcmc.kwargs["num_chains"] == 2
    assert mcmc.kwargs["chain_method"] == "vectorized"


def test_auto_parallelism_true_is_the_default_when_key_absent(monkeypatch):
    """auto_parallelism defaults to True when the key is missing entirely."""
    monkeypatch.setattr(inference, "MCMC", _DummyMCMC)
    monkeypatch.setattr(inference, "NUTS", lambda model_fn: model_fn)
    called = []
    monkeypatch.setattr(
        inference,
        "choose_mcmc_parallelism",
        lambda max_chains: (called.append(max_chains), (3, "parallel"))[1],
    )
    monkeypatch.setattr(inference, "block_until_ready", lambda value: value)

    mcmc = inference.run_mcmc_inference(
        _minimal_data_dict(),
        model_fn=lambda **kwargs: None,
        rank=1,
        config={"model": {}, "mcmc": {}},
    )

    assert called == [4]  # default max_chains when absent
    assert mcmc.kwargs["num_chains"] == 3
    assert mcmc.kwargs["chain_method"] == "parallel"


def test_auto_parallelism_false_uses_literal_config_values(monkeypatch):
    monkeypatch.setattr(inference, "MCMC", _DummyMCMC)
    monkeypatch.setattr(inference, "NUTS", lambda model_fn: model_fn)

    def _fail_if_called(max_chains):
        raise AssertionError("choose_mcmc_parallelism should not be called")

    monkeypatch.setattr(inference, "choose_mcmc_parallelism", _fail_if_called)
    monkeypatch.setattr(inference, "block_until_ready", lambda value: value)

    mcmc = inference.run_mcmc_inference(
        _minimal_data_dict(),
        model_fn=lambda **kwargs: None,
        rank=1,
        config={
            "model": {},
            "mcmc": {
                "auto_parallelism": False,
                "num_chains": 6,
                "chain_method": "vectorized",
            },
        },
    )

    assert mcmc.kwargs["num_chains"] == 6
    assert mcmc.kwargs["chain_method"] == "vectorized"


def test_auto_parallelism_false_defaults_chain_method_to_sequential(monkeypatch):
    """When auto_parallelism=false and chain_method is not given, default
    to 'sequential' (the safe manual-override default), not the old
    hardcoded 'parallel'."""
    monkeypatch.setattr(inference, "MCMC", _DummyMCMC)
    monkeypatch.setattr(inference, "NUTS", lambda model_fn: model_fn)
    monkeypatch.setattr(
        inference,
        "choose_mcmc_parallelism",
        lambda max_chains: (_ for _ in ()).throw(
            AssertionError("should not be called")
        ),
    )
    monkeypatch.setattr(inference, "block_until_ready", lambda value: value)

    mcmc = inference.run_mcmc_inference(
        _minimal_data_dict(),
        model_fn=lambda **kwargs: None,
        rank=1,
        config={
            "model": {},
            "mcmc": {"auto_parallelism": False, "num_chains": 4},
        },
    )

    assert mcmc.kwargs["num_chains"] == 4
    assert mcmc.kwargs["chain_method"] == "sequential"
