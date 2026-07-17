from typing import Any, cast

import numpy as np

from bayesian_panel_nmf import inference


def test_run_mcmc_inference_blocks_until_samples_ready(monkeypatch, make_inference_config):
    samples = {"mu_ctrl": np.ones((1, 1, 1, 1, 1))}
    seen = []

    class DummyMCMC:
        num_chains = 1
        num_samples = 1
        num_warmup = 1
        thinning = 1

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.ran = False

        def run(self, *args, **kwargs):
            self.ran = True

        def get_samples(self, group_by_chain=False):
            assert self.ran
            seen.append(group_by_chain)
            return samples

    monkeypatch.setattr(inference, "NUTS", lambda model_fn: ("kernel", model_fn))
    monkeypatch.setattr(inference, "MCMC", DummyMCMC)
    monkeypatch.setattr(
        inference, "choose_mcmc_parallelism", lambda max_chains: (1, "sequential")
    )
    monkeypatch.setattr(
        inference,
        "block_until_ready",
        lambda value: seen.append(("blocked", value)) or value,
    )

    data_dict = {
        "Y": np.ones((1, 1, 1)),
        "denominators": np.ones((1, 1, 1)),
        "control_idx_array": np.ones((1, 1, 1), dtype=bool),
        "missing_idx_array": np.zeros((1, 1, 1), dtype=bool),
        "groups": ["total"],
        "units": ["A"],
        "times": [np.datetime64("2020-01-01")],
    }

    mcmc = inference.run_mcmc_inference(
        data_dict,
        model_fn=lambda **kwargs: None,
        rank=1,
        config=make_inference_config(
            mcmc={
                "num_chains": 1,
                "num_warmup": 1,
                "num_samples": 1,
                "thinning": 1,
                "progress_bar": False,
            }
        ),
    )

    assert isinstance(mcmc, DummyMCMC)
    assert seen == [False, ("blocked", samples)]
    assert mcmc.kwargs["chain_method"] == "sequential"


def test_generate_predictions_blocks_until_predictions_ready(
    monkeypatch, make_inference_config
):
    predictions = np.ones((1, 1, 1, 1))
    seen = []

    class DummyMCMC:
        num_chains = 1

        def get_samples(self, group_by_chain=False):
            return {"mu_ctrl": np.ones((1, 1, 1, 1, 1))}

    class DummyPredictive:
        def __init__(self, model_fn, samples):
            self.model_fn = model_fn
            self.samples = samples

        def __call__(self, *args, **kwargs):
            return {"y_obs": predictions}

    monkeypatch.setattr(inference, "Predictive", DummyPredictive)
    monkeypatch.setattr(
        inference,
        "block_until_ready",
        lambda value: seen.append(value) or value,
    )

    result = inference.generate_predictions(
        cast(Any, DummyMCMC()),
        data_dict={"denominators": np.ones((1, 1, 1))},
        model_fn=lambda **kwargs: None,
        rank=1,
        config=make_inference_config(mcmc={"random_seed": 1}),
    )

    assert seen == [predictions]
    np.testing.assert_array_equal(result, predictions.reshape(1, 1, 1, 1, 1))
