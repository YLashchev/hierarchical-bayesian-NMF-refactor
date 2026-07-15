"""Validation + resolution rules for the cut configuration."""

import pytest

from bayesian_panel_nmf.validation import ConfigError, validate_config


def _base_config(**model_over):
    model = {
        "types": {"total": {"groups": ["total"]}},
        "inference_mode": "cut",
    }
    model.update(model_over)
    return {
        "data": {
            "input_file": "x.csv",
            "schema": {
                "unit_col": "state",
                "time_col": "time",
                "treatment_col": "exposed",
                "outcomes": [{"outcome_col": "births_total", "label": "total"}],
            },
        },
        "model": model,
    }


def test_joint_config_without_new_keys_still_validates():
    cfg = _base_config()
    del cfg["model"]["inference_mode"]
    validate_config(cfg)


def test_inference_mode_values():
    validate_config(_base_config(inference_mode="joint"))
    validate_config(_base_config(inference_mode="cut"))
    with pytest.raises(ConfigError, match="inference_mode"):
        validate_config(_base_config(inference_mode="bogus"))


def test_cut_requires_model_treated():
    with pytest.raises(ConfigError, match="model_treated"):
        validate_config(_base_config(model_treated=False))


def test_cut_block_type_checks():
    cfg = _base_config()
    cfg["cut"] = "nope"
    with pytest.raises(ConfigError, match="cut.*dict"):
        validate_config(cfg)


@pytest.mark.parametrize("key", ["num_stage1_draws", "stage2_draws_per_component"])
@pytest.mark.parametrize("bad", [0, -1, "5", True, 2.5])
def test_positive_int_keys_rejected(key, bad):
    cfg = _base_config()
    cfg["cut"] = {key: bad}
    with pytest.raises(ConfigError, match=key):
        validate_config(cfg)


@pytest.mark.parametrize("key", ["selection_seed", "stage2_seed"])
def test_seed_keys_must_be_ints(key):
    cfg = _base_config()
    cfg["cut"] = {key: "8675309"}
    with pytest.raises(ConfigError, match=key):
        validate_config(cfg)


def test_stage2_mcmc_random_seed_rejected():
    cfg = _base_config()
    cfg["cut"] = {"stage2_mcmc": {"random_seed": 1}}
    with pytest.raises(ConfigError, match="stage2_seed"):
        validate_config(cfg)


def test_valid_cut_block_accepted():
    cfg = _base_config()
    cfg["cut"] = {
        "num_stage1_draws": 25,
        "selection_seed": 1,
        "stage2_seed": 2,
        "stage2_draws_per_component": 100,
        "stage2_mcmc": {"num_warmup": 500, "num_samples": 500},
    }
    validate_config(cfg)
