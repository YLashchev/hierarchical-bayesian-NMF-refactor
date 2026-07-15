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


from loguru import logger  # noqa: E402

from bayesian_panel_nmf.cut_inference import resolve_cut_settings  # noqa: E402


def test_defaults_resolved_from_minimal_config():
    s = resolve_cut_settings({"mcmc": {"random_seed": 100}})
    assert s.num_stage1_draws == 25
    assert s.selection_seed == 102
    assert s.stage2_seed == 103
    assert s.stage2_draws_per_component == 100


def test_explicit_null_per_component_means_keep_all():
    s = resolve_cut_settings({"cut": {"stage2_draws_per_component": None}})
    assert s.stage2_draws_per_component is None


def test_stage2_mcmc_overlay_inherits_base():
    cfg = {
        "mcmc": {"num_warmup": 2000, "thinning": 5, "max_chains": 4, "random_seed": 1},
        "cut": {"stage2_mcmc": {"num_warmup": 100}},
    }
    s = resolve_cut_settings(cfg)
    assert s.stage2_mcmc["num_warmup"] == 100
    assert s.stage2_mcmc["thinning"] == 5
    assert s.stage2_mcmc["max_chains"] == 4
    assert "random_seed" not in s.stage2_mcmc


def test_seed_collision_warns():
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        resolve_cut_settings({"mcmc": {"random_seed": 7}, "cut": {"selection_seed": 7}})
    finally:
        logger.remove(sink_id)
    assert any("collides" in m for m in messages)
