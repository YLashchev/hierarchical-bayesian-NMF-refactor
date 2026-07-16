from pathlib import Path

import pytest

from bayesian_panel_nmf.config import Config
from bayesian_panel_nmf.validation import ConfigError

CONFIGS = sorted(Path("configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_all_shipped_configs_load(path):
    cfg = Config.from_yaml(str(path))
    assert cfg.data.schema_.unit_col  # schema present
    assert cfg.mcmc.num_warmup >= 1


def test_defaults_match_legacy():
    cfg = Config.model_validate(
        {
            "data": {
                "input_file": "x.csv",
                "output_dir": "results",
                "schema": {
                    "unit_col": "u",
                    "time_col": "t",
                    "treatment_col": "z",
                    "outcomes": [{"outcome_col": "y", "label": "total"}],
                },
            },
            "model": {"types": {"total": {"groups": ["total"], "ranks_to_test": [5]}}},
        }
    )
    assert cfg.mcmc.num_warmup == 1000
    assert cfg.mcmc.num_samples == 2500
    assert cfg.mcmc.thinning == 10
    assert cfg.mcmc.random_seed == 8675309
    assert cfg.mcmc.progress_bar
    assert cfg.mcmc.auto_parallelism
    assert cfg.mcmc.max_chains == 4
    assert cfg.model.outcome_distribution == "NB"
    assert cfg.model.nb_disp == 1e-4
    assert not cfg.model.sample_disp
    assert cfg.model.adjust_for_missingness
    assert cfg.model.model_treated
    assert cfg.model.inference_mode is None
    assert not cfg.data.allow_unbalanced_panel
    assert not cfg.data.aggregation.enabled
    assert cfg.data.aggregation.period == "bimonthly"
    assert cfg.data.date_format == "auto"
    assert not cfg.output.figures
    assert cfg.output.print_tables


def test_forbid_unknown_key():
    with pytest.raises(ConfigError):
        Config.model_validate(
            {
                "data": {
                    "input_file": "x.csv",
                    "output_dir": "r",
                    "schema": {
                        "unit_col": "u",
                        "time_col": "t",
                        "treatment_col": "z",
                        "outcomes": [{"outcome_col": "y", "label": "total"}],
                    },
                },
                "mcmc": {"num_wrmup": 500},  # typo
            }
        )


def test_quoted_boolean_rejected():
    with pytest.raises(ConfigError):
        Config.model_validate(
            {
                "data": {
                    "input_file": "x.csv",
                    "output_dir": "r",
                    "schema": {
                        "unit_col": "u",
                        "time_col": "t",
                        "treatment_col": "z",
                        "outcomes": [{"outcome_col": "y", "label": "total"}],
                    },
                },
                "model": {"sample_disp": "false"},  # quoted → truthy string
            }
        )


def test_schema_xor_enforced():
    base = {"unit_col": "u", "time_col": "t", "treatment_col": "z"}
    with pytest.raises(ConfigError):  # neither
        Config.model_validate(
            {"data": {"input_file": "x", "output_dir": "r", "schema": base}}
        )
