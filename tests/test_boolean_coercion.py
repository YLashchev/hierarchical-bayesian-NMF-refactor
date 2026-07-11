"""
Regression tests for boolean coercion guards in input configurations.

Covers to-do item 33: explicitly quoted YAML booleans (e.g. "false") evaluate
to Python truthy strings. validate_config() must reject them and enforce strict
boolean values.
"""

import pytest
from bayesian_panel_nmf.validation import validate_config, ConfigError

BOOL_PATHS = [
    ("data", "allow_unbalanced_panel"),
    ("data", "aggregation", "enabled"),
    ("model", "sample_disp"),
    ("model", "adjust_for_missingness"),
    ("model", "model_treated"),
    ("mcmc", "progress_bar"),
    ("output", "figures"),
    ("output", "clean"),
]


@pytest.fixture
def minimal_config():
    """Valid minimal configuration required by schema."""
    return {
        "data": {
            "schema": {
                "unit_col": "state",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes": [{"outcome_col": "y", "label": "total"}],
            }
        },
        "model": {},
        "mcmc": {},
        "output": {},
    }


def _set_nested_val(config, path_tuple, value):
    """Helper to set a deeply nested key in a dictionary."""
    curr = config
    for key in path_tuple[:-1]:
        curr.setdefault(key, {})
        curr = curr[key]
    curr[path_tuple[-1]] = value


@pytest.mark.parametrize("path_tuple", BOOL_PATHS)
@pytest.mark.parametrize("bad_value", ["false", "true", "False", "True", 1, 0, "yes"])
def test_quoted_bool_rejected(minimal_config, path_tuple, bad_value):
    _set_nested_val(minimal_config, path_tuple, bad_value)

    # Validation error should specifically call out the final key that failed
    with pytest.raises(ConfigError, match="must be boolean"):
        validate_config(minimal_config)


@pytest.mark.parametrize("path_tuple", BOOL_PATHS)
@pytest.mark.parametrize("good_value", [True, False, None])
def test_valid_bool_accepted(minimal_config, path_tuple, good_value):
    _set_nested_val(minimal_config, path_tuple, good_value)

    # Should not raise exception
    validate_config(minimal_config)


def test_nested_total_all_quoted_rejected(minimal_config):
    minimal_config["model"] = {"types": {"synthetic_total": {"total_all": "true"}}}
    with pytest.raises(ConfigError, match="must be boolean"):
        validate_config(minimal_config)


def test_nested_total_all_accepted(minimal_config):
    minimal_config["model"] = {"types": {"synthetic_total": {"total_all": True}}}
    validate_config(minimal_config)


def test_flag_absent_ok(minimal_config):
    # Base minimal config without any of the boolean flags set
    validate_config(minimal_config)


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, "60"])
def test_progress_interval_rejects_non_positive_integer(minimal_config, bad_value):
    minimal_config["output"]["progress_interval_seconds"] = bad_value
    with pytest.raises(ConfigError, match="progress_interval_seconds"):
        validate_config(minimal_config)


def test_progress_interval_accepts_positive_integer(minimal_config):
    minimal_config["output"]["progress_interval_seconds"] = 30
    validate_config(minimal_config)
