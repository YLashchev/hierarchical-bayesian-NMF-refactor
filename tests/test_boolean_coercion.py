"""
Regression tests for boolean coercion guards in input configurations.

Covers to-do item 33: explicitly quoted YAML booleans (e.g. "false") evaluate
to Python truthy strings. Config.model_validate() must reject them and enforce strict
boolean values.
"""

import pytest

from bayesian_panel_nmf.config import Config
from bayesian_panel_nmf.validation import ConfigError

BOOL_PATHS = [
    ("data", "allow_unbalanced_panel"),
    ("data", "aggregation", "enabled"),
    ("model", "sample_disp"),
    ("model", "adjust_for_missingness"),
    ("model", "model_treated"),
    ("mcmc", "progress_bar"),
    ("output", "clean"),
]

# output.figures is intentionally NOT a plain StrictBool (Phase 7.3): it
# accepts bool | list[str] | Literal["all", "none"] so output.figures can
# select a PLOT_REGISTRY subset. It still rejects quoted-string booleans
# (see test_figures_quoted_bool_rejected below), just with a different
# error message than the other pure-boolean fields.
FIGURES_PATH = ("output", "figures")


@pytest.fixture
def minimal_config():
    """Valid minimal configuration required by schema."""
    return {
        "data": {
            "input_file": "input.csv",
            "output_dir": "results",
            "schema": {
                "unit_col": "state",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes": [{"outcome_col": "y", "label": "total"}],
            },
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

    # Validation error should specifically call out the field that failed;
    # pydantic reports "Input should be a valid boolean" for non-bool values.
    with pytest.raises(ConfigError, match="valid boolean"):
        Config.model_validate(minimal_config)


@pytest.mark.parametrize("path_tuple", BOOL_PATHS)
@pytest.mark.parametrize("good_value", [True, False])
def test_valid_bool_accepted(minimal_config, path_tuple, good_value):
    _set_nested_val(minimal_config, path_tuple, good_value)

    # Should not raise exception
    Config.model_validate(minimal_config)


def test_nested_total_all_quoted_rejected(minimal_config):
    minimal_config["model"] = {
        "types": {
            "synthetic_total": {
                "groups": ["g1"],
                "ranks_to_test": [1],
                "total_all": "true",
            }
        }
    }
    with pytest.raises(ConfigError, match="valid boolean"):
        Config.model_validate(minimal_config)


def test_nested_total_all_accepted(minimal_config):
    minimal_config["model"] = {
        "types": {
            "synthetic_total": {
                "groups": ["g1"],
                "ranks_to_test": [1],
                "total_all": True,
            }
        }
    }
    Config.model_validate(minimal_config)


@pytest.mark.parametrize("bad_value", ["false", "true", "False", "True", "yes", 1, 0])
def test_figures_quoted_bool_rejected(minimal_config, bad_value):
    _set_nested_val(minimal_config, FIGURES_PATH, bad_value)
    with pytest.raises(ConfigError):
        Config.model_validate(minimal_config)


@pytest.mark.parametrize("good_value", [True, False, "all", "none", [], ["interval"]])
def test_figures_accepted_spellings(minimal_config, good_value):
    _set_nested_val(minimal_config, FIGURES_PATH, good_value)
    Config.model_validate(minimal_config)


def test_figures_unknown_name_rejected(minimal_config):
    _set_nested_val(minimal_config, FIGURES_PATH, ["bogus"])
    with pytest.raises(ConfigError, match="unknown figure name"):
        Config.model_validate(minimal_config)


def test_flag_absent_ok(minimal_config):
    # Base minimal config without any of the boolean flags set.
    # Note: explicit null is no longer equivalent to absent under the
    # pydantic schema's StrictBool fields (Phase 3 Task 3.3) -- a deliberate
    # tightening, not a regression: `flag: null` in YAML was a silent no-op
    # under the old dict-based validator.
    Config.model_validate(minimal_config)
