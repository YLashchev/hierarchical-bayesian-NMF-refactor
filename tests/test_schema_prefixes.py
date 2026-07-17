from pathlib import Path

import pandas as pd
import pytest

from bayesian_panel_nmf.config import Config
from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.validation import ConfigError, DataError


@pytest.fixture
def panel_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "state": ["A", "A", "B", "B"],
            "time": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"],
            "treated": [0, 1, 0, 0],
            "births_a": [10, 12, 20, 18],
            "pop_a": [100, 110, 200, 190],
            "births_b": [5, 7, 8, 9],
            "pop_b": [50, 60, 80, 90],
        }
    )
    path = tmp_path / "panel.csv"
    df.to_csv(path, index=False)
    return path


def make_config(csv_path: Path, schema: dict) -> dict:
    return {
        "data": {
            "input_file": str(csv_path),
            "output_dir": "results/test",
            "schema": schema,
            "aggregation": {"enabled": False},
        }
    }


def test_explicit_outcomes_still_work(panel_csv: Path) -> None:
    config = make_config(
        panel_csv,
        {
            "unit_col": "state",
            "time_col": "time",
            "treatment_col": "treated",
            "outcomes": [
                {
                    "outcome_col": "births_a",
                    "denominator_col": "pop_a",
                    "label": "a",
                },
                {
                    "outcome_col": "births_b",
                    "denominator_col": "pop_b",
                    "label": "b",
                },
            ],
        },
    )

    result = load_and_prepare(str(panel_csv), Config.model_validate(config), groups=["a", "b"])

    assert result["groups"] == ["a", "b"]
    assert sorted(result["df_preprocessed"]["group"].unique().tolist()) == ["a", "b"]


def test_prefix_outcomes_with_include(panel_csv: Path) -> None:
    config = make_config(
        panel_csv,
        {
            "unit_col": "state",
            "time_col": "time",
            "treatment_col": "treated",
            "outcomes_from_prefixes": {
                "outcome_prefix": "births_",
                "denominator_prefix": "pop_",
                "include": ["a", "b"],
            },
        },
    )

    result = load_and_prepare(str(panel_csv), Config.model_validate(config), groups=["a", "b"])

    assert result["groups"] == ["a", "b"]
    assert sorted(result["df_preprocessed"]["group"].unique().tolist()) == ["a", "b"]


def test_prefix_outcomes_autodiscover_shared_suffixes(panel_csv: Path) -> None:
    config = make_config(
        panel_csv,
        {
            "unit_col": "state",
            "time_col": "time",
            "treatment_col": "treated",
            "outcomes_from_prefixes": {
                "outcome_prefix": "births_",
                "denominator_prefix": "pop_",
            },
        },
    )

    result = load_and_prepare(str(panel_csv), Config.model_validate(config), groups=["a", "b"])

    assert result["groups"] == ["a", "b"]
    assert sorted(result["df_preprocessed"]["group"].unique().tolist()) == ["a", "b"]


def test_validate_config_rejects_mixed_schema_modes(panel_csv: Path) -> None:
    config = make_config(
        panel_csv,
        {
            "unit_col": "state",
            "time_col": "time",
            "treatment_col": "treated",
            "outcomes": [{"outcome_col": "births_a", "label": "a"}],
            "outcomes_from_prefixes": {
                "outcome_prefix": "births_",
                "denominator_prefix": "pop_",
            },
        },
    )

    with pytest.raises(ConfigError, match="exactly one"):
        Config.model_validate(config)


def test_validate_config_requires_model_section(panel_csv: Path) -> None:
    config = make_config(
        panel_csv,
        {
            "unit_col": "state",
            "time_col": "time",
            "treatment_col": "treated",
            "outcomes": [{"outcome_col": "births_a", "label": "a"}],
        },
    )

    Config.model_validate(config)


def test_validate_config_requires_model_types(panel_csv: Path) -> None:
    config = make_config(
        panel_csv,
        {
            "unit_col": "state",
            "time_col": "time",
            "treatment_col": "treated",
            "outcomes": [{"outcome_col": "births_a", "label": "a"}],
        },
    )
    config["model"] = {}

    Config.model_validate(config)


def test_validate_config_rejects_non_dict_model_types(panel_csv: Path) -> None:
    config = make_config(
        panel_csv,
        {
            "unit_col": "state",
            "time_col": "time",
            "treatment_col": "treated",
            "outcomes": [{"outcome_col": "births_a", "label": "a"}],
        },
    )
    config["model"] = {"types": []}

    with pytest.raises(ConfigError, match="valid dictionary"):
        Config.model_validate(config)


def test_prefix_outcomes_require_matching_denominator_suffixes(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "state": ["A", "A"],
            "time": ["2020-01-01", "2020-02-01"],
            "treated": [0, 1],
            "births_a": [10, 12],
            "pop_a": [100, 110],
            "births_b": [5, 7],
        }
    )
    csv_path = tmp_path / "panel_missing_denom.csv"
    df.to_csv(csv_path, index=False)

    config = make_config(
        csv_path,
        {
            "unit_col": "state",
            "time_col": "time",
            "treatment_col": "treated",
            "outcomes_from_prefixes": {
                "outcome_prefix": "births_",
                "denominator_prefix": "pop_",
            },
        },
    )

    with pytest.raises(DataError, match="Missing denominator columns"):
        load_and_prepare(str(csv_path), Config.model_validate(config), groups=["a", "b"])


def test_prefix_outcomes_reject_unknown_include_suffix(panel_csv: Path) -> None:
    config = make_config(
        panel_csv,
        {
            "unit_col": "state",
            "time_col": "time",
            "treatment_col": "treated",
            "outcomes_from_prefixes": {
                "outcome_prefix": "births_",
                "denominator_prefix": "pop_",
                "include": ["a", "missing"],
            },
        },
    )

    with pytest.raises(DataError, match="Missing outcome columns"):
        load_and_prepare(str(panel_csv), Config.model_validate(config), groups=["a"])
