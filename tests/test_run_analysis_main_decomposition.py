"""Characterization tests for cli.py's `run` subcommand type-selection
logic (formerly run_analysis.py's main(), moved in Phase 9.2), pinned by
_select_types_to_run (Tier 2 of the repo clarity refactor). This logic
previously had no direct test — only indirect exercise via the two dispatch
tests in test_run_analysis_parallel.py, both of which always run all
types."""

import pytest

from bayesian_panel_nmf import cli as run_analysis
from bayesian_panel_nmf import pipeline
from bayesian_panel_nmf.validation import ConfigError


def _minimal_config(tmp_path):
    return {
        "data": {
            "input_file": str(tmp_path / "unused.csv"),
            "output_dir": str(tmp_path / "results"),
            "schema": {
                "unit_col": "unit",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes": [{"outcome_col": "y", "label": "total"}],
            },
        },
        "model": {
            "types": {
                "a": {"groups": ["total"], "ranks_to_test": [1]},
                "b": {"groups": ["total"], "ranks_to_test": [1]},
            }
        },
        "mcmc": {"num_chains": 1},
    }


def test_main_runs_all_types_when_no_type_flag_given(monkeypatch, tmp_path):
    """Baseline: omitting --type runs every configured model type."""
    import yaml

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(_minimal_config(tmp_path)))

    called_types = []
    monkeypatch.setattr(
        pipeline,
        "run_model_type",
        lambda **kwargs: called_types.append(kwargs["model_type"]),
    )
    monkeypatch.setattr("sys.argv", ["bpnmf", "run", "--config", str(config_path)])

    run_analysis.main()

    assert sorted(called_types) == ["a", "b"]


def test_main_runs_only_requested_type_when_type_flag_given(monkeypatch, tmp_path):
    """--type=a restricts the run to only the requested model type."""
    import yaml

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(_minimal_config(tmp_path)))

    called_types = []
    monkeypatch.setattr(
        pipeline,
        "run_model_type",
        lambda **kwargs: called_types.append(kwargs["model_type"]),
    )
    monkeypatch.setattr(
        "sys.argv", ["bpnmf", "run", "--config", str(config_path), "--type", "a"]
    )

    run_analysis.main()

    assert called_types == ["a"]


def test_main_raises_config_error_when_requested_type_not_found(monkeypatch, tmp_path):
    """--type=nonexistent raises ConfigError naming the available types."""
    import yaml

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(_minimal_config(tmp_path)))

    monkeypatch.setattr(
        "sys.argv",
        ["bpnmf", "run", "--config", str(config_path), "--type", "nonexistent"],
    )

    with pytest.raises(ConfigError, match="nonexistent"):
        run_analysis.main()
