import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from bayesian_panel_nmf.config import Config
from bayesian_panel_nmf.inference import generate_predictions
from bayesian_panel_nmf.models import model
from bayesian_panel_nmf.validation import ConfigError, DataError

_MINIMAL_SCHEMA = {
    "unit_col": "unit",
    "time_col": "time",
    "treatment_col": "treated",
    "outcomes": [{"outcome_col": "y", "label": "total"}],
}


def _config(**overrides) -> Config:
    """A complete, minimal Config; overrides replace top-level sections."""
    data = {
        "data": {
            "input_file": "unused.csv",
            "output_dir": "unused",
            "schema": _MINIMAL_SCHEMA,
        },
        "model": {"types": {"total": {"groups": ["total"], "ranks_to_test": [1]}}},
    }
    data.update(overrides)
    return Config.model_validate(data)


RUN_ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_analysis.py"
RUN_ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "test_run_analysis_module", RUN_ANALYSIS_PATH
)
assert RUN_ANALYSIS_SPEC is not None and RUN_ANALYSIS_SPEC.loader is not None
run_analysis = importlib.util.module_from_spec(RUN_ANALYSIS_SPEC)
sys.modules.setdefault("test_run_analysis_module", run_analysis)
RUN_ANALYSIS_SPEC.loader.exec_module(run_analysis)


def test_run_analysis_config_requires_model_types():
    """A complete config missing only model.types still raises ConfigError
    from _validate_run_analysis_config's own check (the generic schema
    validation already ran inside Config.model_validate)."""
    config = _config(model={})
    with pytest.raises(ConfigError, match="missing 'types' section"):
        run_analysis._validate_run_analysis_config(config)


def test_safe_rmtree_refuses_project_root(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()

    run_analysis._safe_rmtree(project_root, project_root)

    assert project_root.exists()


def test_safe_rmtree_refuses_path_outside_root(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    run_analysis._safe_rmtree(outside, project_root)

    assert outside.exists()


def test_safe_rmtree_removes_child_directory(tmp_path: Path):
    project_root = tmp_path / "project"
    child = project_root / "results"
    child.mkdir(parents=True)
    (child / "file.txt").write_text("x")

    run_analysis._safe_rmtree(child, project_root)

    assert not child.exists()


def test_generate_predictions_raises_when_samples_not_divisible_by_chains(monkeypatch):
    class DummyMCMC:
        num_chains = 2

        def get_samples(self, group_by_chain=False):
            return {"dummy": 1}

    class DummyPredictive:
        def __init__(self, model_fn, samples):
            self.model_fn = model_fn
            self.samples = samples

        def __call__(self, *args, **kwargs):
            return {"y_obs": __import__("numpy").zeros((3, 1, 1, 1))}

    monkeypatch.setattr("bayesian_panel_nmf.inference.Predictive", DummyPredictive)

    with pytest.raises(DataError, match="not evenly divisible"):
        generate_predictions(
            cast(Any, DummyMCMC()),
            data_dict={"denominators": __import__("numpy").ones((1, 1, 1))},
            model_fn=model,
            rank=1,
            config=_config(mcmc={"random_seed": 1}, model={}),
        )


def test_model_rejects_none_control_idx_when_model_treated_true():
    with pytest.raises(
        ValueError, match="model_treated=True requires control_idx_array"
    ):
        model(
            denominators=__import__("numpy").ones((1, 1, 1)),
            control_idx_array=None,
            missing_idx_array=None,
            model_treated=True,
        )


def test_run_model_type_without_figures_does_not_import_viz(
    monkeypatch, tmp_path: Path
):
    output_dir = tmp_path / "results"

    monkeypatch.setattr(
        run_analysis,
        "load_and_prepare",
        lambda *args, **kwargs: {
            "Y": __import__("numpy").ones((1, 1, 1)),
            "denominators": __import__("numpy").ones((1, 1, 1)),
            "control_idx_array": __import__("numpy").ones((1, 1, 1), dtype=bool),
            "missing_idx_array": __import__("numpy").zeros((1, 1, 1), dtype=bool),
            "groups": ["total"],
            "units": ["A"],
            "times": ["2020-01-01"],
            "df_preprocessed": __import__("pandas").DataFrame(
                [{"unit": "A", "time": "2020-01-01", "group": "total", "outcome": 1}]
            ),
        },
    )

    class DummyMCMC:
        num_chains = 1
        chain_method = "sequential"

        def get_samples(self, group_by_chain=True):
            return {"mu": __import__("numpy").zeros((1, 1, 1, 1))}

        def get_extra_fields(self):
            return {}

    monkeypatch.setattr(run_analysis, "run_mcmc_inference", lambda *a, **k: DummyMCMC())
    monkeypatch.setattr(
        run_analysis,
        "convergence_summary",
        lambda idata: {
            "rhat_max": 1.0,
            "ess_bulk_min": 1000.0,
            "ess_tail_min": 1000.0,
            "divergences": 0,
            "converged": True,
        },
    )
    monkeypatch.setattr(
        run_analysis,
        "generate_predictions",
        lambda *a, **k: __import__("numpy").zeros((1, 1, 1, 1, 1)),
    )
    monkeypatch.setattr(
        run_analysis,
        "format_draws",
        lambda *a, **k: __import__("pandas").DataFrame([{"ok": 1}]),
    )
    monkeypatch.setattr(
        run_analysis,
        "_run_reporting",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("reporting should not run")
        ),
    )

    type_config_dict = {"groups": ["total"], "ranks_to_test": [1]}
    config = _config(
        data={
            "input_file": "unused.csv",
            "output_dir": str(output_dir),
            "schema": _MINIMAL_SCHEMA,
        },
        model={"types": {"total": type_config_dict}},
        output={"figures": False},
    )

    run_analysis.run_model_type(
        model_type="total",
        type_config=config.model.types["total"],
        config=config,
        rank_override=None,
        log_level="INFO",
        configure_logging=False,
    )

    assert (output_dir / "total" / "df_total.csv").exists()
    assert (output_dir / "total" / "NB_births_total_1.csv").exists()


# ---------------------------------------------------------------------------
# _get_outcome_name
# ---------------------------------------------------------------------------


def _data_with_schema(**data_overrides) -> dict:
    data = {
        "input_file": "unused.csv",
        "output_dir": "unused",
        "schema": {
            "unit_col": "unit",
            "time_col": "time",
            "treatment_col": "treated",
            "outcomes_from_prefixes": {"outcome_prefix": "births_"},
        },
    }
    data.update(data_overrides)
    return data


def test_get_outcome_name_explicit_override():
    config = _config(data=_data_with_schema(outcome="deaths"))
    assert run_analysis._get_outcome_name(config) == "deaths"


def test_get_outcome_name_derives_from_prefix_with_underscore():
    config = _config(
        data=_data_with_schema(
            schema={
                "unit_col": "unit",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes_from_prefixes": {"outcome_prefix": "births_"},
            }
        )
    )
    assert run_analysis._get_outcome_name(config) == "births"


def test_get_outcome_name_derives_from_prefix_without_underscore():
    config = _config(
        data=_data_with_schema(
            schema={
                "unit_col": "unit",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes_from_prefixes": {"outcome_prefix": "deaths"},
            }
        )
    )
    assert run_analysis._get_outcome_name(config) == "deaths"


def test_get_outcome_name_fallback_when_neither_set():
    config = _config(
        data=_data_with_schema(
            schema={
                "unit_col": "unit",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes": [{"outcome_col": "y", "label": "total"}],
            }
        )
    )
    assert run_analysis._get_outcome_name(config) == "births"


def test_get_outcome_name_empty_prefix_falls_back():
    config = _config(
        data=_data_with_schema(
            schema={
                "unit_col": "unit",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes_from_prefixes": {"outcome_prefix": ""},
            }
        )
    )
    assert run_analysis._get_outcome_name(config) == "births"


# ---------------------------------------------------------------------------
# _draws_filename
# ---------------------------------------------------------------------------


def test_draws_filename_fixed_scheme():
    config = _config(
        model={"outcome_distribution": "NB"},
        data=_data_with_schema(outcome="births"),
    )
    assert run_analysis._draws_filename(config, "total", 5) == "NB_births_total_5"


def test_draws_filename_default_distribution_and_outcome():
    config = _config()
    assert run_analysis._draws_filename(config, "groups", 3) == "NB_births_groups_3"
