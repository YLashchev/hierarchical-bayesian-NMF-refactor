import importlib.util
import sys
from pathlib import Path

import pytest
from loguru import logger

from bayesian_panel_nmf.inference import generate_predictions
from bayesian_panel_nmf.logging_config import setup_logging
from bayesian_panel_nmf.models import model
from bayesian_panel_nmf.parallel import (
    get_requested_analysis_workers,
    resolve_analysis_workers,
)
from bayesian_panel_nmf.validation import ConfigError
from bayesian_panel_nmf.validation import DataError


RUN_ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_analysis.py"
RUN_ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "test_run_analysis_module", RUN_ANALYSIS_PATH
)
assert RUN_ANALYSIS_SPEC is not None and RUN_ANALYSIS_SPEC.loader is not None
run_analysis = importlib.util.module_from_spec(RUN_ANALYSIS_SPEC)
sys.modules.setdefault("test_run_analysis_module", run_analysis)
RUN_ANALYSIS_SPEC.loader.exec_module(run_analysis)


def test_run_analysis_config_requires_model_section():
    with pytest.raises(ConfigError, match="config missing 'model' section"):
        run_analysis._validate_run_analysis_config(
            {
                "data": {
                    "schema": {
                        "unit_col": "unit",
                        "time_col": "time",
                        "treatment_col": "treated",
                        "outcomes": [{"outcome_col": "y", "label": "total"}],
                    }
                }
            }
        )


def test_run_analysis_config_requires_model_types():
    with pytest.raises(ConfigError, match="missing 'types' section"):
        run_analysis._validate_run_analysis_config(
            {
                "data": {
                    "schema": {
                        "unit_col": "unit",
                        "time_col": "time",
                        "treatment_col": "treated",
                        "outcomes": [{"outcome_col": "y", "label": "total"}],
                    }
                },
                "model": {},
            }
        )


def test_get_requested_analysis_workers_defaults_to_one():
    assert get_requested_analysis_workers({}) == 1


def test_get_requested_analysis_workers_supports_legacy_num_workers():
    config = {"parallel": {"num_workers": 3}}
    assert get_requested_analysis_workers(config) == 3


def test_resolve_analysis_workers_caps_by_chain_count():
    workers = resolve_analysis_workers(
        requested_workers=4,
        num_chains=4,
        num_model_types=6,
        cpu_count=8,
    )
    assert workers == 2


def test_resolve_analysis_workers_respects_model_type_count():
    workers = resolve_analysis_workers(
        requested_workers=4,
        num_chains=1,
        num_model_types=2,
        cpu_count=8,
    )
    assert workers == 2


def test_resolve_analysis_workers_minus_one_uses_maximum_safe_count():
    workers = resolve_analysis_workers(
        requested_workers=-1,
        num_chains=2,
        num_model_types=10,
        cpu_count=8,
    )
    assert workers == 4


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
            DummyMCMC(),
            data_dict={"denominators": __import__("numpy").ones((1, 1, 1))},
            model_fn=model,
            rank=1,
            config={"mcmc": {"random_seed": 1}, "model": {}},
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


def test_setup_logging_preserves_foreign_handler():
    messages = []
    foreign_id = logger.add(lambda msg: messages.append(str(msg)), level="INFO")
    try:
        setup_logging(level="INFO")
        logger.info("first-message")
        setup_logging(level="DEBUG")
        logger.info("second-message")
    finally:
        logger.remove(foreign_id)

    assert any("first-message" in msg for msg in messages)
    assert any("second-message" in msg for msg in messages)


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
            "df_preprocessed": __import__("pandas").DataFrame(
                [{"unit": "A", "time": "2020-01-01", "group": "total", "outcome": 1}]
            ),
        },
    )

    class DummyMCMC:
        def get_samples(self, group_by_chain=True):
            return {"mu": __import__("numpy").zeros((1, 1, 1, 1))}

    monkeypatch.setattr(run_analysis, "run_mcmc_inference", lambda *a, **k: DummyMCMC())
    monkeypatch.setattr(
        run_analysis, "extract_diagnostics", lambda *a, **k: {"converged": True}
    )
    monkeypatch.setattr(run_analysis, "check_convergence", lambda *a, **k: True)
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

    config = {
        "data": {"input_file": "unused.csv", "output_dir": str(output_dir)},
        "model": {"types": {"total": {"groups": ["total"], "ranks_to_test": [1]}}},
        "output": {"figures": False, "filename_pattern": "{type}_{rank}"},
    }

    run_analysis.run_model_type(
        model_type="total",
        type_config=config["model"]["types"]["total"],
        config=config,
        rank_override=None,
        save_diagnostics=False,
        log_level="INFO",
        configure_logging=False,
        disable_progress_bar=True,
    )

    assert (output_dir / "total" / "df_total.csv").exists()
    assert (output_dir / "total" / "total_1.csv").exists()
