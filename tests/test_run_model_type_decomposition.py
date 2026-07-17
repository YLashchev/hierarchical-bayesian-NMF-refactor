"""Characterization tests for pipeline.py's run_model_type output-dir
cleanup and convergence-gate-failure logging, before extracting the
per-rank pipeline into _run_single_rank and the output-dir prep into
_prepare_type_output_dir (Tier 2b of the repo clarity refactor). These two
behaviors previously had no direct test — only end-to-end coverage of the
happy path via test_run_model_type_without_figures_does_not_import_viz."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_panel_nmf import pipeline as run_analysis
from bayesian_panel_nmf.config import Config


def _config(**overrides) -> Config:
    data = {
        "data": {
            "input_file": "unused.csv",
            "output_dir": "unused",
            "schema": {
                "unit_col": "unit",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes": [{"outcome_col": "y", "label": "total"}],
            },
        },
        "model": {"types": {"total": {"groups": ["total"], "ranks_to_test": [1]}}},
    }
    data.update(overrides)
    return Config.model_validate(data)


class _DummyMCMC:
    num_chains = 1
    chain_method = "sequential"

    def get_samples(self, group_by_chain=True):
        return {"mu": np.zeros((1, 1, 1, 1))}

    def get_extra_fields(self):
        return {}


def _patch_common_collaborators(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_analysis,
        "load_and_prepare",
        lambda *args, **kwargs: {
            "Y": np.ones((1, 1, 1)),
            "denominators": np.ones((1, 1, 1)),
            "control_idx_array": np.ones((1, 1, 1), dtype=bool),
            "missing_idx_array": np.zeros((1, 1, 1), dtype=bool),
            "groups": ["total"],
            "units": ["A"],
            "times": ["2020-01-01"],
            "df_preprocessed": pd.DataFrame(
                [{"unit": "A", "time": "2020-01-01", "group": "total", "outcome": 1}]
            ),
        },
    )
    monkeypatch.setattr(
        run_analysis, "run_mcmc_inference", lambda *a, **k: _DummyMCMC()
    )
    monkeypatch.setattr(
        run_analysis,
        "generate_predictions",
        lambda *a, **k: np.zeros((1, 1, 1, 1, 1)),
    )
    monkeypatch.setattr(
        run_analysis, "format_draws", lambda *a, **k: pd.DataFrame([{"ok": 1}])
    )


def test_clean_true_removes_existing_type_output_dir_before_rerun(
    monkeypatch, tmp_path: Path
):
    """output.clean: true removes a pre-existing type_output_dir subtree
    before writing new results, exercised through run_model_type end to
    end (not just via the isolated _safe_rmtree unit tests)."""
    output_dir = tmp_path / "results"
    _patch_common_collaborators(monkeypatch, tmp_path)
    monkeypatch.setattr(
        run_analysis,
        "convergence_summary",
        lambda idata, params=None: {
            "rhat_max": 1.0,
            "ess_bulk_min": 1000.0,
            "ess_tail_min": 1000.0,
            "divergences": 0,
            "converged": True,
        },
    )
    monkeypatch.setattr(
        run_analysis,
        "_run_reporting",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("reporting should not run")
        ),
    )

    type_output_dir = output_dir / "total"
    type_output_dir.mkdir(parents=True)
    stale_file = type_output_dir / "stale_leftover.csv"
    stale_file.write_text("old data that must be removed")

    type_config_dict = {"groups": ["total"], "ranks_to_test": [1]}
    config = _config(
        data={
            "input_file": "unused.csv",
            "output_dir": str(output_dir),
            "schema": {
                "unit_col": "unit",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes": [{"outcome_col": "y", "label": "total"}],
            },
        },
        model={"types": {"total": type_config_dict}},
        output={"figures": False, "clean": True},
    )

    run_analysis.run_model_type(
        model_type="total",
        type_config=config.model.types["total"],
        config=config,
        rank_override=None,
        log_level="INFO",
        configure_logging=False,
    )

    assert not stale_file.exists()
    assert (output_dir / "total" / "df_total.csv").exists()
    assert (output_dir / "total" / "NB_outcome_total_1.csv").exists()


def test_convergence_gate_failure_logs_warning_but_does_not_abort(
    monkeypatch, tmp_path: Path
):
    """A converged=False gate result logs a warning but still writes the
    convergence JSON and continues the pipeline (draws CSV still written)."""
    output_dir = tmp_path / "results"
    _patch_common_collaborators(monkeypatch, tmp_path)
    monkeypatch.setattr(
        run_analysis,
        "convergence_summary",
        lambda idata, params=None: {
            "rhat_max": 2.5,
            "ess_bulk_min": 3.0,
            "ess_tail_min": 5.0,
            "divergences": 12,
            "converged": False,
        },
    )

    warnings_logged: list[str] = []
    monkeypatch.setattr(
        run_analysis.logger, "warning", lambda msg: warnings_logged.append(msg)
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
            "schema": {
                "unit_col": "unit",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes": [{"outcome_col": "y", "label": "total"}],
            },
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

    # Pipeline must NOT abort: draws CSV still written despite failed gate.
    draws_file = output_dir / "total" / "NB_outcome_total_1.csv"
    assert draws_file.exists()

    convergence_file = output_dir / "total" / "NB_outcome_total_1_convergence.json"
    assert convergence_file.exists()
    written_gate = json.loads(convergence_file.read_text())
    assert written_gate["converged"] is False
    assert written_gate["rhat_max"] == 2.5

    # The failure warning must actually fire, naming the gate metrics.
    assert len(warnings_logged) == 1
    assert "convergence gate FAILED" in warnings_logged[0]
    assert "R-hat=2.5000" in warnings_logged[0]
    assert "divergences=12" in warnings_logged[0]
