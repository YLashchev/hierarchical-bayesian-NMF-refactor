"""Unit tests for the opt-in Parquet draws-artifact helpers (Phase 5b).

Additive feature: ``output.draws_format`` defaults to ``"csv"``; these tests
exercise ``_write_draws``/``_read_draws`` in ``bayesian_panel_nmf.pipeline``
directly against a tiny synthetic frame -- no MCMC run required.
"""

from pathlib import Path

import pandas as pd
import pytest

from bayesian_panel_nmf import pipeline as _pipeline_module
from bayesian_panel_nmf.config import Config

ROOT = Path(__file__).resolve().parents[1]


def _load_run_analysis():
    return _pipeline_module


@pytest.fixture(scope="module")
def ra():
    return _load_run_analysis()


@pytest.fixture
def tiny_draws() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["g0", "g0", "g1", "g1"],
            "unit": ["u0", "u1", "u0", "u1"],
            "time": [0, 1, 0, 1],
            ".chain": [1, 1, 2, 2],
            ".draw": [1, 2, 1, 2],
            "mu": [1.5, 2.5, 3.5, 4.5],
        }
    )


def test_output_config_draws_format_defaults_to_csv():
    config = Config.model_validate(
        {
            "data": {
                "input_file": "unused.csv",
                "output_dir": "unused",
                "schema": {
                    "unit_col": "state",
                    "time_col": "time",
                    "treatment_col": "exposed",
                    "outcomes": [{"outcome_col": "births_total", "label": "total"}],
                },
            },
            "model": {
                "outcome_distribution": "NB",
                "types": {"total": {"groups": ["total"], "ranks_to_test": [1]}},
            },
            "output": {},
        }
    )
    assert config.output.draws_format == "csv"


def test_write_draws_csv_writes_csv_and_returns_that_path(ra, tmp_path, tiny_draws):
    stem = tmp_path / "draws"
    path = ra._write_draws(tiny_draws, stem, "csv")
    assert path == stem.with_suffix(".csv")
    assert path.exists()
    assert not stem.with_suffix(".parquet").exists()


def test_write_draws_parquet_writes_parquet(ra, tmp_path, tiny_draws):
    stem = tmp_path / "draws"
    path = ra._write_draws(tiny_draws, stem, "parquet")
    assert path == stem.with_suffix(".parquet")
    assert path.exists()
    assert not stem.with_suffix(".csv").exists()


def test_parquet_round_trips_via_read_draws(ra, tmp_path, tiny_draws):
    stem = tmp_path / "draws"
    ra._write_draws(tiny_draws, stem, "parquet")
    reloaded = ra._read_draws(stem)
    pd.testing.assert_frame_equal(reloaded, tiny_draws)


def test_read_draws_prefers_parquet_when_both_exist(ra, tmp_path, tiny_draws):
    stem = tmp_path / "draws"
    ra._write_draws(tiny_draws, stem, "csv")
    other = tiny_draws.copy()
    other["mu"] = other["mu"] + 100.0
    ra._write_draws(other, stem, "parquet")
    reloaded = ra._read_draws(stem)
    pd.testing.assert_frame_equal(reloaded, other)


def test_read_draws_falls_back_to_csv_when_only_csv_exists(ra, tmp_path, tiny_draws):
    stem = tmp_path / "draws"
    ra._write_draws(tiny_draws, stem, "csv")
    reloaded = ra._read_draws(stem)
    pd.testing.assert_frame_equal(reloaded, tiny_draws)


def test_read_draws_raises_when_neither_exists(ra, tmp_path):
    stem = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        ra._read_draws(stem)
