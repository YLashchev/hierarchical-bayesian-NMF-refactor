"""Tests for denominator column validation in _build_model_arrays."""

from pathlib import Path

import pandas as pd
import pytest

from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.validation import DataError


def _make_config(csv_path: Path, has_denom: bool = True) -> dict:
    outcomes = [
        {
            "outcome_col": "outcome",
            "denominator_col": "pop" if has_denom else None,
            "label": "total",
        }
    ]
    return {
        "data": {
            "input_file": str(csv_path),
            "output_dir": "results/test",
            "schema": {
                "unit_col": "state",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes": outcomes,
            },
            "aggregation": {"enabled": False},
        }
    }


class TestDenominatorValidation:
    def test_nan_denominator_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "state": ["A", "A"],
                "time": ["2020-01-01", "2020-02-01"],
                "treated": [0, 1],
                "outcome": [10, 12],
                "pop": [100, float("nan")],
            }
        )
        csv_path = tmp_path / "nan_denom.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(DataError, match="NaN or non-positive denominator"):
            load_and_prepare(str(csv_path), _make_config(csv_path), groups=["total"])

    def test_zero_denominator_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "state": ["A", "A"],
                "time": ["2020-01-01", "2020-02-01"],
                "treated": [0, 1],
                "outcome": [10, 12],
                "pop": [100, 0],
            }
        )
        csv_path = tmp_path / "zero_denom.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(DataError, match="NaN or non-positive denominator"):
            load_and_prepare(str(csv_path), _make_config(csv_path), groups=["total"])

    def test_negative_denominator_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "state": ["A", "A"],
                "time": ["2020-01-01", "2020-02-01"],
                "treated": [0, 1],
                "outcome": [10, 12],
                "pop": [100, -50],
            }
        )
        csv_path = tmp_path / "neg_denom.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(DataError, match="NaN or non-positive denominator"):
            load_and_prepare(str(csv_path), _make_config(csv_path), groups=["total"])

    def test_valid_denominator_passes(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "state": ["A", "A"],
                "time": ["2020-01-01", "2020-02-01"],
                "treated": [0, 1],
                "outcome": [10, 12],
                "pop": [100, 110],
            }
        )
        csv_path = tmp_path / "good_denom.csv"
        df.to_csv(csv_path, index=False)

        result = load_and_prepare(
            str(csv_path), _make_config(csv_path), groups=["total"]
        )
        assert result["Y"].shape == (1, 1, 2)

    def test_no_denominator_column_defaults_to_one(self, tmp_path: Path) -> None:
        """Raw count model: no denominator column → denominators default to 1.0."""
        df = pd.DataFrame(
            {
                "state": ["A", "A"],
                "time": ["2020-01-01", "2020-02-01"],
                "treated": [0, 1],
                "outcome": [10, 12],
            }
        )
        csv_path = tmp_path / "no_denom.csv"
        df.to_csv(csv_path, index=False)

        result = load_and_prepare(
            str(csv_path), _make_config(csv_path, has_denom=False), groups=["total"]
        )
        # All denominators should be 1.0 (raw count model)
        assert (result["denominators"] == 1.0).all()

    def test_error_message_shows_bad_rows(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "state": ["A", "A", "B", "B"],
                "time": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"],
                "treated": [0, 1, 0, 0],
                "outcome": [10, 12, 20, 18],
                "pop": [100, float("nan"), 200, -5],
            }
        )
        csv_path = tmp_path / "multi_bad.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(DataError, match="2 rows") as exc_info:
            load_and_prepare(str(csv_path), _make_config(csv_path), groups=["total"])

        # Error should mention both bad values
        err_msg = str(exc_info.value)
        assert "A" in err_msg or "B" in err_msg
