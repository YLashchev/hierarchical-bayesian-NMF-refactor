"""Tests for unbalanced panel handling in _build_model_arrays.

Verifies that structurally absent (group, unit, time) cells are marked as
missing rather than fabricated as observed zeros, and that the
``allow_unbalanced_panel`` config flag controls error vs warning behaviour.
"""

from pathlib import Path

import pandas as pd
import pytest
from loguru import logger

from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.validation import DataError


def _make_config(csv_path: Path, *, allow_unbalanced: bool = False) -> dict:
    return {
        "data": {
            "input_file": str(csv_path),
            "output_dir": "results/test",
            "schema": {
                "unit_col": "state",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes": [
                    {
                        "outcome_col": "outcome",
                        "denominator_col": "pop",
                        "label": "total",
                    },
                ],
            },
            "aggregation": {"enabled": False},
            "allow_unbalanced_panel": allow_unbalanced,
        }
    }


class TestUnbalancedPanel:
    def test_unbalanced_panel_raises_by_default(self, tmp_path: Path) -> None:
        """Without allow_unbalanced_panel, unbalanced data raises DataError."""
        df = pd.DataFrame(
            {
                "state": ["A", "A", "B", "B", "C"],
                "time": [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-01-01",
                    "2020-02-01",
                    "2020-02-01",  # C only has Feb
                ],
                "treated": [0, 1, 0, 0, 0],
                "outcome": [10, 12, 20, 18, 150],
                "pop": [100, 110, 200, 190, 1500],
            }
        )
        csv_path = tmp_path / "unbalanced.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(DataError, match="Unbalanced panel"):
            load_and_prepare(
                filepath=str(csv_path),
                config=_make_config(csv_path, allow_unbalanced=False),
                groups=["total"],
            )

    def test_absent_cell_marked_missing(self, tmp_path: Path) -> None:
        """A cell with no row should be missing_idx=True, not observed zero."""
        # State C has no January row
        df = pd.DataFrame(
            {
                "state": ["A", "A", "B", "B", "C"],
                "time": [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-01-01",
                    "2020-02-01",
                    "2020-02-01",  # C only has Feb
                ],
                "treated": [0, 1, 0, 0, 0],
                "outcome": [10, 12, 20, 18, 150],
                "pop": [100, 110, 200, 190, 1500],
            }
        )
        csv_path = tmp_path / "unbalanced.csv"
        df.to_csv(csv_path, index=False)

        result = load_and_prepare(
            filepath=str(csv_path),
            config=_make_config(csv_path, allow_unbalanced=True),
            groups=["total"],
        )

        # K=1 (total), D=3 (A,B,C), N=2 (Jan,Feb)
        # Units sorted: A=0, B=1, C=2. Times sorted: Jan=0, Feb=1
        missing = result["missing_idx_array"]
        Y = result["Y"]

        # C, Jan (index [0, 2, 0]) should be missing
        assert missing[0, 2, 0], "Absent cell should be marked missing"
        assert Y[0, 2, 0] == 0, "Absent cell Y should be 0"

        # C, Feb (index [0, 2, 1]) should be observed
        assert not missing[0, 2, 1], "Present cell should not be marked missing"
        assert Y[0, 2, 1] == 150

        # All other cells should be observed
        assert not missing[0, 0, 0]  # A, Jan
        assert not missing[0, 0, 1]  # A, Feb
        assert not missing[0, 1, 0]  # B, Jan
        assert not missing[0, 1, 1]  # B, Feb

    def test_balanced_panel_no_missing(self, tmp_path: Path) -> None:
        """A balanced panel should have no missing cells."""
        df = pd.DataFrame(
            {
                "state": ["A", "A", "B", "B"],
                "time": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"],
                "treated": [0, 1, 0, 0],
                "outcome": [10, 12, 20, 18],
                "pop": [100, 110, 200, 190],
            }
        )
        csv_path = tmp_path / "balanced.csv"
        df.to_csv(csv_path, index=False)

        result = load_and_prepare(
            filepath=str(csv_path),
            config=_make_config(csv_path, allow_unbalanced=False),
            groups=["total"],
        )

        assert result["missing_idx_array"].sum() == 0

    def test_nan_outcome_still_marked_missing(self, tmp_path: Path) -> None:
        """A row with NaN outcome (suppressed count) should be missing_idx=True."""
        df = pd.DataFrame(
            {
                "state": ["A", "A"],
                "time": ["2020-01-01", "2020-02-01"],
                "treated": [0, 1],
                "outcome": [float("nan"), 12],
                "pop": [100, 110],
            }
        )
        csv_path = tmp_path / "nan_outcome.csv"
        df.to_csv(csv_path, index=False)

        result = load_and_prepare(
            filepath=str(csv_path),
            config=_make_config(csv_path, allow_unbalanced=False),
            groups=["total"],
        )

        # A, Jan has NaN outcome → missing
        assert result["missing_idx_array"][0, 0, 0]
        assert result["Y"][0, 0, 0] == 0

        # A, Feb has real outcome → observed
        assert not result["missing_idx_array"][0, 0, 1]
        assert result["Y"][0, 0, 1] == 12

    def test_unbalanced_panel_warns_when_allowed(self, tmp_path: Path) -> None:
        """Unbalanced panel should emit a warning when allow_unbalanced_panel=True."""
        df = pd.DataFrame(
            {
                "state": ["A", "A", "B"],
                "time": ["2020-01-01", "2020-02-01", "2020-01-01"],
                "treated": [0, 1, 0],
                "outcome": [10, 12, 20],
                "pop": [100, 110, 200],
            }
        )
        csv_path = tmp_path / "unbalanced_warn.csv"
        df.to_csv(csv_path, index=False)

        # Capture loguru output via a custom sink
        warnings: list[str] = []
        sink_id = logger.add(warnings.append, level="WARNING")
        try:
            load_and_prepare(
                filepath=str(csv_path),
                config=_make_config(csv_path, allow_unbalanced=True),
                groups=["total"],
            )
        finally:
            logger.remove(sink_id)

        warning_text = "".join(warnings)
        assert "Unbalanced panel" in warning_text
        assert "1 of 4" in warning_text  # 1 absent cell out of 2×2=4
