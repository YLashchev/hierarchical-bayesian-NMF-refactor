"""Tests for duplicate (group, unit, time) detection in _build_model_arrays."""

from pathlib import Path

import pandas as pd
import pytest

from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.validation import DataError


@pytest.fixture
def panel_csv_with_dupes(tmp_path: Path) -> Path:
    """Panel data with a duplicate (group, unit, time) row."""
    df = pd.DataFrame(
        {
            "state": ["X", "X", "X", "Y", "Y"],
            "time": [
                "2020-01-01",
                "2020-01-01",  # duplicate for X, Jan
                "2020-02-01",
                "2020-01-01",
                "2020-02-01",
            ],
            "treated": [0, 0, 1, 0, 0],
            "outcome": [10, 99, 12, 20, 18],
            "pop": [100, 999, 110, 200, 190],
        }
    )
    path = tmp_path / "panel_dupes.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def panel_csv_clean(tmp_path: Path) -> Path:
    """Panel data with no duplicates."""
    df = pd.DataFrame(
        {
            "state": ["X", "X", "Y", "Y"],
            "time": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"],
            "treated": [0, 1, 0, 0],
            "outcome": [10, 12, 20, 18],
            "pop": [100, 110, 200, 190],
        }
    )
    path = tmp_path / "panel_clean.csv"
    df.to_csv(path, index=False)
    return path


class TestDuplicateDetection:
    def test_duplicates_raise_data_error(
        self, panel_csv_with_dupes: Path, make_data_config
    ) -> None:
        config = make_data_config(panel_csv_with_dupes)

        with pytest.raises(DataError, match="duplicate.*group.*unit.*time"):
            load_and_prepare(str(panel_csv_with_dupes), config, groups=["total"])

    def test_clean_data_passes(self, panel_csv_clean: Path, make_data_config) -> None:
        config = make_data_config(panel_csv_clean)

        result = load_and_prepare(str(panel_csv_clean), config, groups=["total"])

        assert result["Y"].shape == (1, 2, 2)

    def test_error_message_shows_examples(
        self, panel_csv_with_dupes: Path, make_data_config
    ) -> None:
        config = make_data_config(panel_csv_with_dupes)

        with pytest.raises(DataError, match="X") as exc_info:
            load_and_prepare(str(panel_csv_with_dupes), config, groups=["total"])

        # Error message should mention the duplicate unit
        assert "X" in str(exc_info.value)
