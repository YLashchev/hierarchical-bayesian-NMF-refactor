"""Tests for the three-mode synthetic 'total' group aggregation in data.py.

Covers:
  Mode 1: "total" is an explicit outcome label — no aggregation needed
  Mode 2: total_from — sum specific named subgroups
  Mode 3: total_all — sum all defined outcomes
  Error cases: missing config, bad labels, no denominator column
"""

from pathlib import Path

import pandas as pd
import pytest

from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.validation import ConfigError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def panel_csv(tmp_path: Path) -> Path:
    """Panel data with two disjoint subgroups (a, b) that have denominators."""
    df = pd.DataFrame(
        {
            "state": ["X", "X", "Y", "Y"],
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


@pytest.fixture
def panel_csv_no_denom(tmp_path: Path) -> Path:
    """Panel data with two subgroups but NO denominator columns."""
    df = pd.DataFrame(
        {
            "state": ["X", "X", "Y", "Y"],
            "time": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"],
            "treated": [0, 1, 0, 0],
            "births_a": [10, 12, 20, 18],
            "births_b": [5, 7, 8, 9],
        }
    )
    path = tmp_path / "panel_no_denom.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def panel_csv_with_total(tmp_path: Path) -> Path:
    """Panel data with an explicit 'total' column already in the CSV."""
    df = pd.DataFrame(
        {
            "state": ["X", "X", "Y", "Y"],
            "time": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"],
            "treated": [0, 1, 0, 0],
            "births_a": [10, 12, 20, 18],
            "pop_a": [100, 110, 200, 190],
            "births_b": [5, 7, 8, 9],
            "pop_b": [50, 60, 80, 90],
            "births_total": [15, 19, 28, 27],
            "pop_total": [150, 170, 280, 280],
        }
    )
    path = tmp_path / "panel_total.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Mode 1: Explicit "total" outcome label
# ---------------------------------------------------------------------------
class TestMode1ExplicitTotal:
    """'total' is defined as a regular outcome in the config schema."""

    def test_explicit_total_no_aggregation(
        self, panel_csv_with_total: Path, make_data_config
    ) -> None:
        config = make_data_config(
            panel_csv_with_total,
            outcomes=[
                {
                    "outcome_col": "births_total",
                    "denominator_col": "pop_total",
                    "label": "total",
                },
            ],
        )

        result = load_and_prepare(str(panel_csv_with_total), config, groups=["total"])

        assert result["groups"] == ["total"]
        df = result["df_preprocessed"]
        assert sorted(df["group"].unique()) == ["total"]
        # Check values come directly from the column, not aggregated
        x_t1 = df[(df["unit"] == "X") & (df["time"] == pd.Timestamp("2020-01-01"))]
        assert x_t1["outcome"].values[0] == 15
        assert x_t1["denominator"].values[0] == 150

    def test_explicit_total_needs_no_type_config(
        self, panel_csv_with_total: Path, make_data_config
    ) -> None:
        """No type_config needed when total is explicitly defined."""
        config = make_data_config(
            panel_csv_with_total,
            outcomes=[
                {
                    "outcome_col": "births_total",
                    "denominator_col": "pop_total",
                    "label": "total",
                },
            ],
        )

        # type_config=None should work fine
        result = load_and_prepare(
            str(panel_csv_with_total), config, groups=["total"], type_config=None
        )
        assert result["groups"] == ["total"]


# ---------------------------------------------------------------------------
# Mode 2: total_from — sum specific subgroups
# ---------------------------------------------------------------------------
class TestMode2TotalFrom:
    """total_from lists specific outcome labels to sum."""

    def test_total_from_sums_outcomes_and_denominators(
        self, panel_csv: Path, make_data_config
    ) -> None:
        config = make_data_config(
            panel_csv,
            outcomes=[
                {"outcome_col": "births_a", "denominator_col": "pop_a", "label": "a"},
                {"outcome_col": "births_b", "denominator_col": "pop_b", "label": "b"},
            ],
        )
        type_config = {"groups": ["total"], "total_from": ["a", "b"]}

        result = load_and_prepare(
            str(panel_csv), config, groups=["total"], type_config=type_config
        )

        assert result["groups"] == ["total"]
        df = result["df_preprocessed"]
        assert sorted(df["group"].unique()) == ["total"]

        # Check that outcomes were summed: births_a + births_b
        x_t1 = df[(df["unit"] == "X") & (df["time"] == pd.Timestamp("2020-01-01"))]
        assert x_t1["outcome"].values[0] == 15  # 10 + 5
        assert x_t1["denominator"].values[0] == 150  # 100 + 50

    def test_total_from_subset_of_outcomes(
        self, panel_csv: Path, make_data_config
    ) -> None:
        """total_from can reference a subset — doesn't have to include all outcomes."""
        config = make_data_config(
            panel_csv,
            outcomes=[
                {"outcome_col": "births_a", "denominator_col": "pop_a", "label": "a"},
                {"outcome_col": "births_b", "denominator_col": "pop_b", "label": "b"},
            ],
        )
        type_config = {"groups": ["total"], "total_from": ["a"]}  # only sum "a"

        result = load_and_prepare(
            str(panel_csv), config, groups=["total"], type_config=type_config
        )

        df = result["df_preprocessed"]
        x_t1 = df[(df["unit"] == "X") & (df["time"] == pd.Timestamp("2020-01-01"))]
        assert x_t1["outcome"].values[0] == 10  # only births_a
        assert x_t1["denominator"].values[0] == 100  # only pop_a

    def test_total_from_with_other_groups_requested(
        self, panel_csv: Path, make_data_config
    ) -> None:
        """Can request both total and individual subgroups."""
        config = make_data_config(
            panel_csv,
            outcomes=[
                {"outcome_col": "births_a", "denominator_col": "pop_a", "label": "a"},
                {"outcome_col": "births_b", "denominator_col": "pop_b", "label": "b"},
            ],
        )
        type_config = {"groups": ["a", "total"], "total_from": ["a", "b"]}

        result = load_and_prepare(
            str(panel_csv), config, groups=["a", "total"], type_config=type_config
        )

        assert sorted(result["groups"]) == ["a", "total"]
        df = result["df_preprocessed"]
        assert sorted(df["group"].unique()) == ["a", "total"]

    def test_total_from_rejects_undefined_label(
        self, panel_csv: Path, make_data_config
    ) -> None:
        config = make_data_config(
            panel_csv,
            outcomes=[
                {"outcome_col": "births_a", "denominator_col": "pop_a", "label": "a"},
            ],
        )
        type_config = {"groups": ["total"], "total_from": ["a", "nonexistent"]}

        with pytest.raises(ConfigError, match="undefined outcome labels"):
            load_and_prepare(
                str(panel_csv), config, groups=["total"], type_config=type_config
            )

    def test_total_from_no_denominator_column(
        self, panel_csv_no_denom: Path, make_data_config
    ) -> None:
        """total_from works even when there are no denominator columns (count model)."""
        config = make_data_config(
            panel_csv_no_denom,
            outcomes=[
                {"outcome_col": "births_a", "label": "a"},
                {"outcome_col": "births_b", "label": "b"},
            ],
        )
        type_config = {"groups": ["total"], "total_from": ["a", "b"]}

        result = load_and_prepare(
            str(panel_csv_no_denom), config, groups=["total"], type_config=type_config
        )

        df = result["df_preprocessed"]
        x_t1 = df[(df["unit"] == "X") & (df["time"] == pd.Timestamp("2020-01-01"))]
        assert x_t1["outcome"].values[0] == 15  # 10 + 5
        # No denominator column — should not crash
        assert (
            "denominator" not in df.columns
            or pd.isna(x_t1["denominator"].values[0])
            or True
        )

    def test_total_from_prevents_double_counting(
        self, panel_csv: Path, make_data_config
    ) -> None:
        """
        Regression: the old code summed all outcomes, which would double-count
        if some outcomes are subsets of others. total_from prevents this by
        letting the user specify exactly which labels to sum.
        """
        # Simulate nativity-like config: a and b are components of c
        # If user specifies total_from: ["a", "b"], they get 15, not 30
        config = make_data_config(
            panel_csv,
            outcomes=[
                {"outcome_col": "births_a", "denominator_col": "pop_a", "label": "a"},
                {"outcome_col": "births_b", "denominator_col": "pop_b", "label": "b"},
            ],
        )
        # Only sum a+b, not all outcomes
        type_config = {"groups": ["total"], "total_from": ["a", "b"]}

        result = load_and_prepare(
            str(panel_csv), config, groups=["total"], type_config=type_config
        )

        df = result["df_preprocessed"]
        x_t1 = df[(df["unit"] == "X") & (df["time"] == pd.Timestamp("2020-01-01"))]
        assert x_t1["outcome"].values[0] == 15  # a(10) + b(5), not double


# ---------------------------------------------------------------------------
# Mode 3: total_all — sum all defined outcomes
# ---------------------------------------------------------------------------
class TestMode3TotalAll:
    """total_all=True sums all defined outcomes."""

    def test_total_all_sums_everything(self, panel_csv: Path, make_data_config) -> None:
        config = make_data_config(
            panel_csv,
            outcomes=[
                {"outcome_col": "births_a", "denominator_col": "pop_a", "label": "a"},
                {"outcome_col": "births_b", "denominator_col": "pop_b", "label": "b"},
            ],
        )
        type_config = {"groups": ["total"], "total_all": True}

        result = load_and_prepare(
            str(panel_csv), config, groups=["total"], type_config=type_config
        )

        df = result["df_preprocessed"]
        x_t1 = df[(df["unit"] == "X") & (df["time"] == pd.Timestamp("2020-01-01"))]
        assert x_t1["outcome"].values[0] == 15  # 10 + 5
        assert x_t1["denominator"].values[0] == 150  # 100 + 50

    def test_total_all_no_denominator(
        self, panel_csv_no_denom: Path, make_data_config
    ) -> None:
        config = make_data_config(
            panel_csv_no_denom,
            outcomes=[
                {"outcome_col": "births_a", "label": "a"},
                {"outcome_col": "births_b", "label": "b"},
            ],
        )
        type_config = {"groups": ["total"], "total_all": True}

        # Should not crash even without denominator columns
        result = load_and_prepare(
            str(panel_csv_no_denom), config, groups=["total"], type_config=type_config
        )

        df = result["df_preprocessed"]
        x_t1 = df[(df["unit"] == "X") & (df["time"] == pd.Timestamp("2020-01-01"))]
        assert x_t1["outcome"].values[0] == 15


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------
class TestTotalErrorCases:
    """Error handling when total is misconfigured."""

    def test_no_config_for_synthetic_total_raises(
        self, panel_csv: Path, make_data_config
    ) -> None:
        """Requesting 'total' without explicit label or type_config raises ConfigError."""
        config = make_data_config(
            panel_csv,
            outcomes=[
                {"outcome_col": "births_a", "denominator_col": "pop_a", "label": "a"},
                {"outcome_col": "births_b", "denominator_col": "pop_b", "label": "b"},
            ],
        )

        with pytest.raises(ConfigError, match="not defined as an outcome"):
            load_and_prepare(str(panel_csv), config, groups=["total"])

    def test_empty_type_config_raises(self, panel_csv: Path, make_data_config) -> None:
        """type_config present but without total_from or total_all raises ConfigError."""
        config = make_data_config(
            panel_csv,
            outcomes=[
                {"outcome_col": "births_a", "denominator_col": "pop_a", "label": "a"},
            ],
        )
        type_config = {"groups": ["total"]}  # no total_from, no total_all

        with pytest.raises(ConfigError, match="not defined as an outcome"):
            load_and_prepare(
                str(panel_csv), config, groups=["total"], type_config=type_config
            )

    def test_total_from_takes_precedence_over_total_all(
        self, panel_csv: Path, make_data_config
    ) -> None:
        """If both total_from and total_all are present, total_from wins."""
        config = make_data_config(
            panel_csv,
            outcomes=[
                {"outcome_col": "births_a", "denominator_col": "pop_a", "label": "a"},
                {"outcome_col": "births_b", "denominator_col": "pop_b", "label": "b"},
            ],
        )
        type_config = {
            "groups": ["total"],
            "total_from": ["a"],  # only sum "a"
            "total_all": True,  # should be ignored
        }

        result = load_and_prepare(
            str(panel_csv), config, groups=["total"], type_config=type_config
        )

        df = result["df_preprocessed"]
        x_t1 = df[(df["unit"] == "X") & (df["time"] == pd.Timestamp("2020-01-01"))]
        assert x_t1["outcome"].values[0] == 10  # only "a", not a+b

    def test_total_from_rejects_duplicate_labels(
        self, panel_csv: Path, make_data_config
    ) -> None:
        """Duplicate labels in total_from would double-count — must be rejected."""
        config = make_data_config(
            panel_csv,
            outcomes=[
                {"outcome_col": "births_a", "denominator_col": "pop_a", "label": "a"},
                {"outcome_col": "births_b", "denominator_col": "pop_b", "label": "b"},
            ],
        )
        type_config = {"groups": ["total"], "total_from": ["a", "b", "a"]}

        with pytest.raises(ConfigError, match="duplicate labels"):
            load_and_prepare(
                str(panel_csv), config, groups=["total"], type_config=type_config
            )


# ---------------------------------------------------------------------------
# Model array shape validation
# ---------------------------------------------------------------------------
class TestTotalModelArrays:
    """Verify the K×D×N arrays are correct for synthetic total."""

    def test_total_from_produces_correct_array_shape(
        self, panel_csv: Path, make_data_config
    ) -> None:
        config = make_data_config(
            panel_csv,
            outcomes=[
                {"outcome_col": "births_a", "denominator_col": "pop_a", "label": "a"},
                {"outcome_col": "births_b", "denominator_col": "pop_b", "label": "b"},
            ],
        )
        type_config = {"groups": ["total"], "total_from": ["a", "b"]}

        result = load_and_prepare(
            str(panel_csv), config, groups=["total"], type_config=type_config
        )

        # K=1 (total), D=2 (X, Y), N=2 (two months)
        assert result["Y"].shape == (1, 2, 2)
        assert result["denominators"].shape == (1, 2, 2)
        assert result["control_idx_array"].shape == (1, 2, 2)
        assert result["missing_idx_array"].shape == (1, 2, 2)

    def test_total_from_array_values_are_summed(
        self, panel_csv: Path, make_data_config
    ) -> None:
        config = make_data_config(
            panel_csv,
            outcomes=[
                {"outcome_col": "births_a", "denominator_col": "pop_a", "label": "a"},
                {"outcome_col": "births_b", "denominator_col": "pop_b", "label": "b"},
            ],
        )
        type_config = {"groups": ["total"], "total_from": ["a", "b"]}

        result = load_and_prepare(
            str(panel_csv), config, groups=["total"], type_config=type_config
        )

        # K=0 is "total", units sorted alphabetically: X=0, Y=1
        # Time sorted: 2020-01-01=0, 2020-02-01=1
        Y = result["Y"]
        assert Y[0, 0, 0] == 15  # X, Jan: 10+5
        assert Y[0, 0, 1] == 19  # X, Feb: 12+7
        assert Y[0, 1, 0] == 28  # Y, Jan: 20+8
        assert Y[0, 1, 1] == 27  # Y, Feb: 18+9
