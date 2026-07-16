import matplotlib.pyplot as plt
import pandas as pd
import pytest

from bayesian_panel_nmf.plots import (
    _detect_outcome_column,
    make_interval_plot,
    make_summary_table,
    make_unit_fit_plot,
    make_unit_gap_plot,
)


def test_prefers_standard_outcome():
    df = pd.DataFrame({"outcome": [1], "births": [2]})
    assert _detect_outcome_column(df) == "outcome"


@pytest.mark.parametrize("legacy", ["births", "count", "y"])
def test_falls_back_to_legacy(legacy):
    df = pd.DataFrame({legacy: [1], "unit": ["a"]})
    assert _detect_outcome_column(df) == legacy


def test_raises_when_absent():
    with pytest.raises(ValueError, match="outcome column"):
        _detect_outcome_column(pd.DataFrame({"unit": ["a"]}))


def test_unit_plots_fall_back_to_standard_outcome_when_requested_column_missing():
    df = pd.DataFrame(
        {
            "unit": ["A", "A"],
            "group": ["total", "total"],
            "time": pd.date_range("2020-01-01", periods=2),
            "outcome": [10.0, 12.0],
            "ypred_mean": [9.0, 11.0],
            "ypred_lower": [8.0, 10.0],
            "ypred_upper": [10.0, 12.0],
        }
    )

    for plot_func in (make_unit_fit_plot, make_unit_gap_plot):
        fig, _ = plot_func(df, "A", outcome_col="missing")
        plt.close(fig)


def test_interval_plot_falls_back_to_standard_outcome_when_requested_column_missing():
    df = pd.DataFrame(
        {
            ".draw": [1, 2],
            "unit": ["A", "A"],
            "group": ["total", "total"],
            "treatment": [1, 1],
            "outcome": [10.0, 12.0],
            "ypred": [9.0, 11.0],
            "denominator": [100.0, 100.0],
        }
    )

    fig, _ = make_interval_plot(df, outcome_col="missing", method="pred")
    plt.close(fig)


def test_summary_table_falls_back_to_standard_outcome_when_requested_column_missing():
    df = pd.DataFrame(
        {
            ".draw": [1, 2],
            "unit": ["A", "A"],
            "group": ["total", "total"],
            "treatment": [1, 1],
            "outcome": [10.0, 12.0],
            "ypred": [9.0, 11.0],
            "denominator": [100.0, 100.0],
            "mu": [2.0, 2.0],
            "mu_treated": [2.1, 2.1],
        }
    )

    table = make_summary_table(df, "A", outcome_col="missing")

    assert table.loc[0, "Observed"] == 11
