from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from rich.console import Console

import numpy as np

from bayesian_panel_nmf import visualization
from bayesian_panel_nmf.reporting import (
    _compute_per_unit_post_treatment,
    _print_rich_tables,
    generate_reports,
)
from bayesian_panel_nmf.visualization import make_abs_ppc_plot, make_all_ppc_plots


def _ppc_df() -> pd.DataFrame:
    rows = []
    times = pd.date_range("2020-01-01", periods=8, freq="MS")
    for draw in [1, 2, 3]:
        for unit in ["A", "B", "C", "Agg"]:
            for idx, time in enumerate(times):
                treated = int(unit in {"A", "B", "Agg"} and idx >= 6)
                rows.append(
                    {
                        ".draw": draw,
                        ".chain": 1,
                        ".iteration": draw,
                        "unit": unit,
                        "time": time,
                        "group": "g",
                        "outcome": 10 + draw + idx,
                        "ypred": 9 + draw + idx,
                        "denominator": 100,
                        "treatment": treated,
                        "mu": 2.0,
                        "mu_treated": 2.1,
                    }
                )
    return pd.DataFrame(rows)


def test_ppc_units_filter_stats_to_configured_units() -> None:
    df = _ppc_df()

    fig, pvals = make_abs_ppc_plot(df, ppc_units=["A", "Agg"])
    plt.close(fig)

    assert set(pvals["unit"]) == {"A", "Agg"}


def test_ppc_units_empty_list_warns() -> None:
    df = _ppc_df()

    with pytest.warns(UserWarning, match="empty list"):
        fig, pvals = make_abs_ppc_plot(df, ppc_units=[])
    plt.close(fig)

    assert pvals.empty


def test_ppc_units_missing_warn_and_skip() -> None:
    df = _ppc_df()

    with pytest.warns(UserWarning, match="ppc_units missing"):
        fig, pvals = make_abs_ppc_plot(df, ppc_units=["A", "Missing"])
    plt.close(fig)

    assert set(pvals["unit"]) == {"A"}


def test_make_all_ppc_plots_acf_lags_write_expected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _ppc_df()

    def fake_abs(*args, **kwargs):
        fig, _ = plt.subplots()
        return fig, pd.DataFrame({"unit": ["A"], "group": ["g"], "pval": [0.1]})

    def fake_acf(*args, **kwargs):
        fig, _ = plt.subplots()
        lag = kwargs["lag"]
        return fig, pd.DataFrame({"unit": ["A"], "group": ["g"], "pval": [lag / 10]})

    def fake_rmse(*args, **kwargs):
        fig, _ = plt.subplots()
        return fig, pd.DataFrame({"unit": ["A"], "group": ["g"], "pval": [0.2]})

    def fake_corr(*args, **kwargs):
        fig, _ = plt.subplots()
        return fig, pd.DataFrame({"group": ["g"], "pval": [0.3]})

    monkeypatch.setattr(visualization, "make_abs_ppc_plot", fake_abs)
    monkeypatch.setattr(visualization, "make_acf_ppc_plot", fake_acf)
    monkeypatch.setattr(visualization, "make_rmse_ppc_plot", fake_rmse)
    monkeypatch.setattr(visualization, "make_unit_corr_ppc_plot", fake_corr)

    results = make_all_ppc_plots(df, output_dir=str(tmp_path), acf_lags=[1, 3, 6])

    assert {"acf_lag1", "acf_lag3", "acf_lag6"}.issubset(results)
    assert (tmp_path / "ppc_acf_lag1.png").exists()
    assert (tmp_path / "ppc_acf_lag3.png").exists()
    assert (tmp_path / "ppc_acf_lag6.png").exists()
    pvals = pd.read_csv(tmp_path / "ppc_pvalues.csv")
    assert set(pvals["check_type"]).issuperset({"acf_lag1", "acf_lag3", "acf_lag6"})
    assert "acf" not in set(pvals["check_type"])

    for item in results.values():
        plt.close(item["fig"])


def test_rich_table_prints_group_column(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []

    def fake_print(self, obj, *args, **kwargs):
        captured.append(obj)

    monkeypatch.setattr(Console, "print", fake_print)

    summary = pd.DataFrame({"Group": ["g"], "Observed": [1]})
    per_unit = pd.DataFrame(
        {
            "unit": ["A"],
            "group": ["g"],
            "n_periods": [2],
            "observed": [10.0],
            "expected_mean": [8.0],
            "expected_lower_95": [7.0],
            "expected_upper_95": [9.0],
            "excess_mean": [2.0],
            "excess_lower_95": [1.0],
            "excess_upper_95": [3.0],
            "excess_pct_mean": [25.0],
            "excess_pct_lower_95": [12.5],
            "excess_pct_upper_95": [37.5],
        }
    )
    draws = pd.DataFrame(
        {
            ".draw": [1, 2],
            "unit": ["A", "A"],
            "treatment": [1, 1],
            "outcome": [10.0, 10.0],
            "ypred": [8.0, 9.0],
        }
    )

    _print_rich_tables(summary, per_unit, draws, "A")

    # Only two tables now: headline summary + per-unit totals (Table 3 removed)
    assert len(captured) == 2
    table = captured[1]
    assert any(column.header == "Group" for column in table.columns)
    assert "g" in str(table.rows[0])


def test_per_unit_post_treatment_uses_mu_not_ypred(tmp_path: Path) -> None:
    """_compute_per_unit_post_treatment must use exp(mu) as expected, not ypred."""
    rows = []
    for draw in range(1, 21):
        for unit in ["A", "B"]:
            for t_idx in range(3):
                mu_val = np.log(100.0)  # exp(mu) = 100
                ypred_val = 500.0  # deliberately wrong if ypred were used
                rows.append(
                    {
                        ".draw": draw,
                        ".chain": 1,
                        ".iteration": draw,
                        "unit": unit,
                        "time": pd.Timestamp(f"2022-0{t_idx + 1}-01"),
                        "group": "g",
                        "outcome": 110.0,
                        "denominator": 1000.0,
                        "treatment": 1,
                        "mu": mu_val,
                        "mu_treated": mu_val + 0.1,
                        "ypred": ypred_val,
                    }
                )
    draws = pd.DataFrame(rows)

    result = _compute_per_unit_post_treatment(draws, tmp_path / "out.csv")

    # expected_mean should be close to 3 * 100 = 300 (3 periods * exp(mu)=100)
    # NOT close to 3 * 500 = 1500 (ypred)
    assert result.loc[result["unit"] == "A", "expected_mean"].values[
        0
    ] == pytest.approx(300.0)
    # observed should be sum of outcome = 3 * 110 = 330
    assert result.loc[result["unit"] == "A", "observed"].values[0] == pytest.approx(
        330.0
    )
    assert result.loc[result["unit"] == "A", "n_periods"].values[0] == 3
    # excess_pct has proper posterior CI (draw-by-draw, not point estimate)
    assert "excess_pct_mean" in result.columns
    assert "excess_pct_lower_95" in result.columns
    assert "excess_pct_upper_95" in result.columns
    # CSV written
    assert (tmp_path / "out.csv").exists()


def test_generate_reports_writes_per_treated_state_summaries(tmp_path: Path) -> None:
    """generate_reports must write summary_table_<state>.csv for each treated unit."""
    rows = []
    times = pd.date_range("2020-01-01", periods=4, freq="MS")
    for draw in [1, 2]:
        for unit in ["Texas", "Alabama", "Control"]:
            for idx, time in enumerate(times):
                treated = int(unit in {"Texas", "Alabama"} and idx >= 2)
                rows.append(
                    {
                        ".draw": draw,
                        ".chain": 1,
                        ".iteration": draw,
                        "unit": unit,
                        "time": time,
                        "group": "total",
                        "outcome": 100.0 + idx,
                        "ypred": 95.0 + idx,
                        "denominator": 1000.0,
                        "treatment": treated,
                        "mu": 4.5,
                        "mu_treated": 4.6,
                    }
                )
    draws = pd.DataFrame(rows)

    result = generate_reports(
        draws,
        output_dir=tmp_path,
        target_unit="Texas",
        print_tables=False,
    )

    # Should return treated_units list
    assert set(result["treated_units"]) == {"Texas", "Alabama"}

    # Main summary table for target_unit
    assert (tmp_path / "figs" / "summary_table.csv").exists()

    # Per-treated-state tables
    assert (tmp_path / "figs" / "summary_table_texas.csv").exists()
    assert (tmp_path / "figs" / "summary_table_alabama.csv").exists()


def test_print_target_table_false_skips_headline_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When print_target_table=False, only Table 2 (per-unit totals) prints."""
    captured = []

    def fake_print(self, obj, *args, **kwargs):
        captured.append(obj)

    monkeypatch.setattr(Console, "print", fake_print)

    summary = pd.DataFrame({"Group": ["g"], "Observed": [1]})
    per_unit = pd.DataFrame(
        {
            "unit": ["A"],
            "group": ["g"],
            "n_periods": [2],
            "observed": [10.0],
            "expected_mean": [8.0],
            "expected_lower_95": [7.0],
            "expected_upper_95": [9.0],
            "excess_mean": [2.0],
            "excess_lower_95": [1.0],
            "excess_upper_95": [3.0],
            "excess_pct_mean": [25.0],
            "excess_pct_lower_95": [12.5],
            "excess_pct_upper_95": [37.5],
        }
    )
    draws = pd.DataFrame(
        {
            ".draw": [1, 2],
            "unit": ["A", "A"],
            "treatment": [1, 1],
            "outcome": [10.0, 10.0],
            "ypred": [8.0, 9.0],
        }
    )

    _print_rich_tables(summary, per_unit, draws, "A", print_target_table=False)

    # Only Table 2 should print
    assert len(captured) == 1
    table = captured[0]
    assert table.title == "Post-treatment totals by unit (ranked by % excess)"
