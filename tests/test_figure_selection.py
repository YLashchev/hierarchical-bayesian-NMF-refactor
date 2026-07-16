"""output.figures selection: PLOT_REGISTRY subset gating in generate_reports().

Uses the same tiny synthetic draws frame as tests/test_figures_smoke.py.
"""

import numpy as np
import pandas as pd
import pytest

import bayesian_panel_nmf.reporting as reporting
from bayesian_panel_nmf.config import OutputConfig
from bayesian_panel_nmf.plots import PLOT_REGISTRY


def _draws_frame(units=("u0", "u1", "u2")) -> pd.DataFrame:
    """Minimal multi-unit, multi-draw frame with one treated unit post-t2."""
    rows = []
    times = pd.to_datetime(
        ["2020-01-01", "2020-03-01", "2020-05-01", "2020-07-01", "2020-09-01"]
    )
    rng = np.random.default_rng(0)
    for draw in (1, 2, 3, 4):
        for u in units:
            for t_i, t in enumerate(times):
                treated = int(u == units[0] and t_i >= 2)
                base = 50.0 + draw + rng.normal(scale=0.5)
                rows.append(
                    {
                        ".draw": draw,
                        ".chain": 1,
                        ".iteration": draw,
                        "unit": u,
                        "time": t,
                        "group": "total",
                        "outcome": 50.0 + draw,
                        "denominator": 1000.0,
                        "treatment": treated,
                        "ypred": base,
                        "mu": np.log(50.0),
                        "mu_treated": np.log(50.0) + 0.1 * treated,
                    }
                )
    return pd.DataFrame(rows)


def _run(tmp_path, figures):
    import matplotlib
    import matplotlib.pyplot as plt

    draws_df = _draws_frame()
    try:
        with matplotlib.rc_context():
            reporting.generate_reports(
                draws_df,
                output_dir=tmp_path,
                target_unit="u0",
                print_tables=False,
                figures=figures,
            )
    finally:
        plt.close("all")
    return tmp_path / "figs"


_ALL_FIGURE_PNGS = {
    "fit_u0.png",
    "gap_u0.png",
    "raw_rate.png",
    "interval.png",
    "group_comparison.png",
}


def test_figures_subset_renders_only_selected(tmp_path):
    figs = _run(tmp_path, figures=["interval"])

    assert (figs / "interval.png").exists()
    assert (figs / "interval.png").stat().st_size > 0
    for name in _ALL_FIGURE_PNGS - {"interval.png"}:
        assert not (figs / name).exists(), f"unexpected artifact: {name}"
    assert not (figs / "ppc").exists()

    # Always-on tables still render regardless of figure selection.
    assert (figs / "summary_table.csv").exists()
    assert (figs / "expected_vs_observed.csv").exists()
    assert (figs / "post_treatment_summary.csv").exists()


def test_figures_empty_list_renders_no_figures(tmp_path):
    figs = _run(tmp_path, figures=[])

    for name in _ALL_FIGURE_PNGS:
        assert not (figs / name).exists()
    assert not (figs / "ppc").exists()

    # Tables are still always-on for an explicit empty selection passed
    # directly to generate_reports (run_analysis.py's own gate, tested
    # separately via OutputConfig, is what skips _run_reporting entirely).
    assert (figs / "summary_table.csv").exists()


def test_figures_none_renders_all_registry_entries(tmp_path):
    figs = _run(tmp_path, figures=None)

    for name in _ALL_FIGURE_PNGS:
        assert (figs / name).exists()
        assert (figs / name).stat().st_size > 0
    assert (figs / "ppc" / "ppc_pvalues.csv").exists()


def test_unknown_figure_name_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown name"):
        _run(tmp_path, figures=["not_a_real_figure"])


def test_plot_registry_names_match_config_normalization():
    """OutputConfig.figures=true normalizes to exactly PLOT_REGISTRY's keys."""
    assert set(OutputConfig(figures=True).figures) == set(PLOT_REGISTRY)


def test_output_config_figures_false_and_none_string_both_empty():
    assert OutputConfig(figures=False).figures == []
    assert OutputConfig(figures="none").figures == []


def test_output_config_figures_rejects_unknown_name():
    with pytest.raises(Exception, match="unknown figure name"):
        OutputConfig(figures=["bogus"])
