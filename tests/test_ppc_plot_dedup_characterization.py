"""Characterization tests pinning current make_acf_ppc_plot and
make_rmse_ppc_plot behavior before extracting their shared setup pipeline
into a common helper (Tier 1a of the repo clarity refactor)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesian_panel_nmf.visualization import (
    make_abs_ppc_plot,
    make_acf_ppc_plot,
    make_rmse_ppc_plot,
)


def _ppc_test_df(n_units: int = 3, n_draws: int = 4, n_times: int = 10) -> pd.DataFrame:
    """Build a minimal draws-shaped DataFrame with deterministic values,
    exercising both control (treatment=0) and treated (treatment=1) rows
    so PPC filtering logic has something to filter."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2020-01-01", periods=n_times, freq="MS")
    unit_names = [f"unit{i}" for i in range(n_units)]
    rows = []
    for draw in range(1, n_draws + 1):
        for unit in unit_names:
            for t_idx, time in enumerate(times):
                treated = int(unit == "unit0" and t_idx >= n_times // 2)
                mu = 2.0 + 0.01 * t_idx
                rows.append(
                    {
                        ".draw": draw,
                        ".chain": 1,
                        ".iteration": draw,
                        "unit": unit,
                        "time": time,
                        "group": "g",
                        "outcome": float(rng.poisson(np.exp(mu))),
                        "ypred": float(rng.poisson(np.exp(mu))),
                        "denominator": 100.0,
                        "treatment": treated,
                        "mu": mu,
                        "mu_treated": mu + 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_make_acf_ppc_plot_pvals_shape_and_columns():
    df = _ppc_test_df()
    fig, pvals = make_acf_ppc_plot(df, lag=2)
    plt.close(fig)

    assert set(pvals.columns) == {"unit", "group", "pval"}
    assert set(pvals["unit"]) == {"unit0"}  # only treated unit0 is a control-period PPC target
    assert (pvals["pval"] >= 0).all() and (pvals["pval"] <= 1).all()


def test_make_acf_ppc_plot_is_deterministic_for_fixed_input():
    """Same input, same lag -> identical pvals (no RNG in this function itself;
    the fixture's RNG is seeded, so repeated calls on the same df must agree)."""
    df = _ppc_test_df()
    fig1, pvals1 = make_acf_ppc_plot(df, lag=3)
    plt.close(fig1)
    fig2, pvals2 = make_acf_ppc_plot(df, lag=3)
    plt.close(fig2)

    pd.testing.assert_frame_equal(
        pvals1.reset_index(drop=True), pvals2.reset_index(drop=True)
    )


def test_make_rmse_ppc_plot_pvals_shape_and_columns():
    df = _ppc_test_df()
    fig, pvals = make_rmse_ppc_plot(df)
    plt.close(fig)

    assert set(pvals.columns) == {"unit", "group", "pval"}
    assert set(pvals["unit"]) == {"unit0"}
    assert (pvals["pval"] >= 0).all() and (pvals["pval"] <= 1).all()


def test_make_rmse_ppc_plot_is_deterministic_for_fixed_input():
    df = _ppc_test_df()
    fig1, pvals1 = make_rmse_ppc_plot(df)
    plt.close(fig1)
    fig2, pvals2 = make_rmse_ppc_plot(df)
    plt.close(fig2)

    pd.testing.assert_frame_equal(
        pvals1.reset_index(drop=True), pvals2.reset_index(drop=True)
    )


def test_make_abs_ppc_plot_pvals_shape_and_columns_baseline():
    """Baseline characterization for make_abs_ppc_plot on the same fixture
    shape used for acf/rmse above, so post-extraction comparison has a
    like-for-like reference across all three functions."""
    df = _ppc_test_df()
    fig, pvals = make_abs_ppc_plot(df)
    plt.close(fig)

    assert set(pvals.columns) == {"unit", "group", "pval"}
    assert set(pvals["unit"]) == {"unit0"}
    assert (pvals["pval"] >= 0).all() and (pvals["pval"] <= 1).all()
