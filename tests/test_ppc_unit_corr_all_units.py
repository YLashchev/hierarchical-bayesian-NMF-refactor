"""Unit-correlation PPC must use ALL units, ignoring ppc_units/ppc_exclude_units.

The spectral norm of the cross-unit residual correlation matrix is undefined
with <2 units. Upstream runs this check on the full frame while filtering only
the per-unit checks (rmse/abs/acf) to ppc_states. Regression guard for the
"insufficient data for spectral norm" bug when ppc_units narrowed the set.
"""

import numpy as np
import pandas as pd

from bayesian_panel_nmf.plots import make_all_ppc_plots


def _draws(n_units: int, n_time: int = 8, n_draws: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    times = pd.date_range("2016-01-01", periods=n_time, freq="2MS")
    for u in range(n_units):
        for d in range(n_draws):
            for t in times:
                mu = np.log(100.0)
                rows.append(
                    {
                        "unit": f"U{u}",
                        "group": "total",
                        "time": t,
                        "treatment": 0,
                        ".draw": d,
                        "outcome": 100 + rng.normal(),
                        "ypred": 100 + rng.normal(),
                        "mu": mu,
                        "mu_treated": mu,
                    }
                )
    return pd.DataFrame(rows)


def test_unit_corr_runs_on_all_units_despite_ppc_units_of_one():
    """ppc_units=[one unit] must NOT starve the cross-unit correlation check."""
    df = _draws(n_units=4)
    results = make_all_ppc_plots(
        df,
        output_dir=None,
        acf_lags=[1],
        ppc_units=["U0"],  # narrows per-unit checks; corr must ignore it
    )
    # corr PPC saw all 4 units -> at least one category p-value, not the
    # empty "insufficient data" placeholder frame.
    assert not results["unit_corr"]["pvals"].empty
