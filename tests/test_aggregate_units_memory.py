"""add_aggregate_units must not mutate the input and must avoid gratuitous
full-frame copies (the cut-reporting OOM driver on large PPC frames)."""

import numpy as np
import pandas as pd

from bayesian_panel_nmf.aggregate_units import add_aggregate_units
from bayesian_panel_nmf.config import AggregateUnitSpec


def _draws(n_units=6, n_time=4, n_draws=3):
    rng = np.random.default_rng(0)
    rows = []
    times = pd.date_range("2016-01-01", periods=n_time, freq="2MS")
    for u in range(n_units):
        for d in range(n_draws):
            for t in times:
                rows.append(
                    {
                        ".draw": d,
                        ".chain": 0,
                        ".iteration": d,
                        "unit": f"U{u}",
                        "time": t,
                        "group": "total",
                        "outcome": 100.0 + rng.normal(),
                        "denominator": 1000.0,
                        "treatment": 1 if u < 3 else 0,
                        "ypred": 100.0 + rng.normal(),
                        "mu": np.log(100.0),
                        "mu_treated": np.log(100.0),
                        "start_date": t,
                        "end_date": t + pd.DateOffset(months=2) - pd.Timedelta(days=1),
                    }
                )
    return pd.DataFrame(rows)


def test_does_not_mutate_input():
    df = _draws()
    before = df.copy(deep=True)
    add_aggregate_units(df, [AggregateUnitSpec(unit="AGG", include_treated_units=True)])
    pd.testing.assert_frame_equal(df, before)  # input untouched


def test_aggregate_rows_appended_correctly():
    df = _draws()
    out = add_aggregate_units(
        df, [AggregateUnitSpec(unit="AGG", include_treated_units=True)]
    )
    assert "AGG" in set(out["unit"])
    # original rows all preserved
    assert len(out) == len(df) + len(out[out["unit"] == "AGG"])
    # aggregate sums the 3 treated units' outcomes per (draw, time)
    d0t0 = out[(out["unit"] == "AGG") & (out[".draw"] == 0)].iloc[0]
    src = df[(df["unit"].isin(["U0", "U1", "U2"])) & (df[".draw"] == 0)
             & (df["time"] == d0t0["time"])]
    assert abs(d0t0["outcome"] - src["outcome"].sum()) < 1e-6


def test_empty_specs_returns_input_unchanged():
    df = _draws()
    out = add_aggregate_units(df, [])
    pd.testing.assert_frame_equal(out, df)


def test_overwrite_replaces_without_mutating_input():
    df = _draws()
    before = df.copy(deep=True)
    specs = [
        AggregateUnitSpec(unit="U0", include_treated_units=True, overwrite=True),
    ]
    out = add_aggregate_units(df, specs)
    pd.testing.assert_frame_equal(df, before)  # input still untouched
    # U0 now the aggregate (treated-units pooled), not the original single unit
    assert "U0" in set(out["unit"])
