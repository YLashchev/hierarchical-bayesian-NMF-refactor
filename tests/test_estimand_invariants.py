"""Person-years fix must not move the denominator-free causal estimands.

The absolute-rate bug fix (carry start_date/end_date into the draws frame so
reporting computes real person-years instead of years=1.0) changes person-years
and every rate column. It must NOT change the two published, denominator-free
estimands: the count difference sum(exp(mu_treated)) - sum(exp(mu)) and the
percent change 100*(ratio - 1). This pins that invariance.
"""

import numpy as np
import pandas as pd

from bayesian_panel_nmf.tables import make_summary_table


def _draws_frame(*, with_dates: bool) -> pd.DataFrame:
    """Two draws, one unit, two bimonthly post-treatment periods.

    mu / mu_treated are fixed so treated and untreated counts are exact; only
    the presence of start/end dates (hence person-years) differs between the
    two builds.
    """
    rows = []
    # per-period expected counts: untreated exp(mu)=40, treated exp(mu_treated)=50
    mu = np.log(40.0)
    mu_treated = np.log(50.0)
    periods = [("2023-01-01", "2023-02-28"), ("2023-03-01", "2023-04-30")]
    for draw in (0, 1):
        for start, end in periods:
            row = {
                ".draw": draw,
                "unit": "A",
                "group": "total",
                "treatment": 1,
                "outcome": 45.0,
                "denominator": 6000.0,
                "ypred": 45.0,
                "mu": mu,
                "mu_treated": mu_treated,
            }
            if with_dates:
                row["start_date"] = start
                row["end_date"] = end
            rows.append(row)
    return pd.DataFrame(rows)


def _parse_ci(cell: str) -> tuple[float, float, float]:
    """Parse a 'mean (lo, hi)' or 'mean% (lo%, hi%)' summary cell to floats."""
    mean_part, rest = cell.split("(")
    lo, hi = rest.rstrip(")").split(",")
    strip = lambda s: float(s.replace("%", "").strip())  # noqa: E731
    return strip(mean_part), strip(lo), strip(hi)


def test_count_diff_and_pct_change_invariant_to_person_years():
    """Count diff and percent change are identical whether years=1/6 or 1.0."""
    with_dates = make_summary_table(_draws_frame(with_dates=True), "A")
    no_dates = make_summary_table(_draws_frame(with_dates=False), "A")

    assert with_dates["Diff (95% CI)"].iloc[0] == no_dates["Diff (95% CI)"].iloc[0]
    assert with_dates["Pct Change CI"].iloc[0] == no_dates["Pct Change CI"].iloc[0]

    # Sanity: count diff is the model contrast 2*(exp(log 50)-exp(log 40)) ~= 20.
    # The table renders int(mean); exp(log 50)=49.9999.. truncates to 19.
    diff_mean, _, _ = _parse_ci(with_dates["Diff (95% CI)"].iloc[0])
    assert diff_mean == 19.0
    # Percent change: 100*(50/40 - 1) = 25%, rendered to one decimal.
    pct_mean, _, _ = _parse_ci(with_dates["Pct Change CI"].iloc[0])
    assert pct_mean == 25.0


def test_person_years_and_rates_scale_with_years():
    """Person-years and rates must reflect real interval length, not years=1."""
    with_dates = make_summary_table(_draws_frame(with_dates=True), "A")
    no_dates = make_summary_table(_draws_frame(with_dates=False), "A")

    # Two ~2-month periods -> years per period < 1; person-years must be far
    # below the years=1.0 fallback (which is just summed population).
    py_dated = with_dates["Person-Years"].iloc[0]
    py_flat = no_dates["Person-Years"].iloc[0]
    assert py_dated < py_flat
    # denom = sum(pop * years); pop=6000, two periods ~59/59 days each.
    expected_years = (58 + 60) / 365.25  # Jan1-Feb28 + Mar1-Apr30 day spans
    assert abs(py_dated - 6000.0 * expected_years) < 6000.0  # within one period

    # Rate scales inversely with person-years, so dated rate > flat rate.
    assert with_dates["Obs Rate"].iloc[0] > no_dates["Obs Rate"].iloc[0]
