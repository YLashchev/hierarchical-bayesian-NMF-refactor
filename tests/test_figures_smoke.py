"""End-to-end figure smoke net for the plots.py teardown (Phase 7).

The golden gate verifies only ppc_pvalues.csv (plot DATA), not PNG pixels, and
the pure-plotting functions (fit/gap/interval/raw_rate/group_comparison) emit
only images. This test runs generate_reports() with real figures on a tiny
synthetic frame and asserts every expected artifact is written and non-empty —
so a refactor that crashes or silently drops a figure is caught even though no
pixel comparison exists.
"""

import numpy as np
import pandas as pd

import bayesian_panel_nmf.reporting as reporting


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


def test_generate_reports_writes_all_figures_nonempty(tmp_path):
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
            )
    finally:
        plt.close("all")

    figs = tmp_path / "figs"
    expected = [
        figs / "fit_u0.png",
        figs / "gap_u0.png",
        figs / "raw_rate.png",
        figs / "interval.png",
        figs / "group_comparison.png",
        figs / "ppc" / "ppc_pvalues.csv",
        figs / "summary_table.csv",
        figs / "expected_vs_observed.csv",
        figs / "post_treatment_summary.csv",
    ]
    for path in expected:
        assert path.exists(), f"missing artifact: {path}"
        assert path.stat().st_size > 0, f"empty artifact: {path}"

    # At least one PPC png rendered.
    ppc_pngs = list((figs / "ppc").glob("*.png"))
    assert ppc_pngs, "no PPC png written"
    assert all(p.stat().st_size > 0 for p in ppc_pngs)
