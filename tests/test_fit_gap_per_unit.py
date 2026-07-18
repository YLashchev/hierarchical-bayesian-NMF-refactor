"""fit_gap_per_unit: render fit/gap for the target + every treated unit,
restricted to the 'total' group (matches upstream, which loops treated states
at category='Total' only). No-op when no 'total' group is present.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_panel_nmf.reports import generate_reports


def _draws(groups: list[str], units: list[str], treated: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    times = pd.date_range("2016-01-01", periods=6, freq="2MS")
    rows = []
    for g in groups:
        for u in units:
            for d in range(4):
                for i, t in enumerate(times):
                    treat = 1 if (u in treated and i >= 3) else 0
                    mu = np.log(100.0)
                    rows.append(
                        {
                            "unit": u,
                            "group": g,
                            "time": t,
                            "treatment": treat,
                            ".draw": d,
                            "outcome": 100 + rng.normal(),
                            "denominator": 1000.0,
                            "ypred": 100 + rng.normal(),
                            "mu": mu,
                            "mu_treated": mu,
                        }
                    )
    return pd.DataFrame(rows)


def test_per_unit_renders_fit_gap_for_each_treated_unit(tmp_path: Path):
    df = _draws(groups=["total"], units=["A", "B", "C"], treated=["A", "B"])
    generate_reports(
        df,
        output_dir=tmp_path,
        target_unit="A",
        figures=["unit_fit", "unit_gap"],
        fit_gap_per_unit=True,
    )
    figs = tmp_path / "figs"
    # per-unit fit/gap for each TREATED unit (A, B), not the control C
    assert (figs / "fit_A.png").exists()
    assert (figs / "fit_B.png").exists()
    assert (figs / "gap_A.png").exists()
    assert (figs / "gap_B.png").exists()
    assert not (figs / "fit_C.png").exists()


def test_default_off_renders_only_target(tmp_path: Path):
    df = _draws(groups=["total"], units=["A", "B"], treated=["A", "B"])
    generate_reports(
        df,
        output_dir=tmp_path,
        target_unit="A",
        figures=["unit_fit", "unit_gap"],
        # fit_gap_per_unit defaults False
    )
    figs = tmp_path / "figs"
    assert (figs / "fit_A.png").exists()
    assert not (figs / "fit_B.png").exists()  # B not rendered when flag off


def test_no_op_when_no_total_group(tmp_path: Path):
    # education-type run: no 'total' group -> per-unit expansion must not fire
    df = _draws(groups=["nohs", "coll"], units=["A", "B"], treated=["A", "B"])
    generate_reports(
        df,
        output_dir=tmp_path,
        target_unit="A",
        figures=["unit_fit", "unit_gap"],
        fit_gap_per_unit=True,
    )
    # only the target's per-group fit exists; no per-unit B expansion
    figs = tmp_path / "figs"
    assert not (figs / "nohs" / "fit_B.png").exists()
    assert not (figs / "coll" / "fit_B.png").exists()
