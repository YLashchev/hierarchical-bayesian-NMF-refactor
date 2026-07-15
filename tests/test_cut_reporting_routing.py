"""ppc_draws_df must route ONLY the JAMA PPC suite to the Stage-1 posterior."""

import numpy as np
import pandas as pd
import pytest

import bayesian_panel_nmf.reporting as reporting


def _draws_frame(units=("u0", "u1"), marker=0.0):
    rows = []
    times = pd.to_datetime(["2020-01-01", "2020-03-01", "2020-05-01", "2020-07-01"])
    for draw in (1, 2, 3, 4):
        for u in units:
            for t_i, t in enumerate(times):
                treated = int(u == units[0] and t_i >= 2)
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
                        "ypred": 50.0 + marker,
                        "mu": np.log(50.0),
                        "mu_treated": np.log(50.0) + 0.1 * treated,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("use_ppc_frame", [True, False])
def test_ppc_source_routing(monkeypatch, tmp_path, use_ppc_frame):
    captured = {}

    def fake_ppc(df, **kwargs):
        captured["units"] = set(df["unit"])
        return {}

    monkeypatch.setattr(reporting, "make_all_ppc_plots", fake_ppc)

    draws_df = _draws_frame(units=("u0", "u1"))
    ppc_df = _draws_frame(units=("p0", "p1")) if use_ppc_frame else None
    reporting.generate_reports(
        draws_df,
        output_dir=tmp_path,
        target_unit="u0",
        print_tables=False,
        ppc_draws_df=ppc_df,
    )
    if use_ppc_frame:
        assert captured["units"] == {"p0", "p1"}
    else:
        assert captured["units"] == {"u0", "u1"}
