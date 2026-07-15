"""Combined cut-draw formatting: provenance, global IDs, manifest."""

import numpy as np
import pandas as pd

from bayesian_panel_nmf.cut_inference import Stage1DrawRef
from bayesian_panel_nmf.cut_output import (
    build_cut_convergence_manifest,
    format_cut_component_draws,
    format_stage1_ppc_draws,
)

K, D, N = 2, 2, 2
CELLS = K * D * N


def _data_dict():
    groups = ["g0", "g1"]
    units = ["u0", "u1"]
    times = list(pd.to_datetime(["2020-01-01", "2020-03-01"]))
    rows = []
    for g in groups:
        for u in units:
            for t in times:
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "group": g,
                        "outcome": 10.0,
                        "denominator": 100.0,
                        "treatment": int(u == "u0" and t == times[1]),
                    }
                )
    return {
        "groups": groups,
        "units": units,
        "times": times,
        "df_preprocessed": pd.DataFrame(rows),
    }


def _ref(component=3):
    return Stage1DrawRef(
        component=component,
        stage1_draw=7,
        stage1_chain=2,
        stage1_iteration=3,
        mu_ctrl=np.zeros((K, D, N)),
        nb_concentration=np.ones(D) * 1e4,
    )


def _component_frame(draw_offset=0, component=3):
    n_out = 4
    te = np.arange(n_out * CELLS, dtype=float).reshape(n_out, K, D, N)
    ypred = np.ones((n_out, K, D, N))
    chain_ids = np.array([1, 1, 2, 2])
    return format_cut_component_draws(
        te, ypred, chain_ids, _ref(component), _data_dict(), draw_offset
    )


def test_component_frame_rows_and_provenance():
    df = _component_frame()
    assert len(df) == 4 * CELLS
    assert set(df["cut_component"]) == {3}
    assert set(df["stage1_draw"]) == {7}
    assert set(df["stage1_chain"]) == {2}
    assert set(df["stage1_iteration"]) == {3}
    assert sorted(df[".draw"].unique()) == [1, 2, 3, 4]


def test_real_stage2_chain_and_subsample_iteration():
    df = _component_frame()
    per_draw = df.drop_duplicates(".draw").sort_values(".draw")
    assert per_draw[".chain"].tolist() == [1, 1, 2, 2]
    assert per_draw[".iteration"].tolist() == [1, 2, 1, 2]


def test_mu_pairing_and_mu_treated():
    df = _component_frame()
    assert np.allclose(df["mu"], 0.0)  # fixed mu_ctrl repeats
    # format_draws drops its internal K/D/N merge-key columns before
    # returning, so we sort on group/unit/time instead -- their categorical
    # category order is set from data_dict's groups/units/times lists, which
    # is exactly the K/D/N meshgrid order used to build mu_treated below.
    first = df[df[".draw"] == 1].sort_values(["group", "unit", "time"])
    np.testing.assert_allclose(
        first["mu_treated"].to_numpy(), np.arange(CELLS, dtype=np.float32)
    )


def test_global_draw_uniqueness_across_components():
    a = _component_frame(draw_offset=0, component=1)
    b = _component_frame(draw_offset=4, component=2)
    combined = pd.concat([a, b], ignore_index=True)
    assert combined[".draw"].nunique() == 8
    assert set(combined.loc[combined["cut_component"] == 2, ".draw"]) == {5, 6, 7, 8}


def test_stage1_ppc_frame_has_no_te_effect():
    mu = np.zeros((2, 3, K, D, N))
    ypred = np.ones((2, 3, K, D, N))
    df = format_stage1_ppc_draws(mu, ypred, _data_dict())
    assert len(df) == 2 * 3 * CELLS
    np.testing.assert_array_equal(df["mu"].to_numpy(), df["mu_treated"].to_numpy())


def test_manifest_gate_logic():
    stage1 = {
        "rhat_max": 1.0,
        "ess_bulk_min": 500.0,
        "ess_tail_min": 500.0,
        "divergences": 0,
        "converged": True,
    }
    rec = {
        "component": 1,
        "stage1_draw": 1,
        "stage1_chain": 1,
        "stage1_iteration": 1,
        "rhat_max": 1.0,
        "ess_bulk_min": 500.0,
        "ess_tail_min": 500.0,
        "divergences": 0,
        "converged": True,
        "retained_draws": 32,
        "output_draws": 4,
    }
    bad = {**rec, "component": 2, "converged": False}
    manifest = build_cut_convergence_manifest(stage1, [rec, bad])
    assert manifest["inference_mode"] == "cut"
    assert manifest["converged"] is False
    assert manifest["stage2"]["all_converged"] is False
    assert manifest["stage2"]["failed_fits"] == 1
    assert len(manifest["stage2"]["fits"]) == 2
    good = build_cut_convergence_manifest(stage1, [rec])
    assert good["converged"] is True
