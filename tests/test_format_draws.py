"""Characterization tests for format_draws before extracting its
posterior-frame-construction and observed-data-merge steps into named
helpers (Tier 1b of the repo clarity refactor). format_draws had zero
direct tests before this file — see AGENTS.md / safe-refactor-scientific
skill: critical code lacking tests must be characterized before refactor."""

import numpy as np
import pandas as pd
import pytest

from bayesian_panel_nmf.output import format_draws
from bayesian_panel_nmf.validation import DataError


def _tiny_inputs(C=2, S=3, K=2, D=2, N=4, with_te=True):
    """Build minimal samples/predictions/data_dict matching format_draws'
    documented shape contract: (chains, samples, groups, units, times)."""
    rng = np.random.default_rng(0)
    shape = (C, S, K, D, N)
    mu_ctrl = rng.normal(size=shape).astype(np.float32)
    predictions = rng.poisson(5.0, size=shape).astype(np.float64)

    samples = {"mu_ctrl": mu_ctrl}
    if with_te:
        samples["te"] = rng.normal(scale=0.1, size=shape).astype(np.float32)

    groups = [f"g{i}" for i in range(K)]
    units = [f"u{i}" for i in range(D)]
    times = pd.date_range("2020-01-01", periods=N, freq="MS").tolist()

    rows = []
    for k, g in enumerate(groups):
        for d, u in enumerate(units):
            for n, t in enumerate(times):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "group": g,
                        "outcome": float(10 + k + d + n),
                        "denominator": 1000.0,
                        "treatment": int(d == 0 and n >= N // 2),
                    }
                )
    df_preprocessed = pd.DataFrame(rows)

    data_dict = {
        "groups": groups,
        "units": units,
        "times": times,
        "df_preprocessed": df_preprocessed,
    }
    return samples, predictions, data_dict


def test_format_draws_returns_expected_columns_in_fixed_order():
    samples, predictions, data_dict = _tiny_inputs()
    result = format_draws(samples, predictions, data_dict)

    assert list(result.columns) == [
        ".draw",
        ".chain",
        ".iteration",
        "unit",
        "time",
        "group",
        "outcome",
        "denominator",
        "treatment",
        "ypred",
        "mu",
        "mu_treated",
    ]


def test_format_draws_row_count_matches_full_grid():
    C, S, K, D, N = 2, 3, 2, 2, 4
    samples, predictions, data_dict = _tiny_inputs(C, S, K, D, N)
    result = format_draws(samples, predictions, data_dict)

    assert len(result) == C * S * K * D * N


def test_format_draws_draw_chain_iteration_are_one_indexed():
    samples, predictions, data_dict = _tiny_inputs(C=2, S=3)
    result = format_draws(samples, predictions, data_dict)

    assert result[".chain"].min() == 1
    assert result[".chain"].max() == 2
    assert result[".iteration"].min() == 1
    assert result[".iteration"].max() == 3
    # .draw = (chain - 1) * S + iteration, so with C=2,S=3 draw ranges 1..6
    assert result[".draw"].min() == 1
    assert result[".draw"].max() == 6


def test_format_draws_mu_treated_equals_mu_plus_te_when_te_present():
    samples, predictions, data_dict = _tiny_inputs(with_te=True)
    result = format_draws(samples, predictions, data_dict)

    # mu_treated - mu should equal the (float32-cast) te values; check via
    # round-trip through the same dtype the implementation uses.
    te_flat = samples["te"].ravel().astype(np.float32)
    mu_flat = samples["mu_ctrl"].ravel().astype(np.float32)
    expected_mu_treated = mu_flat + te_flat

    # Row order from format_draws is (chain, sample, group, unit, time)
    # per the meshgrid indexing="ij" order — sort both sides by the same
    # keys before comparing to avoid depending on exact row order here.
    np.testing.assert_allclose(
        np.sort(result["mu_treated"].to_numpy()),
        np.sort(expected_mu_treated),
        rtol=1e-6,
    )


def test_format_draws_mu_treated_equals_mu_when_te_absent():
    samples, predictions, data_dict = _tiny_inputs(with_te=False)
    result = format_draws(samples, predictions, data_dict)

    np.testing.assert_allclose(
        result["mu_treated"].to_numpy(), result["mu"].to_numpy(), rtol=1e-6
    )


def test_format_draws_merges_observed_outcome_correctly():
    samples, predictions, data_dict = _tiny_inputs(C=1, S=1, K=1, D=1, N=2)
    result = format_draws(samples, predictions, data_dict)

    # With K=D=1, every row's (group, unit, time) maps 1:1 to a row in
    # df_preprocessed; outcome must match exactly regardless of draw.
    df_preprocessed = data_dict["df_preprocessed"]
    for _, row in result.iterrows():
        expected_outcome = df_preprocessed.loc[
            (df_preprocessed["unit"] == row["unit"])
            & (df_preprocessed["time"] == row["time"])
            & (df_preprocessed["group"] == row["group"]),
            "outcome",
        ].iloc[0]
        assert row["outcome"] == pytest.approx(expected_outcome)


def test_format_draws_raises_dataerror_on_missing_data_dict_key():
    samples, predictions, data_dict = _tiny_inputs()
    del data_dict["units"]

    with pytest.raises(DataError, match="missing keys"):
        format_draws(samples, predictions, data_dict)


def test_format_draws_raises_dataerror_on_missing_mu_ctrl():
    samples, predictions, data_dict = _tiny_inputs()
    del samples["mu_ctrl"]

    with pytest.raises(DataError, match="mu_ctrl"):
        format_draws(samples, predictions, data_dict)


def test_format_draws_raises_dataerror_on_shape_mismatch():
    samples, predictions, data_dict = _tiny_inputs()
    bad_predictions = predictions[:, :, :, :, :-1]  # drop last time period

    with pytest.raises(DataError):
        format_draws(samples, bad_predictions, data_dict)


def test_format_draws_dtype_optimization_applied():
    samples, predictions, data_dict = _tiny_inputs()
    result = format_draws(samples, predictions, data_dict)

    assert result["unit"].dtype.name == "category"
    assert result["group"].dtype.name == "category"
    assert result[".chain"].dtype == np.int8
    assert result["ypred"].dtype == np.float32
    assert result["mu"].dtype == np.float32
