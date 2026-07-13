import numpy as np
import pandas as pd
import pytest

from bayesian_panel_nmf.aggregate_units import add_aggregate_units
from bayesian_panel_nmf.validation import ConfigError


def _draws_df() -> pd.DataFrame:
    rows = []
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    values = {
        "A": {
            "outcome": [10, 12],
            "ypred": [9, 11],
            "denominator": [100, 110],
            "mu": [2, 3],
        },
        "B": {
            "outcome": [20, 22],
            "ypred": [19, 21],
            "denominator": [200, 210],
            "mu": [5, 7],
        },
        "C": {
            "outcome": [30, 32],
            "ypred": [29, 31],
            "denominator": [300, 310],
            "mu": [11, 13],
        },
    }
    for draw in [1, 2]:
        for unit, vals in values.items():
            for t_idx, time in enumerate(times):
                treated = int(unit in {"A", "B"} and t_idx == 1)
                rows.append(
                    {
                        ".draw": draw,
                        ".chain": 1,
                        ".iteration": draw,
                        "unit": unit,
                        "time": time,
                        "group": "g",
                        "outcome": vals["outcome"][t_idx] + draw,
                        "ypred": vals["ypred"][t_idx] + draw,
                        "denominator": vals["denominator"][t_idx],
                        "treatment": treated,
                        "mu": np.log(vals["mu"][t_idx]),
                        "mu_treated": np.log(vals["mu"][t_idx] + treated),
                    }
                )
    return pd.DataFrame(rows)


def _row(df: pd.DataFrame, unit: str, time: str, draw: int = 1) -> pd.Series:
    mask = (
        (df["unit"] == unit)
        & (df["time"] == pd.Timestamp(time))
        & (df[".draw"] == draw)
    )
    rows = df.loc[mask]
    assert len(rows) == 1
    return rows.iloc[0]


def test_treated_unit_selection_and_aggregation() -> None:
    df = _draws_df()

    result = add_aggregate_units(
        df, [{"unit": "Treated units", "include_treated_units": True}]
    )

    agg = _row(result, "Treated units", "2020-01-01")
    a = _row(df, "A", "2020-01-01")
    b = _row(df, "B", "2020-01-01")
    assert agg["outcome"] == a["outcome"] + b["outcome"]
    assert agg["ypred"] == a["ypred"] + b["ypred"]
    assert agg["denominator"] == a["denominator"] + b["denominator"]
    assert agg["treatment"] == 0
    assert np.exp(agg["mu"]) == pytest.approx(np.exp(a["mu"]) + np.exp(b["mu"]))
    assert np.exp(agg["mu_treated"]) == pytest.approx(
        np.exp(a["mu_treated"]) + np.exp(b["mu_treated"])
    )

    post = _row(result, "Treated units", "2020-02-01")
    assert post["treatment"] == 1


def test_exclude_units_from_treated_selection() -> None:
    df = _draws_df()

    result = add_aggregate_units(
        df,
        [
            {
                "unit": "Treated excluding B",
                "include_treated_units": True,
                "exclude_units": ["B"],
            }
        ],
    )

    agg = _row(result, "Treated excluding B", "2020-02-01")
    a = _row(df, "A", "2020-02-01")
    assert agg["outcome"] == a["outcome"]
    assert agg["ypred"] == a["ypred"]
    assert agg["denominator"] == a["denominator"]
    assert agg["treatment"] == a["treatment"]


def test_explicit_include_units() -> None:
    df = _draws_df()

    result = add_aggregate_units(
        df, [{"unit": "A plus C", "include_units": ["A", "C"]}]
    )

    agg = _row(result, "A plus C", "2020-01-01")
    a = _row(df, "A", "2020-01-01")
    c = _row(df, "C", "2020-01-01")
    assert agg["outcome"] == a["outcome"] + c["outcome"]
    assert agg["ypred"] == a["ypred"] + c["ypred"]
    assert agg["denominator"] == a["denominator"] + c["denominator"]
    assert np.exp(agg["mu"]) == pytest.approx(np.exp(a["mu"]) + np.exp(c["mu"]))


def test_add_aggregate_units_does_not_mutate_original() -> None:
    df = _draws_df()
    before = df.copy(deep=True)

    result = add_aggregate_units(
        df, [{"unit": "A plus B", "include_units": ["A", "B"]}]
    )

    pd.testing.assert_frame_equal(df, before)
    assert "A plus B" not in set(df["unit"])
    assert "A plus B" in set(result["unit"])


def test_missing_include_units_warn_and_skip() -> None:
    df = _draws_df()

    with pytest.warns(UserWarning, match="skips missing units"):
        result = add_aggregate_units(
            df, [{"unit": "A plus missing", "include_units": ["A", "Missing"]}]
        )

    agg = _row(result, "A plus missing", "2020-01-01")
    a = _row(df, "A", "2020-01-01")
    assert agg["outcome"] == a["outcome"]


def test_missing_include_units_strict_raises() -> None:
    df = _draws_df()

    with pytest.raises(ConfigError, match="references missing units"):
        add_aggregate_units(
            df,
            [
                {
                    "unit": "A plus missing",
                    "include_units": ["A", "Missing"],
                    "strict": True,
                }
            ],
        )


def test_requires_exactly_one_include_selector() -> None:
    df = _draws_df()

    with pytest.raises(ConfigError, match="exactly one"):
        add_aggregate_units(
            df,
            [
                {
                    "unit": "bad",
                    "include_treated_units": True,
                    "include_units": ["A"],
                }
            ],
        )


def test_include_all_units_sums_every_unit() -> None:
    df = _draws_df()

    result = add_aggregate_units(df, [{"unit": "All units", "include_all_units": True}])

    agg = _row(result, "All units", "2020-01-01")
    a = _row(df, "A", "2020-01-01")
    b = _row(df, "B", "2020-01-01")
    c = _row(df, "C", "2020-01-01")
    assert agg["outcome"] == a["outcome"] + b["outcome"] + c["outcome"]
    assert agg["denominator"] == a["denominator"] + b["denominator"] + c["denominator"]
    assert np.exp(agg["mu"]) == pytest.approx(
        np.exp(a["mu"]) + np.exp(b["mu"]) + np.exp(c["mu"])
    )


def test_collision_requires_overwrite() -> None:
    df = _draws_df()

    with pytest.raises(ConfigError, match="collides"):
        add_aggregate_units(df, [{"unit": "A", "include_units": ["B"]}])

    result = add_aggregate_units(
        df, [{"unit": "A", "include_units": ["B"], "overwrite": True}]
    )
    assert len(result[result["unit"] == "A"]) == len(df[df["unit"] == "B"])
