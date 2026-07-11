import pandas as pd
import pytest

from bayesian_panel_nmf.visualization import _detect_outcome_column


def test_prefers_standard_outcome():
    df = pd.DataFrame({"outcome": [1], "births": [2]})
    assert _detect_outcome_column(df) == "outcome"


@pytest.mark.parametrize("legacy", ["births", "count", "y"])
def test_falls_back_to_legacy(legacy):
    df = pd.DataFrame({legacy: [1], "unit": ["a"]})
    assert _detect_outcome_column(df) == legacy


def test_raises_when_absent():
    with pytest.raises(ValueError, match="outcome column"):
        _detect_outcome_column(pd.DataFrame({"unit": ["a"]}))
