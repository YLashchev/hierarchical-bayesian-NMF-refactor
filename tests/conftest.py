# No sys.path hacks needed — all test imports use the installed package.
from collections.abc import Callable
from pathlib import Path

import pytest

from bayesian_panel_nmf.config import Config

# Shared data section for inference tests that only vary model/mcmc. The panel
# CSV is never read (these tests stub sampling), so input_file is a placeholder.
_INFERENCE_DATA = {
    "input_file": "unused.csv",
    "output_dir": "unused",
    "schema": {
        "unit_col": "state",
        "time_col": "time",
        "treatment_col": "exposed",
        "outcomes": [{"outcome_col": "births_total", "label": "total"}],
    },
}

_DEFAULT_OUTCOME = [
    {"outcome_col": "outcome", "denominator_col": "pop", "label": "total"}
]


@pytest.fixture
def make_inference_config() -> Callable[..., Config]:
    """Config that varies only model/mcmc over a fixed data section."""

    def build(model: dict | None = None, mcmc: dict | None = None) -> Config:
        return Config.model_validate(
            {"data": _INFERENCE_DATA, "model": model or {}, "mcmc": mcmc or {}}
        )

    return build


@pytest.fixture
def make_data_config() -> Callable[..., Config]:
    """Config for data-loading tests; vary outcomes and panel flags per call."""

    def build(
        csv_path: Path,
        *,
        outcomes: list | None = None,
        allow_unbalanced: bool = False,
    ) -> Config:
        data = {
            "input_file": str(csv_path),
            "output_dir": "results/test",
            "schema": {
                "unit_col": "state",
                "time_col": "time",
                "treatment_col": "treated",
                "outcomes": outcomes if outcomes is not None else _DEFAULT_OUTCOME,
            },
            "aggregation": {"enabled": False},
        }
        if allow_unbalanced:
            data["allow_unbalanced_panel"] = True
        return Config.model_validate({"data": data})

    return build
