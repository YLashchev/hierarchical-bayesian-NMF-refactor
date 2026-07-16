"""Parquet and CSV draws round-trips must produce identical downstream tables.

`_read_draws` returns different dtypes by format (CSV read-back gives
str/float64/int64; parquet preserves category/float32/int8). The golden gate is
CSV-only, so nothing else checks that reporting produces the same summary tables
regardless of `output.draws_format`. This closes that gap: write the same draws
frame both ways, reload each via `_read_draws`, run generate_reports on both,
and assert the human-facing table CSVs match (dtype-insensitive).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_run_analysis():
    spec = importlib.util.spec_from_file_location(
        "run_analysis_parity_test", ROOT / "scripts" / "run_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ra():
    return _load_run_analysis()


def _draws_frame() -> pd.DataFrame:
    rows = []
    times = pd.to_datetime(["2020-01-01", "2020-03-01", "2020-05-01", "2020-07-01"])
    rng = np.random.default_rng(0)
    for draw in (1, 2, 3, 4):
        for u in ("u0", "u1", "u2"):
            for t_i, t in enumerate(times):
                treated = int(u == "u0" and t_i >= 2)
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
                        "ypred": 50.0 + draw + rng.normal(scale=0.5),
                        "mu": np.log(50.0),
                        "mu_treated": np.log(50.0) + 0.1 * treated,
                    }
                )
    return pd.DataFrame(rows)


TABLES = [
    "figs/summary_table.csv",
    "figs/expected_vs_observed.csv",
    "figs/post_treatment_summary.csv",
    "figs/ppc/ppc_pvalues.csv",
]


def _report_from(ra, draws_df: pd.DataFrame, stem_name: str, fmt: str, tmp: Path):
    import matplotlib
    import matplotlib.pyplot as plt

    import bayesian_panel_nmf.reporting as reporting

    stem = tmp / stem_name
    ra._write_draws(draws_df, stem, fmt)
    reloaded = ra._read_draws(stem)
    out = tmp / f"report_{fmt}"
    try:
        with matplotlib.rc_context():
            reporting.generate_reports(
                reloaded, output_dir=out, target_unit="u0", print_tables=False
            )
    finally:
        plt.close("all")
    return out


def test_csv_and_parquet_yield_identical_tables(ra, tmp_path):
    draws = _draws_frame()
    csv_out = _report_from(ra, draws, "draws_csv", "csv", tmp_path)
    pq_out = _report_from(ra, draws, "draws_pq", "parquet", tmp_path)

    for rel in TABLES:
        csv_tbl = pd.read_csv(csv_out / rel)
        pq_tbl = pd.read_csv(pq_out / rel)
        # check_dtype=False: the point is that the DERIVED numbers agree even
        # though the reloaded draws frames carry different dtypes per format.
        pd.testing.assert_frame_equal(csv_tbl, pq_tbl, check_dtype=False)
