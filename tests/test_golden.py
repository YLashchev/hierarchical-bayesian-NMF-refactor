"""Golden-fixture regression gate for the refactor.

Re-runs the smoke-test config and asserts bit-identical draws + tables vs the
frozen Phase-1 baseline (JAX 0.10 / NumPyro 0.21 / ArviZ 1.2). Every pure-move
refactor phase (2,4,5,6,7,8,9) MUST keep this green. A single differing value
means an accidental behavior change.

Skips automatically if the fixtures are absent (they are branch-local and
gitignored; regenerate via the smoke config — see tests/fixtures/README.md).
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "fixtures" / "golden" / "total"
DRAWS = "NB_births_total_3.csv"
TABLES = [
    "figs/summary_table.csv",
    "figs/expected_vs_observed.csv",
    "figs/post_treatment_summary.csv",
    "figs/ppc/ppc_pvalues.csv",
]

pytestmark = pytest.mark.skipif(
    not (GOLDEN / DRAWS).exists(),
    reason="golden fixtures absent (branch-local; see tests/fixtures/README.md)",
)


@pytest.fixture(scope="module")
def fresh_run(tmp_path_factory) -> Path:
    """Re-run the smoke config to a temp dir on the CURRENT code."""
    out = tmp_path_factory.mktemp("golden_check")
    cfg = out / "cfg.yaml"
    src = (ROOT / "configs" / "fertility_smoke_test.yaml").read_text()
    cfg.write_text(src.replace('output_dir: "results"', f'output_dir: "{out}"'))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bayesian_panel_nmf.cli",
            "run",
            "--config",
            str(cfg),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return out / "total"


def test_golden_smoke_draws_bit_identical(fresh_run):
    new = pd.read_csv(fresh_run / DRAWS)
    old = pd.read_csv(GOLDEN / DRAWS)
    pd.testing.assert_frame_equal(new, old, check_exact=True)


@pytest.mark.parametrize("rel", TABLES)
def test_golden_smoke_tables_bit_identical(fresh_run, rel):
    new = pd.read_csv(fresh_run / rel)
    old = pd.read_csv(GOLDEN / rel)
    pd.testing.assert_frame_equal(new, old, check_exact=True)
