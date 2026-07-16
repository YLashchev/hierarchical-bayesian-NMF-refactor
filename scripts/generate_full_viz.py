"""Regenerate figures + tables from an existing draws artifact (csv or parquet).

Standalone counterpart to the automatic reporting done by
``scripts/run_analysis.py`` when ``output.figures: true`` is set in
the config. Useful when you want to re-render plots after tweaking
visualization code without re-running MCMC.

Usage:
    uv run scripts/generate_full_viz.py \
        --results results/total/NB_births_total_5.csv \
        [--target Texas] [--group total]
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

from bayesian_panel_nmf.reports import generate_reports

_ROOT = Path(__file__).resolve().parent


def _read_draws_by_stem(results_path: Path) -> pd.DataFrame:
    """Load a draws artifact given a stem or either extension (.csv/.parquet).

    Delegates to ``run_analysis._read_draws`` (tries .parquet then .csv off
    the stem) so this script stays in sync with what ``run_analysis.py``
    actually writes.
    """
    spec = importlib.util.spec_from_file_location(
        "run_analysis_for_viz", _ROOT / "run_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stem = results_path.with_suffix("")
    return module._read_draws(stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        type=Path,
        default=Path("results/total/NB_births_total_5.csv"),
        help="Path to the draws CSV produced by run_analysis.py",
    )
    ap.add_argument(
        "--ppc-results",
        type=Path,
        default=None,
        help=(
            "Optional Stage-1 PPC draws CSV (cut mode: <stem>_cut_stage1_ppc.csv); "
            "routes the JAMA PPC suite to the full Stage-1 posterior"
        ),
    )
    ap.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target unit for fit/gap/summary plots (auto-detected if omitted)",
    )
    ap.add_argument(
        "--group",
        type=str,
        default=None,
        action="append",
        help="Group label for per-unit plots. Repeat for multiple groups; "
        "omit to render every group present in the draws CSV.",
    )
    args = ap.parse_args()

    draws_df = _read_draws_by_stem(args.results)
    output_dir = args.results.parent
    ppc_draws_df = pd.read_csv(args.ppc_results) if args.ppc_results else None
    generate_reports(
        draws_df,
        output_dir=output_dir,
        target_unit=args.target,
        groups=args.group,
        ppc_draws_df=ppc_draws_df,
    )


if __name__ == "__main__":
    main()
