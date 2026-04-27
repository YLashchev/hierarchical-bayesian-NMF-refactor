"""Regenerate figures + tables from an existing draws CSV.

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
from pathlib import Path

import pandas as pd

from bayesian_panel_nmf.reporting import generate_reports


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        type=Path,
        default=Path("results/total/NB_births_total_5.csv"),
        help="Path to the draws CSV produced by run_analysis.py",
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

    draws_df = pd.read_csv(args.results)
    output_dir = args.results.parent
    generate_reports(
        draws_df,
        output_dir=output_dir,
        target_unit=args.target,
        groups=args.group,
    )


if __name__ == "__main__":
    main()
