"""Check saved MCMC diagnostics JSON files.

Use after running with --save-diagnostics or output.save_diagnostics: true.

Usage:
    python scripts/check_diagnostics.py results/total/NB_births_total_5_diagnostics.json
    python scripts/check_diagnostics.py results/*/*_diagnostics.json
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger


def _check_file(path: Path) -> dict:
    """Print convergence status for one diagnostics JSON file."""
    with open(path) as f:
        diag = json.load(f)

    converged = diag.get("converged", False)
    rhat_max = diag.get("rhat_max", float("nan"))
    n_eff_min = diag.get("n_eff_min", float("nan"))
    divergences = diag.get("divergences", 0)

    status = "PASS" if converged else "FAIL"
    print(f"\n{path}")
    print(f"  Status: {status}")
    print(f"  Max R-hat: {rhat_max:.4f}")
    print(f"  Min ESS:   {n_eff_min:.0f}")
    print(f"  Divergences: {divergences}")

    issues = []
    if rhat_max > 1.01:
        issues.append(f"R-hat {rhat_max:.4f} > 1.01")
    if n_eff_min < 400:
        issues.append(f"ESS {n_eff_min:.0f} < 400")
    if divergences > 0:
        issues.append(f"{divergences} divergences")

    if issues:
        print(f"  Issues: {'; '.join(issues)}")

    return diag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check saved MCMC diagnostics JSON files"
    )
    parser.add_argument("paths", nargs="+", help="Path(s) to diagnostics JSON file(s)")
    args = parser.parse_args()

    all_pass = True
    for p in args.paths:
        path = Path(p)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            all_pass = False
            continue
        diag = _check_file(path)
        if not diag.get("converged", False):
            all_pass = False

    print()
    if all_pass:
        print("All diagnostics passed.")
        sys.exit(0)
    else:
        print("Some diagnostics failed. Review issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
