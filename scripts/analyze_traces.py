"""Compute ESS/R-hat directly from ArviZ NetCDF traces.

Replaces the old CSV-reshaping diagnostics with native ArviZ metrics,
allowing full hierarchical latent parameters (te, unit_weight, etc)
to be analyzed accurately.

Usage:
    uv run python scripts/analyze_traces.py results/total/NB_births_total_3_traces.nc
    uv run python scripts/analyze_traces.py results/total/NB_births_total_3_traces.nc --param-filter te,time_fac
"""

import argparse
import sys
from pathlib import Path

import arviz as az
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Compute diagnostics natively via ArviZ NetCDF.")
    parser.add_argument("nc_path", type=Path, help="Path to NetCDF traces file (e.g. ..._traces.nc)")
    parser.add_argument("--param-filter", help="Comma-separated list of parameter prefixes")
    args = parser.parse_args()

    if not args.nc_path.exists():
        logger.error(f"File not found: {args.nc_path}")
        sys.exit(1)

    logger.info(f"Loading NetCDF traces: {args.nc_path.name}")
    idata = az.from_netcdf(args.nc_path)

    var_names = None
    if args.param_filter:
        prefixes = args.param_filter.split(",")
        var_names = [v for v in idata.posterior.data_vars if any(v.startswith(p) for p in prefixes)]
        if not var_names:
            logger.error("No parameters matched --param-filter.")
            sys.exit(1)

    logger.info("Computing metrics via ArviZ...")
    
    summary_df = az.summary(idata, var_names=var_names, filter_vars="like")

    summary_df['base_param'] = summary_df.index.to_series().apply(lambda x: x.split('[')[0])
    
    all_ok = True
    
    for name, group in summary_df.groupby('base_param'):
        rhat = float(group['r_hat'].max())
        ess = float(group['ess_bulk'].min())

        status = "PASS" if rhat < 1.01 and ess > 400 else "WARN"
        if rhat >= 1.01 or ess < 100:
            status = "FAIL"
            all_ok = False

        print(f"\n{name}: {status}")
        print(f"  max R-hat: {rhat:.4f}")
        print(f"  min ESS:   {ess:.0f}")

    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
