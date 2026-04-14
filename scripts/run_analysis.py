"""
Main script for running the Bayesian Panel NMF analysis pipeline.

This script:
1. Loads and preprocesses panel data using schema-based configuration
2. Runs Bayesian inference for specified model types and ranks
3. Extracts and saves MCMC diagnostics (ESS, R-hat, divergences)
4. Saves results in tidy format with standardized column names

Usage:
    python scripts/run_analysis.py --config configs/nativity_config.yaml
    python scripts/run_analysis.py --config configs/nativity_config.yaml --type groups --rank 10
    python scripts/run_analysis.py --config configs/nativity_config.yaml --verbose
    python scripts/run_analysis.py --config configs/nativity_config.yaml --log-file logs/analysis.log
    python scripts/run_analysis.py --config configs/nativity_config.yaml --save-diagnostics
"""

import argparse
import json
import yaml
from pathlib import Path

from loguru import logger

from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.output import format_draws
from bayesian_panel_nmf.inference import (
    run_mcmc_inference,
    generate_predictions,
    extract_diagnostics,
    check_convergence,
)
from bayesian_panel_nmf.models import model
from bayesian_panel_nmf.logging_config import setup_logging


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Run Bayesian Panel NMF analysis')
    parser.add_argument('--config', type=str, default='configs/nativity_config.yaml',
                        help='Path to config file')
    parser.add_argument('--type', type=str, default=None,
                        help='Model type to run; if not specified, runs all')
    parser.add_argument('--rank', type=int, default=None,
                        help='Model rank (overrides config)')
    parser.add_argument('--no-aggregate', action='store_true',
                        help='Skip temporal aggregation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose (DEBUG level) logging')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to log file (enables file logging)')
    parser.add_argument('--save-diagnostics', action='store_true',
                        help='Save MCMC diagnostics to JSON file')
    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, log_file=args.log_file)

    # Load configuration
    config = load_config(args.config)
    logger.info("=" * 60)
    logger.info("BAYESIAN PANEL NMF ANALYSIS")
    logger.info("=" * 60)
    logger.debug(f"Config file: {args.config}")
    logger.debug(f"Log level: {log_level}")

    # Handle --no-aggregate flag by modifying config
    if args.no_aggregate:
        config = config.copy()
        config['data'] = config['data'].copy()
        config['data']['aggregation'] = {'enabled': False}
        logger.info("Temporal aggregation disabled via --no-aggregate flag")

    # Setup output directory
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {output_dir}")

    output_config = config.get('output', {})

    # Loop over model types
    for model_type, type_config in config['model']['types'].items():
        if args.type and args.type != model_type:
            logger.debug(f"Skipping model type '{model_type}' (--type={args.type})")
            continue

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"RUNNING MODEL TYPE: {model_type.upper()}")
        logger.info("=" * 60)

        groups = type_config['groups']
        ranks = [args.rank] if args.rank else type_config.get('ranks_to_test', [10])
        exclude_units = type_config.get('exclude_units', None)
        logger.debug(f"Groups: {groups}, Ranks: {ranks}")
        if exclude_units:
            logger.debug(f"Excluding units: {exclude_units}")

        # Load and prepare data (single call)
        logger.info(f"[STEP 1] Loading and preparing data for groups: {groups}")
        data_dict = load_and_prepare(
            config['data']['input_file'],
            config,
            groups,
            exclude_units=exclude_units
        )

        logger.info(f"Processed data:")
        logger.info(f"  - Y shape: {data_dict['Y'].shape}")
        logger.info(f"  - Control observations: {data_dict['control_idx_array'].sum()}")
        logger.info(f"  - Treated observations: {(~data_dict['control_idx_array']).sum()}")
        logger.info(f"  - Missing observations: {data_dict['missing_idx_array'].sum()}")

        # Save preprocessed data
        df_preprocessed = data_dict['df_preprocessed']
        preproc_file = output_dir / f'df_{model_type}.csv'
        df_preprocessed.to_csv(preproc_file, index=False)
        logger.info(f"Saved preprocessed data to: {preproc_file}")

        for rank in ranks:
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"FITTING {model_type.upper()} MODEL WITH RANK {rank}")
            logger.info("=" * 60)

            # Run inference
            logger.info("[STEP 2] Running MCMC inference...")
            mcmc = run_mcmc_inference(data_dict, model, rank, config)

            # Extract diagnostics
            diagnostics = extract_diagnostics(mcmc)
            converged = check_convergence(diagnostics)
            
            # Save diagnostics if requested
            if args.save_diagnostics:
                dist = config['model'].get('outcome_distribution', 'NB')
                pattern = output_config.get('filename_pattern', '{distribution}_{type}_{rank}')
                diag_filename = pattern.format(
                    distribution=dist,
                    outcome='births',
                    type=model_type,
                    rank=rank
                )
                diagnostics_file = output_dir / f'{diag_filename}_diagnostics.json'
                
                # Make diagnostics JSON-serializable (exclude nested summary for brevity)
                diag_output = {
                    'n_eff_min': diagnostics['n_eff_min'],
                    'n_eff_mean': diagnostics['n_eff_mean'],
                    'rhat_max': diagnostics['rhat_max'],
                    'rhat_mean': diagnostics['rhat_mean'],
                    'divergences': diagnostics['divergences'],
                    'converged': diagnostics['converged'],
                    'num_chains': diagnostics['num_chains'],
                    'num_samples': diagnostics['num_samples'],
                    'num_warmup': diagnostics['num_warmup'],
                    'thinning': diagnostics['thinning'],
                    'model_type': model_type,
                    'rank': rank,
                }
                
                with open(diagnostics_file, 'w') as f:
                    json.dump(diag_output, f, indent=2)
                logger.info(f"Saved diagnostics to: {diagnostics_file}")
            
            if not converged:
                logger.warning(
                    "MCMC may not have converged. Consider increasing num_samples or num_warmup."
                )

            # Generate predictions
            logger.info("[STEP 3] Generating posterior predictions...")
            predictions = generate_predictions(mcmc, data_dict, model, rank, config)

            # Format and save output
            logger.info("[STEP 4] Processing and saving results...")
            samples = mcmc.get_samples(group_by_chain=True)
            draws_df = format_draws(samples, predictions, data_dict)

            # Save results
            dist = config['model'].get('outcome_distribution', 'NB')
            pattern = output_config.get('filename_pattern', '{distribution}_{type}_{rank}')
            filename = pattern.format(
                distribution=dist,
                outcome='births',
                type=model_type,
                rank=rank
            )

            results_file = output_dir / f'{filename}.csv'
            draws_df.to_csv(results_file, index=False)
            logger.info(f"Saved results to: {results_file}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review MCMC diagnostics (convergence, ESS, Rhat)")
    logger.info("  2. Load results in R/Python for visualization")
    logger.info("  3. Generate figures with downstream analysis scripts")


if __name__ == '__main__':
    main()
