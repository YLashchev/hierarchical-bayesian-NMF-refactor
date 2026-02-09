"""
Main script for running the Bayesian Panel NMF analysis pipeline.

This script:
1. Loads and preprocesses panel data using schema-based configuration
2. Runs Bayesian inference for specified model types and ranks
3. Saves results in tidy format with standardized column names

Usage:
    python scripts/run_analysis.py --config configs/nativity_config.yaml
    python scripts/run_analysis.py --config configs/nativity_config.yaml --type groups --rank 10
"""

import argparse
import yaml
from pathlib import Path

from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.output import format_draws
from bayesian_panel_nmf.inference import run_mcmc_inference, generate_predictions
from bayesian_panel_nmf.models import model


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
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print("=" * 60)
    print("BAYESIAN PANEL NMF ANALYSIS")
    print("=" * 60)

    # Handle --no-aggregate flag by modifying config
    if args.no_aggregate:
        config = config.copy()
        config['data'] = config['data'].copy()
        config['data']['aggregation'] = {'enabled': False}

    # Setup output directory
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_config = config.get('output', {})

    # Loop over model types
    for model_type, type_config in config['model']['types'].items():
        if args.type and args.type != model_type:
            continue

        print("\n" + "=" * 60)
        print(f"RUNNING MODEL TYPE: {model_type.upper()}")
        print("=" * 60)

        groups = type_config['groups']
        ranks = [args.rank] if args.rank else type_config.get('ranks_to_test', [10])

        # Load and prepare data (single call)
        print(f"\n[STEP 1] Loading and preparing data for groups: {groups}")
        data_dict = load_and_prepare(
            config['data']['input_file'],
            config,
            groups
        )

        print(f"Processed data:")
        print(f"  - Y shape: {data_dict['Y'].shape}")
        print(f"  - Control observations: {data_dict['control_idx_array'].sum()}")
        print(f"  - Treated observations: {(~data_dict['control_idx_array']).sum()}")
        print(f"  - Missing observations: {data_dict['missing_idx_array'].sum()}")

        # Save preprocessed data
        df_preprocessed = data_dict['df_preprocessed']
        preproc_file = output_dir / f'df_{model_type}.csv'
        df_preprocessed.to_csv(preproc_file, index=False)
        print(f"Saved preprocessed data to: {preproc_file}")

        for rank in ranks:
            print(f"\n{'=' * 60}")
            print(f"FITTING {model_type.upper()} MODEL WITH RANK {rank}")
            print(f"{'=' * 60}")

            # Run inference
            print("\n[STEP 2] Running MCMC inference...")
            mcmc = run_mcmc_inference(data_dict, model, rank, config)

            # Generate predictions
            print("\n[STEP 3] Generating posterior predictions...")
            predictions = generate_predictions(mcmc, data_dict, model, rank, config)

            # Format and save output
            print("\n[STEP 4] Processing and saving results...")
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
            print(f"Saved results to: {results_file}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review MCMC diagnostics (convergence, ESS, Rhat)")
    print("  2. Load results in R/Python for visualization")
    print("  3. Generate figures with downstream analysis scripts")


if __name__ == '__main__':
    main()
