"""
Main script for running the complete nativity analysis pipeline.

This script:
1. Loads and preprocesses the nativity data
2. Runs Bayesian inference for multiple model ranks
3. Saves results
4. Generates key figures

Usage:
    python scripts/run_full_analysis.py --config configs/nativity_config.yaml
"""

import argparse
import yaml
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nativity_analysis.data import (
    load_nativity_data,
    wide_to_long,
    aggregate_to_bimonthly,
    create_exposure_codes,
    prepare_model_data,
    filter_time_period
)
from nativity_analysis.inference import (
    run_mcmc_inference,
    generate_predictions,
    merge_draws_and_data
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Run nativity analysis pipeline')
    parser.add_argument('--config', type=str, default='configs/nativity_config.yaml',
                        help='Path to config file')
    parser.add_argument('--type', type=str, default=None,
                        help='Model type to run (total or nativity); if not specified, runs all')
    parser.add_argument('--rank', type=int, default=None,
                        help='Model rank (overrides config)')
    parser.add_argument('--no-bimonthly', action='store_true',
                        help='Skip bimonthly aggregation (use monthly)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print("="*60)
    print("NATIVITY ANALYSIS PIPELINE")
    print("="*60)
    
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which types to run
    if args.type is not None:
        types_to_run = [args.type]
    else:
        types_to_run = list(config['model']['types'].keys())
    
    print(f"\nModel types to run: {types_to_run}")
    
    # Loop over types (like dobbs_fertility loops over total, race, age, etc.)
    for model_type in types_to_run:
        print("\n" + "="*60)
        print(f"RUNNING MODEL TYPE: {model_type.upper()}")
        print("="*60)
        
        type_config = config['model']['types'][model_type]
        sub_groups = type_config['sub_groups']
        
        # =====================================================================
        # STEP 1: Load and preprocess data
        # =====================================================================
        print("\n[STEP 1] Loading and preprocessing data...")
        
        # Load raw data
        df_raw = load_nativity_data(config['data']['input_file'])
        
        # Convert to long format
        print(f"Converting to long format with groups: {sub_groups}")
        df_long = wide_to_long(df_raw, sub_groups=sub_groups)
        
        # Filter time period
        df_long = filter_time_period(
            df_long,
            start_date=config['data'].get('start_date'),
            end_date=config['data'].get('end_date')
        )
        
        # Aggregate to bimonthly if requested
        if config['data'].get('aggregate_to_bimonthly', True) and not args.no_bimonthly:
            print("Aggregating to bimonthly periods...")
            df_long = aggregate_to_bimonthly(df_long)
        
        # Create exposure codes using raw 'exposed' column (has correct state-specific timing)
        df_long = create_exposure_codes(df_long, use_raw_exposed=True)
        
        print(f"Processed data shape: {df_long.shape}")
        print(f"Time range: {df_long['time'].min()} to {df_long['time'].max()}")
        print(f"States: {df_long['state'].nunique()}")
        print(f"Groups: {df_long['group'].unique()}")
        
        # Save preprocessed data for this type
        group_name = '_'.join(sub_groups)
        df_long.to_csv(output_dir / f'df_{model_type}_{group_name}.csv', index=False)
        print(f"\nSaved preprocessed data to {output_dir / f'df_{model_type}_{group_name}.csv'}")
        
        # =====================================================================
        # STEP 2: Prepare data for modeling
        # =====================================================================
        print("\n[STEP 2] Preparing data for Bayesian model...")
        
        data_dict = prepare_model_data(
            df_long,
            sub_groups=sub_groups,
            outcome_type=config['model']['outcome_type']
        )
        
        print(f"Model data prepared:")
        print(f"  - Y shape: {data_dict['Y'].shape}")
        print(f"  - Denominators shape: {data_dict['denominators'].shape}")
        print(f"  - Control observations: {data_dict['control_idx_array'].sum()}")
        print(f"  - Treated observations: {(~data_dict['control_idx_array']).sum()}")
        print(f"  - Missing observations: {data_dict['missing_idx_array'].sum()}")
        
        # =====================================================================
        # STEP 3: Run Bayesian inference
        # =====================================================================
        print("\n[STEP 3] Running Bayesian inference...")
        
        # Determine ranks to test
        if args.rank is not None:
            ranks = [args.rank]
        else:
            ranks = type_config['ranks_to_test']
        
        for rank in ranks:
            print(f"\n{'='*60}")
            print(f"FITTING {model_type.upper()} MODEL WITH RANK {rank}")
            print(f"{'='*60}")
            
            # Run MCMC
            mcmc = run_mcmc_inference(
                data_dict=data_dict,
                rank=rank,
                outcome_dist=config['model']['outcome_distribution'],
                nb_disp=config['model']['nb_disp'],
                sample_disp=config['model']['sample_disp'],
                adjust_for_missingness=config['model']['adjust_for_missingness'],
                model_treated=config['model']['model_treated'],
                num_chains=config['mcmc']['num_chains'],
                num_warmup=config['mcmc']['num_warmup'],
                num_samples=config['mcmc']['num_samples'],
                thinning=config['mcmc']['thinning'],
                random_seed=config['mcmc']['random_seed']
            )
            
            # Generate predictions
            print("\nGenerating posterior predictions...")
            predictions = generate_predictions(
                mcmc=mcmc,
                data_dict=data_dict,
                rank=rank,
                outcome_dist=config['model']['outcome_distribution'],
                nb_disp=config['model']['nb_disp'],
                sample_disp=config['model']['sample_disp'],
                random_seed=config['mcmc']['random_seed']
            )
            
            # =================================================================
            # STEP 4: Process and save results
            # =================================================================
            print("\n[STEP 4] Processing and saving results...")
            
            # Get posterior samples
            samples = mcmc.get_samples(group_by_chain=True)

            # Build a single merged draws + observed data table (R-style)
            samples_dict = {'mu_ctrl': samples['mu_ctrl']}
            if 'te' in samples:
                samples_dict['te'] = samples['te']

            merged = merge_draws_and_data(
                samples=samples_dict,
                predictions=predictions,
                df_long=df_long,
                sub_groups=sub_groups,
                add_total_category=False,
                add_ban_states=False,  # Will be added in QMD like dobbs
            )

            # Save merged results with dobbs-style naming
            # Format: NB_births_{type}_{rank}.csv
            outcome_type = config['model']['outcome_type']
            dist = config['model']['outcome_distribution']

            results_file = output_dir / f'{dist}_{outcome_type}_{model_type}_{rank}.csv'
            merged.to_csv(results_file, index=False)
            print(f"Saved merged draws+data to {results_file}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review MCMC diagnostics (convergence, ESS, Rhat)")
    print("  2. Update nativity_paper_figures.qmd to load and stitch models")
    print("  3. Run: quarto render nativity_paper_figures.qmd")


if __name__ == '__main__':
    main()
