"""
Main script for running the Bayesian Panel NMF analysis pipeline.

This script:
1. Loads and preprocesses panel data using schema-based configuration
2. Runs Bayesian inference for specified model types and ranks
3. Saves results in tidy format

Usage:
    python scripts/run_analysis.py --config configs/nativity_config.yaml
    python scripts/run_analysis.py --config configs/nativity_config.yaml --type groups --rank 10
"""

import argparse
import yaml
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bayesian_panel_nmf.data import (
    load_panel_data,
    wide_to_long,
    preprocess_pipeline,
    DataSchema
)
from bayesian_panel_nmf.models import model
from bayesian_panel_nmf.inference import (
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
    print("="*60)
    print("BAYESIAN PANEL NMF ANALYSIS")
    print("="*60)
    
    data_config = config['data']
    model_config = config['model']
    mcmc_config = config['mcmc']
    output_config = config.get('output', {})
    
    # Setup output directory
    output_dir = Path(data_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which types to run
    if args.type is not None:
        types_to_run = [args.type]
    else:
        types_to_run = list(model_config['types'].keys())
    
    print(f"\nModel types to run: {types_to_run}")
    
    # Create schema from config
    schema = DataSchema.from_config(config)
    
    # =========================================================================
    # STEP 1: Load raw data
    # =========================================================================
    print("\n[STEP 1] Loading data...")
    
    df_raw = load_panel_data(
        filepath=data_config['input_file'],
        schema=schema,
        validate=True
    )
    
    # Loop over model types
    for model_type in types_to_run:
        print("\n" + "="*60)
        print(f"RUNNING MODEL TYPE: {model_type.upper()}")
        print("="*60)
        
        type_config = model_config['types'][model_type]
        groups = type_config['groups']
        
        # =====================================================================
        # STEP 2: Convert to long format and preprocess
        # =====================================================================
        print(f"\n[STEP 2] Preprocessing data for groups: {groups}")
        
        # Convert to long format
        df_long = wide_to_long(df_raw, schema=schema)
        
        # Handle special "total" group - aggregate all outcomes
        if groups == ["total"] and "total" not in df_long['group'].unique():
            print("  -> Creating 'total' group by aggregating all outcomes...")
            # Aggregate all groups into a single "total" group
            df_total = df_long.groupby(['unit', 'time', 'treatment'], as_index=False).agg({
                'outcome': 'sum',
                'denominator': 'sum' if 'denominator' in df_long.columns else 'first',
                **{col: 'first' for col in df_long.columns 
                   if col not in ['unit', 'time', 'treatment', 'group', 'outcome', 'denominator']}
            })
            df_total['group'] = 'total'
            df_long = df_total
            print(f"  -> Aggregated into {len(df_long)} total rows")
        else:
            # Filter to requested groups
            df_long = df_long[df_long['group'].isin(groups)].copy()
        
        # Override aggregation if requested
        if args.no_aggregate:
            config_copy = config.copy()
            config_copy['data'] = config['data'].copy()
            config_copy['data']['aggregation'] = {'enabled': False}
        else:
            config_copy = config
        
        # Run preprocessing pipeline
        data_dict = preprocess_pipeline(
            df=df_long,
            groups=groups,
            config=config_copy,
            outcome_col='outcome',
            denominator_col='denominator' if 'denominator' in df_long.columns else None,
            unit_col='unit',
            time_col='time',
            group_col='group',
            treatment_col='treatment'
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
        
        # =====================================================================
        # STEP 3: Run inference for each rank
        # =====================================================================
        print(f"\n[STEP 3] Running Bayesian inference...")
        
        # Determine ranks to test
        if args.rank is not None:
            ranks = [args.rank]
        else:
            ranks = type_config.get('ranks_to_test', [10])
        
        for rank in ranks:
            print(f"\n{'='*60}")
            print(f"FITTING {model_type.upper()} MODEL WITH RANK {rank}")
            print(f"{'='*60}")
            
            # Run MCMC
            mcmc = run_mcmc_inference(
                data_dict=data_dict,
                model_fn=model,
                rank=rank,
                outcome_dist=model_config.get('outcome_distribution', 'NB'),
                nb_disp=model_config.get('nb_disp', 1e-4),
                sample_disp=model_config.get('sample_disp', False),
                adjust_for_missingness=model_config.get('adjust_for_missingness', True),
                model_treated=model_config.get('model_treated', True),
                num_chains=mcmc_config.get('num_chains', 4),
                num_warmup=mcmc_config.get('num_warmup', 1000),
                num_samples=mcmc_config.get('num_samples', 2500),
                thinning=mcmc_config.get('thinning', 10),
                random_seed=mcmc_config.get('random_seed', 8675309),
                progress_bar=mcmc_config.get('progress_bar', True)
            )
            
            # Generate predictions
            print("\nGenerating posterior predictions...")
            predictions = generate_predictions(
                mcmc=mcmc,
                model_fn=model,
                data_dict=data_dict,
                rank=rank,
                outcome_dist=model_config.get('outcome_distribution', 'NB'),
                nb_disp=model_config.get('nb_disp', 1e-4),
                sample_disp=model_config.get('sample_disp', False),
                random_seed=mcmc_config.get('random_seed', 8675309)
            )
            
            # =================================================================
            # STEP 4: Process and save results
            # =================================================================
            print("\n[STEP 4] Processing and saving results...")
            
            # Get posterior samples
            samples = mcmc.get_samples(group_by_chain=True)
            
            samples_dict = {'mu_ctrl': samples['mu_ctrl']}
            if 'te' in samples:
                samples_dict['te'] = samples['te']
            
            # Merge draws with data
            merged = merge_draws_and_data(
                samples=samples_dict,
                predictions=predictions,
                df_preprocessed=df_preprocessed,
                groups=data_dict['groups'],
                units=data_dict['units'],
                times=data_dict['times'],
                unit_col='unit',
                time_col='time',
                group_col='group',
                outcome_col='outcome',
                denominator_col='denominator' if 'denominator' in df_preprocessed.columns else None,
                treatment_col='treatment'
            )
            
            # Save results using pattern from config
            dist = model_config.get('outcome_distribution', 'NB')
            pattern = output_config.get('filename_pattern', '{distribution}_{type}_{rank}')
            
            filename = pattern.format(
                distribution=dist,
                outcome='births',  # Default for nativity
                type=model_type,
                rank=rank
            )
            
            results_file = output_dir / f'{filename}.csv'
            merged.to_csv(results_file, index=False)
            print(f"Saved results to: {results_file}")
            
            # Save summary if requested
            if output_config.get('save_summary', True):
                from bayesian_panel_nmf.inference.postprocessing import compute_summary_statistics
                summary = compute_summary_statistics(merged)
                summary_file = output_dir / f'{filename}_summary.csv'
                summary.to_csv(summary_file, index=False)
                print(f"Saved summary to: {summary_file}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review MCMC diagnostics (convergence, ESS, Rhat)")
    print("  2. Load results in R/Python for visualization")
    print("  3. Generate figures with downstream analysis scripts")


if __name__ == '__main__':
    main()
