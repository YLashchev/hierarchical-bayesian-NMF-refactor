# Nativity Analysis

A professional statistical software package for analyzing birth patterns by nativity status using Bayesian hierarchical panel models.

## Overview

This package analyzes the impact of abortion policy changes on birth rates, stratified by mother's nativity status (US-born vs. foreign-born). It uses a Bayesian hierarchical panel model with non-negative matrix factorization to estimate causal effects.

## Project Structure

```
nativity_analysis/
├── src/nativity_analysis/     # Main package code
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Bayesian panel NMF model
│   ├── inference/              # MCMC sampling and postprocessing
│   └── visualization/          # Figure generation
├── scripts/                    # Executable scripts
│   ├── run_full_analysis.py   # Complete pipeline
│   └── generate_figures.py    # Figure generation only
├── configs/                    # Configuration files
├── data/                       # Input data
├── results/                    # Model outputs
└── figures/                    # Generated figures
```

## Installation

### 1. Create Virtual Environment

```bash
# Navigate to nativity_analysis directory
cd nativity_analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Package in Development Mode

```bash
pip install -e .
```

## Quick Start

### Generate Preliminary Figures (No Model Fitting)

To quickly generate descriptive figures without running the computationally intensive Bayesian model:

```bash
# Activate virtual environment
source venv/bin/activate

# Run figure generation on preprocessed data
python scripts/generate_figures.py
```

This will:
1. Load the nativity data
2. Preprocess to long format
3. Generate Figure 1 (relative birth rates) and Figure 2 (absolute fertility)

### Run Full Analysis

To run the complete Bayesian analysis pipeline:

```bash
# Activate virtual environment
source venv/bin/activate

# Run full analysis with a single rank (faster for testing)
python scripts/run_full_analysis.py --rank 10

# Or run with all ranks specified in config (slower, for final analysis)
python scripts/run_full_analysis.py
```

## Configuration

Edit `configs/nativity_config.yaml` to customize:

- **Data settings**: Time periods, nativity groups
- **Model settings**: Distribution type, rank values, dispersion
- **MCMC settings**: Chains, warmup, samples, thinning
- **Figure settings**: Output format, styling

## Workflow

### 1. Data Preprocessing

The pipeline converts wide-format monthly data to long format and aggregates to bimonthly periods:

```python
from nativity_analysis.data import (
    load_nativity_data,
    wide_to_long,
    aggregate_to_bimonthly,
    create_exposure_codes
)

# Load raw data
df = load_nativity_data('data/nativity_analyticdata.csv')

# Convert to long format
df_long = wide_to_long(df, nativity_groups=['usborn', 'foreign'])

# Aggregate to bimonthly
df_bm = aggregate_to_bimonthly(df_long)

# Create exposure codes
df_bm = create_exposure_codes(df_bm)
```

### 2. Bayesian Inference

Run MCMC sampling on the panel NMF model:

```python
from nativity_analysis.inference import run_mcmc_inference, generate_predictions
from nativity_analysis.data import prepare_model_data

# Prepare data for modeling
data_dict = prepare_model_data(df_bm, nativity_groups=['usborn', 'foreign'])

# Run MCMC
mcmc = run_mcmc_inference(
    data_dict=data_dict,
    rank=10,
    num_chains=4,
    num_samples=2500
)

# Generate counterfactual predictions
predictions = generate_predictions(mcmc, data_dict, rank=10)
```

### 3. Visualization

Generate publication-ready figures:

```python
from nativity_analysis.visualization import (
    plot_relative_birth_rate,
    plot_absolute_fertility
)

# Figure 1: Relative birth rates
plot_relative_birth_rate(df_bm, nativity_group='foreign')

# Figure 2: Absolute fertility
plot_absolute_fertility(df_bm, nativity_group='foreign')
```

## Key Features

### Data Processing
- **Flexible format conversion**: Wide ↔ Long format
- **Time aggregation**: Monthly → Bimonthly
- **Exposure coding**: Automatic treatment period identification
- **Validation**: Built-in data quality checks

### Bayesian Modeling
- **Hierarchical panel model**: Accounts for state and time effects
- **Matrix factorization**: Low-rank approximation for efficiency
- **Missingness handling**: Adjusts for suppressed/missing data
- **Treatment effects**: Estimates causal impacts of policy changes

### Visualization
- **Professional styling**: Publication-ready figures
- **Consistent colors**: Matches main analysis aesthetic
- **Informative annotations**: Exposure timing marked clearly

## Model Details

The Bayesian hierarchical panel model decomposes birth rates as:

```
log(births) = StateEffect + TimeEffect + LowRankFactors + TreatmentEffect + error
```

Where:
- **StateEffect**: State-specific baseline
- **TimeEffect**: Time-specific trends
- **LowRankFactors**: Low-dimensional representation of state-time interactions
- **TreatmentEffect**: Causal effect of abortion bans
- **error**: Negative Binomial or Poisson noise

## Output Files

### Results Directory
- `preprocessed_nativity_data.csv`: Long format preprocessed data
- `df_nativity_usborn_foreign.csv`: Model input data
- `NB_births_usborn_foreign_[rank]_nativity.csv`: Posterior samples

### Figures Directory
- `relative_birthrate_[group].png`: Figure 1 for each nativity group
- `absolute_fertility_[group].png`: Figure 2 for each nativity group

## Computational Requirements

- **Memory**: ~8-16 GB RAM recommended
- **Time**: 
  - Single rank (10): ~30-60 minutes
  - All ranks (6-12): ~3-5 hours
- **Cores**: Utilizes all available cores for parallel chains

## Troubleshooting

### Import Errors
If you encounter import errors, ensure:
1. Virtual environment is activated
2. Package is installed: `pip install -e .`
3. You're running from the nativity_analysis directory

### JAX/NumPyro Issues
For CPU-only systems, install JAX with:
```bash
pip install jax[cpu]
```

For GPU support, see [JAX installation guide](https://github.com/google/jax#installation).

### Memory Issues
If you run out of memory:
- Reduce `num_chains` in config
- Reduce `num_samples` in config
- Process one rank at a time

## Citation

This code implements the methods described in:

[Paper citation to be added]

## License

[License information to be added]

## Contact

For questions or issues, please contact: [contact information]
