# Bayesian Panel NMF

Python package for causal inference in panel data using Bayesian hierarchical models with low-rank matrix factorization. credit: Professor Alex Franks (UCSB, Statistics & Applied Probability)

## Overview

This package provides tools for estimating treatment effects from panel data (e.g., state-time observations) using Bayesian hierarchical models. Key features:

- **Flexible data format**: Works with any panel data via schema-based configuration
- **Low-rank factorization**: Captures complex state-time interactions efficiently
- **Treatment effect estimation**: Hierarchical treatment effects with uncertainty quantification
- **Configurable priors**: Customize prior distributions via YAML
- **Multiple output formats**: Full posterior draws and summary statistics
- **Parallel execution**: Optional joblib parallelization for multiple analyses

## Installation

```bash
cd bayesian_panel_nmf
pip install -e .
```

### Requirements

- Python 3.8+
- NumPyro >= 0.12.0
- JAX >= 0.4.0
- Pandas >= 1.3.0
- NumPy >= 1.20.0
- PyYAML >= 5.4
- Joblib >= 1.0.0

## Quick Start

### Running with the Included Nativity Data

The package includes sample data and a ready-to-use configuration for testing:

```bash
# Navigate to the package directory
cd bayesian_panel_nmf

# Install the package
pip install -e .

# Run the nativity analysis with the included config and data
python scripts/run_analysis.py --config configs/nativity_config.yaml
```

This will:
1. Load the nativity data from `data/raw/nativity_analyticdata.csv`
2. Run the Bayesian panel NMF model
3. Output results to the `results/` directory

For a quick test run with fewer iterations, use the test config:

```bash
python scripts/run_analysis.py --config configs/test_config.yaml
```

---

### Using Your Own Data

### 1. Prepare Your Data

Your panel data should be a CSV with:
- A **unit** column (e.g., state, firm, county)
- A **time** column (date/timestamp)
- A **treatment** column (0/1 indicator)
- One or more **outcome** columns
- Optional **denominator** columns (for rate calculations)

### 2. Create a Configuration File

```yaml
# my_config.yaml
data:
  input_file: "data/my_panel_data.csv"
  output_dir: "results"
  
  schema:
    unit_col: "state"
    time_col: "date"
    treatment_col: "treated"
    outcomes:
      - outcome_col: "sales"
        denominator_col: "population"
        label: "total"

model:
  outcome_distribution: "NB"
  rank: 10
  types:
    total:
      groups: ["total"]
      ranks_to_test: [10]

mcmc:
  num_chains: 4
  num_warmup: 1000
  num_samples: 2500
  thinning: 10
```

### 3. Run Analysis

```bash
python scripts/run_analysis.py --config configs/my_config.yaml
```

## Configuration Options

### Data Schema

```yaml
data:
  schema:
    unit_col: "state"           # Panel unit identifier
    time_col: "time"            # Time column
    treatment_col: "treated"    # Treatment indicator (0/1)
    outcomes:                   # List of outcomes to analyze
      - outcome_col: "y1"
        denominator_col: "pop1"  # Optional
        label: "group1"
    additional_cols:            # Extra columns to preserve
      - "region"
      - "category"
```

### Temporal Aggregation

```yaml
data:
  aggregation:
    enabled: true
    period: "bimonthly"  # monthly, bimonthly, quarterly, yearly
```

### Custom Priors

```yaml
priors:
  time_factor:
    distribution: "Gamma"
    alpha: 20.0
    beta: 20.0
  treatment_state_scale:
    distribution: "HalfNormal"
    scale: 1.0
```

Supported distributions: `Gamma`, `HalfNormal`, `Normal`, `HalfCauchy`, `Exponential`, `InverseGamma`, `Uniform`, `Beta`, `StudentT`, `Laplace`

### Parallelization

```yaml
parallel:
  num_workers: 4  # 1 = sequential, -1 = all cores
```

### Output Options

```yaml
output:
  save_draws: true    # Full posterior draws (large files)
  save_summary: true  # Summary statistics (small files)
```

## Output Format

### Full Draws (`*_draws.csv`)

Tidy format compatible with R's tidybayes:

| .draw | .chain | category | state | time | outcome | ypred | mu | mu_treated |
|-------|--------|----------|-------|------|---------|-------|-----|------------|
| 1 | 1 | total | AL | 2020-01 | 1000 | 987 | 6.89 | 6.91 |

### Summary (`*_summary.csv`)

Pre-computed statistics:

| category | state | time | ypred_mean | ypred_lower | ypred_upper | te_mean | te_lower | te_upper |
|----------|-------|------|------------|-------------|-------------|---------|----------|----------|
| total | AL | 2020-01 | 987 | 950 | 1024 | 0.02 | -0.01 | 0.05 |

## R Integration

The output format is designed for seamless R integration:

```r
library(tidyverse)

# Load draws
draws_df <- read_csv("results/NB_births_total_10.csv")

# Or load summary for quick plots
summary_df <- read_csv("results/NB_births_total_10_summary.csv")

# Use with tidybayes
library(tidybayes)
draws_df %>%
  group_by(state, category) %>%
  summarize(
    mean_effect = mean(mu_treated - mu),
    ci_lower = quantile(mu_treated - mu, 0.025),
    ci_upper = quantile(mu_treated - mu, 0.975)
  )
```

## Examples

### Nativity Analysis (Dobbs Fertility)

Original use case analyzing birth rates by nativity status:

```bash
python scripts/run_analysis.py --config configs/nativity_config.yaml
```

### Custom Analysis

```python
from bayesian_panel_nmf import (
    load_panel_data, 
    DataSchema,
    preprocess_pipeline,
    run_mcmc_inference,
    model
)

# Load data with custom schema
schema = DataSchema(
    unit_col="state",
    time_col="time",
    treatment_col="treatment",
    outcomes=[OutcomeSpec("deaths", "births", "etc")]
)
df = load_panel_data("my_data.csv", schema=schema)

# Preprocess
data_dict = preprocess_pipeline(df, groups=["total"], config=config)

# Run inference
mcmc = run_mcmc_inference(data_dict, model, rank=10, **mcmc_config)
```

## Citation

If you use this package, please cite:

```bibtex
@article{dobbs_fertility_2024,
  title={Impact of Abortion Bans on Fertility Rates},
  author={Alex Franks, UCSB},
  journal={JAMA},
  year={2024}
}
```

## License

MIT License

---

## To-Do / Roadmap

The following features are planned for future development:
- [ ] **GPU Support**: Add GPU acceleration via JAX for faster MCMC inference on large datasets
- [ ] **Spillover Analysis**: Implement scripts for testing potential spillover effects between treated and control units
- [ ] **Sensitivity Analysis**: Add donor pool sensitivity tests (excluding neighboring states from control group)
- [ ] **R Graphing Generalization**: Create generalized R plotting functions that work with any panel data schema (not hardcoded to specific column names)
- [ ] **Unit Testing and Debugging**: Add comprehensive unit tests and debugging utilities for the package
