# Bayesian Panel NMF - Developer Guide

## 1. Project Overview

This repository implements **Bayesian hierarchical panel models with low-rank factorization** for causal inference using JAX and NumPyro. The framework is designed to be **dataset-agnostic** - any panel data with similar structure should work via configuration alone.

### Core Statistical Model
```
log(outcome) = unit_fe + time_fe + low_rank_factors + treatment_effect + error
```
Where treatment effects are hierarchical: `te = unit_te + category_te + unit×category_te + time_te`

### Design Philosophy
- **Configuration over code**: New datasets should require only a YAML config, not code changes
- **Pre-cleaned data assumption**: Users are responsible for data cleaning/imputation before using this framework
- **Standardized outputs**: All outputs use consistent column names regardless of input format

## 2. Environment & Commands

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install in editable mode with visualization extras
pip install -e ".[viz]"
```

### Running Analysis
```bash
# Full analysis (uses nativity_config.yaml)
python scripts/run_analysis.py --config configs/nativity_config.yaml

# Quick test with specific rank
python scripts/run_analysis.py --config configs/nativity_config.yaml --type total --rank 5

# Run specific model type
python scripts/run_analysis.py --type groups --rank 10

# Skip temporal aggregation (use monthly data)
python scripts/run_analysis.py --no-aggregate
```

### Testing
**Note:** Formal test suite (`tests/`) is not yet implemented.
```bash
# When tests are added:
pytest                                    # Run all tests
pytest tests/test_preprocessing.py        # Run specific file
pytest tests/test_loader.py::test_wide_to_long  # Run single test
pytest -v --tb=short                      # Verbose with short traceback
```

### Linting & Formatting
```bash
ruff check .                  # Lint check
ruff check . --fix            # Auto-fix issues
black src/ scripts/           # Format code (if installed)
```

## 3. Code Style & Conventions

### Imports
Group imports in this order: Standard library → Third-party → Local
```python
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

from bayesian_panel_nmf import (
    load_and_prepare,
    run_mcmc_inference,
    generate_predictions,
    format_draws,
    model
)
```

### Type Hints
Strongly encouraged, especially for public APIs:
```python
def run_mcmc_inference(
    data_dict: Dict[str, np.ndarray],
    model_fn: Callable,
    rank: int = 10,
    num_chains: int = 4,
) -> MCMC:
    ...
```

### Docstrings
Use **NumPy/SciPy style**. See `src/bayesian_panel_nmf/inference.py` for examples:
```python
def run_mcmc_inference(data_dict, model_fn, rank=10):
    """
    Run MCMC inference on the panel NMF model.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing model data with keys:
        - Y: outcome array (K x D x N)
        - denominators: population array (K x D x N)
        - control_idx_array: boolean control mask (K x D x N)
    model_fn : callable
        NumPyro model function
    rank : int, default=10
        Rank for matrix factorization
        
    Returns
    -------
    MCMC
        Fitted MCMC object containing posterior samples
    """
```

### Naming Conventions
| Type | Convention | Examples |
|------|------------|----------|
| Functions/Variables | `snake_case` | `run_mcmc_inference`, `control_idx_array` |
| Classes | `PascalCase` | `DataSchema`, `MCMC` |
| Constants | `UPPER_SNAKE` | `DEFAULT_RANK`, `NUM_CHAINS` |
| Array dimensions | Capital letters | `K` (groups), `D` (units), `N` (time periods) |

### Error Handling
- Validate inputs at function boundaries
- Use descriptive error messages with context
- Prefer explicit checks over silent failures
```python
if not all(col in df.columns for col in required_cols):
    missing = set(required_cols) - set(df.columns)
    raise ValueError(f"Missing required columns: {missing}")
```

## 4. JAX & NumPyro Patterns

### Array Immutability
JAX arrays are immutable. Use functional update syntax:
```python
# WRONG: arr[idx] = val
# RIGHT:
arr = arr.at[idx].set(val)
arr = arr.at[~control_idx].add(treatment_effect)
```

### Random Key Management
Always explicitly split PRNG keys:
```python
rng_key = random.PRNGKey(8675309)
rng_key, rng_key_ = random.split(rng_key)
mcmc.run(rng_key_, ...)
```

### NumPyro Plates
Use `numpyro.plate` for independence assumptions in the model:
```python
with numpyro.plate('K', K):           # Groups
    with numpyro.plate('D', D):       # Units
        unit_fe = numpyro.sample('unit_fe', dist.ImproperUniform(...))
```

### Broadcasting
Use explicit `None` indexing for broadcasting:
```python
fixed_effects = unit_fe[:, :, None] + time_fe[:, None, :]  # (K, D, N)
```

### Deterministic Sites
Use `numpyro.deterministic` to track computed quantities:
```python
mu = numpyro.deterministic('mu', f_all + te)
```

## 5. Directory Structure

```
bayesian_panel_nmf/
├── configs/
│   ├── base_config.yaml        # Default settings template
│   ├── nativity_config.yaml    # Nativity analysis config
│   └── test_config.yaml        # Quick testing (rank 5)
├── data/raw/                   # Input CSV files
├── results/                    # Output: CSVs with posterior draws
├── figs/                       # Output: Generated figures
├── reference_directories/      # Original implementations (for reference only)
│   ├── dobbs_fertility/        # Original race/age/edu analysis
│   └── nativity_analysis/      # Original nativity analysis
├── scripts/
│   └── run_analysis.py         # Main entry point
├── src/bayesian_panel_nmf/
│   ├── __init__.py             # Package exports
│   ├── data.py                 # Combined loading + preparation (load_and_prepare)
│   ├── inference.py            # Simplified MCMC execution (run_mcmc_inference, generate_predictions)
│   ├── output.py               # Draw formatting (format_draws)
│   ├── visualization.py        # PPC plots (make_*_ppc_plot functions)
│   └── models/
│       ├── __init__.py         # Models subpackage exports
│       ├── panel_nmf_model.py  # Core Bayesian model (model function)
│       ├── priors.py           # Prior configurations and distribution helpers
│       └── utils.py            # Missingness adjustment
├── nativity_analysis.qmd       # R/Quarto visualization
└── plot_utilities.R            # R plotting functions
```

## 6. Data Flow

```
Config (YAML)
        ↓
load_and_prepare(filepath, config, groups) → data_dict
        ↓
run_mcmc_inference(data_dict, model, rank, config) → MCMC
        ↓
generate_predictions(mcmc, data_dict, model, rank, config) → predictions
        ↓
format_draws(samples, predictions, data_dict) → DataFrame
        ↓
Results CSV
```

## 7. Key Data Structures

### data_dict (model input)
```python
data_dict = {
    'Y': np.ndarray,                # (K, D, N) - outcome counts
    'denominators': np.ndarray,     # (K, D, N) - population/exposure
    'control_idx_array': np.ndarray,  # (K, D, N) - bool, True=control period
    'missing_idx_array': np.ndarray,  # (K, D, N) - bool, True=missing
    'groups': List[str],            # Group names (K dimension)
    'units': List[str],             # Unit names (D dimension)
    'times': List[str],             # Time period labels (N dimension)
    'df_preprocessed': pd.DataFrame # Long-format data
}
```

### MCMC samples
```python
samples = mcmc.get_samples(group_by_chain=True)
# samples['mu_ctrl']: (chains, samples, K, D, N)
# samples['te']: (chains, samples, K, D, N)
```

## 8. Configuration (YAML)

```yaml
data:
  input_file: "data/raw/panel_data.csv"
  output_dir: "results"
  start_date: "2016-01-01"
  end_date: "2024-01-01"
  
  schema:
    unit_col: "state"
    time_col: "time"
    treatment_col: "treated"
    outcomes:
      - outcome_col: "births_total"
        denominator_col: "pop_total"
        label: "total"
  
  aggregation:
    enabled: true
    period: "bimonthly"  # monthly, bimonthly, quarterly

model:
  outcome_distribution: "NB"  # NB or Poisson
  nb_disp: 1.0e-4
  adjust_for_missingness: true
  model_treated: true
  types:
    total:
      groups: ["total"]
      ranks_to_test: [10]

mcmc:
  num_chains: 4
  num_warmup: 1000
  num_samples: 2500
  thinning: 10
  random_seed: 8675309
```

## 9. Critical Implementation Notes

### Missingness Adjustment
Outcome counts below a threshold may be suppressed for privacy. The model explicitly adjusts for this censoring:
```python
# In panel_nmf_model.py
scope(missingness_adjustment, "low_births")(
    mu, missing_idx, control_idx, 
    jnp.array([1,2,3,4,5,6,7,8,9]),  # Possible suppressed values
    outcome_dist, dispersion
)
```

### Treatment Effect Hierarchy
```python
te = treatment_kt +                    # Time-specific within treated
     unit_treatment_effect +           # Unit-specific
     category_treatment_effect +       # Category-specific
     unit_category_te                  # Interaction
```

### Model Refactoring Rules
When modifying `panel_nmf_model.py`:
1. **Preserve mathematical consistency** - the low-rank factorization structure must remain valid
2. **Keep plate structure intact** - independence assumptions are encoded in plates
3. **Test convergence** - any prior changes should be validated with R-hat < 1.01, ESS > 400

## 10. Codebase Simplification Refactor (COMPLETED)

**Status:** Completed on February 9, 2026.

The refactor successfully consolidated the codebase from ~1,724 lines across 6+ files to ~735 lines across 4 files (a ~57% reduction). The deprecated `data/` and `inference/` subdirectories have been removed and replaced with the simplified flat module structure described in Section 5.

### 10.1 Problem Diagnosis

The config-driven approach is architecturally correct, but the implementation doesn't commit to standardization. Here's the current anti-pattern:

1. Config specifies user's column names (e.g., `unit_col: "state"`, `outcome_col: "birthshisp_usborn"`)
2. `wide_to_long()` converts to standardized names: `unit`, `time`, `group`, `outcome`, `denominator`, `treatment`
3. **BUT** every downstream function still accepts column name parameters that are always the same values
4. **AND** postprocessing renames them again to R-specific names (`state`, `population`, `exposure_code`)

**Result:** ~1,400 lines across 6 files when ~400-500 lines across 3-4 files would suffice.

### 10.2 Why This Approach is Correct for Generalization

The goal is to support diverse panel datasets (fertility, hospital readmissions, crime data, sales, etc.) via configuration alone. This requires:

| Requirement | How It's Handled |
|-------------|------------------|
| Different column names | Config maps user columns → standard names |
| Multiple outcome groups | Config lists outcome/denominator pairs with labels |
| Different time granularities | Config specifies aggregation period |
| Different distributions | Config specifies NB vs Poisson |
| Different MCMC settings | Config specifies chains, warmup, samples |

**The model itself is already general** - `panel_nmf_model.py` operates on (K, D, N) arrays and doesn't care about column names. The simplification removes unnecessary complexity in the data pipeline without reducing flexibility.

### 10.3 The Solution

**Core Principle:** Once `wide_to_long()` standardizes column names, they are **FIXED**. No more passing them around.

### 10.4 Target Architecture

```
Config (YAML)
    ↓
load_and_prepare(filepath, config, groups) → data_dict with FIXED columns:
    - df_preprocessed: DataFrame with columns {unit, time, group, outcome, denominator, treatment}
    - Y, denominators, masks: NumPy arrays (K×D×N)
    ↓
run_mcmc_inference(data_dict, config) → MCMC object
    ↓
generate_predictions(mcmc, data_dict, config) → predictions array
    ↓
format_draws(samples, predictions, data_dict) → DataFrame with FIXED output columns
```

### 10.5 File Structure Comparison

| Old Files | Lines | New Files | Lines | Notes |
|-----------|-------|-----------|-------|-------|
| `data/schema.py` | ~186 | (removed) | — | Inlined into data.py |
| `data/loader.py` | ~249 | `data.py` | ~250 | Combined loading + prep |
| `data/preprocessing.py` | ~348 | (merged above) | — | |
| `inference/postprocessing.py` | ~353 | `output.py` | ~150 | Simplified formatting |
| `inference/sampler.py` | ~317 | `inference.py` | ~200 | Simplified, removed col params |
| `scripts/run_analysis.py` | ~271 | (simplified) | ~135 | Cleaner main loop |
| **Total** | **~1,724** | **Actual** | **~735** | **~57% reduction** |

### 10.6 Standard Column Names

#### Internal DataFrame (after `load_and_prepare()`)

All code after loading uses these **fixed names**—no parameters needed:

| Column | Type | Description |
|--------|------|-------------|
| `unit` | str | Panel entity (was user's `unit_col`) |
| `time` | datetime | Time period (parsed from user's `time_col`) |
| `group` | str | Outcome category label |
| `outcome` | numeric | Outcome value |
| `denominator` | numeric | Population/exposure (may be NaN if not provided) |
| `treatment` | int | Binary 0/1 |

#### Output DataFrame (from `format_draws()`)

| Column | Type | Description |
|--------|------|-------------|
| `.draw` | int | Draw number (1-indexed) |
| `.chain` | int | Chain number (1-indexed) |
| `.iteration` | int | Iteration within chain |
| `unit` | str | Panel entity |
| `time` | datetime | Time period |
| `group` | str | Outcome category |
| `outcome` | numeric | Observed outcome |
| `denominator` | numeric | Population/exposure |
| `treatment` | int | Binary 0/1 |
| `ypred` | numeric | Posterior predictive draw |
| `mu` | numeric | Counterfactual log-rate |
| `mu_treated` | numeric | Treated log-rate (mu + te) |

### 10.7 Implementation Tasks

All tasks were completed as part of this refactor. The following documents what was implemented:

#### Task 1: Created `src/bayesian_panel_nmf/data.py` ✓

**Purpose:** Single module for all data loading and preparation.

**Main function:**

```python
def load_and_prepare(
    filepath: str,
    config: dict,
    groups: List[str]
) -> dict:
    """
    Load CSV, standardize columns, filter, aggregate, and prepare model arrays.
    
    Parameters
    ----------
    filepath : str
        Path to input CSV file
    config : dict
        Full config dict (data, model, mcmc sections)
    groups : list of str
        Which outcome groups to include (e.g., ["total"] or ["hisp_usborn", "hisp_foreign"])
    
    Returns
    -------
    dict
        data_dict with keys:
        - Y: ndarray (K, D, N) outcome counts
        - denominators: ndarray (K, D, N) populations
        - control_idx_array: ndarray (K, D, N) bool
        - missing_idx_array: ndarray (K, D, N) bool
        - groups: list of str (K labels)
        - units: list of str (D labels)
        - times: list of datetime (N labels)
        - df_preprocessed: DataFrame with standardized columns
    """
```

**Internal helper functions (private, no column name params):**

```python
def _parse_schema(config: dict) -> dict:
    """Extract column mapping from config['data']['schema']."""
    
def _load_and_standardize(filepath: str, schema: dict) -> pd.DataFrame:
    """Load CSV and rename to standard columns (unit, time, group, outcome, denominator, treatment)."""
    
def _filter_time_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Filter to start_date/end_date range."""
    
def _aggregate_temporal(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Aggregate to bimonthly/quarterly if configured."""
    
def _build_model_arrays(df: pd.DataFrame, groups: List[str]) -> dict:
    """Convert DataFrame to K×D×N arrays."""
```

**Key simplification:** All internal functions use fixed column names—no column name parameters.

---

#### Task 2: Created `src/bayesian_panel_nmf/output.py` ✓

**Purpose:** Single module for formatting MCMC output.

**Main function:**

```python
def format_draws(
    samples: dict,
    predictions: np.ndarray,
    data_dict: dict
) -> pd.DataFrame:
    """
    Merge MCMC draws with observed data into tidy DataFrame.
    
    Parameters
    ----------
    samples : dict
        MCMC samples from mcmc.get_samples(group_by_chain=True)
        Must contain 'mu_ctrl', optionally 'te'
    predictions : np.ndarray
        Posterior predictive samples, shape (C, S, K, D, N)
    data_dict : dict
        Output from load_and_prepare()
    
    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns:
        .draw, .chain, .iteration, unit, time, group, 
        outcome, denominator, treatment, ypred, mu, mu_treated
    """
```

**Key simplification:** 
- No column name parameters
- No R-specific renaming (`state`, `population`, `exposure_code`)
- Uses fixed standardized column names throughout

---

#### Task 3: Simplified `src/bayesian_panel_nmf/inference.py` ✓

**Purpose:** Keep inference logic, remove column name clutter.

**Functions (simplified signatures):**

```python
def run_mcmc_inference(
    data_dict: dict,
    model_fn: Callable,
    rank: int,
    config: dict
) -> MCMC:
    """Run MCMC inference. Config provides mcmc settings and model options."""

def generate_predictions(
    mcmc: MCMC,
    data_dict: dict,
    model_fn: Callable,
    rank: int,
    config: dict
) -> np.ndarray:
    """Generate posterior predictive samples."""
```

**Removed:**
- Any column name parameters
- Redundant data reshaping (done in `data.py`)

---

#### Task 4: Simplified `scripts/run_analysis.py` ✓

**Implemented structure:**

```python
def main():
    args = parse_args()
    config = load_config(args.config)
    
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_type, type_config in config['model']['types'].items():
        if args.type and args.type != model_type:
            continue
            
        groups = type_config['groups']
        ranks = [args.rank] if args.rank else type_config.get('ranks_to_test', [10])
        
        # Load and prepare data (single call)
        data_dict = load_and_prepare(
            config['data']['input_file'],
            config,
            groups
        )
        
        for rank in ranks:
            # Run inference
            mcmc = run_mcmc_inference(data_dict, model, rank, config)
            predictions = generate_predictions(mcmc, data_dict, model, rank, config)
            
            # Format and save output
            samples = mcmc.get_samples(group_by_chain=True)
            draws_df = format_draws(samples, predictions, data_dict)
            
            dist = config['model'].get('outcome_distribution', 'NB')
            filename = f"{dist}_{model_type}_{rank}.csv"
            draws_df.to_csv(output_dir / filename, index=False)
            print(f"Saved: {output_dir / filename}")
```

**Key simplification:**
- No column name extraction from config for downstream functions
- No redundant schema parsing in the main loop
- Clean linear flow: load → infer → format → save

---

#### Task 5: Updated `src/bayesian_panel_nmf/__init__.py` ✓

**Exports:**

```python
from bayesian_panel_nmf.data import load_and_prepare
from bayesian_panel_nmf.inference import run_mcmc_inference, generate_predictions
from bayesian_panel_nmf.output import format_draws
from bayesian_panel_nmf.models import model

__all__ = [
    'load_and_prepare',
    'run_mcmc_inference', 
    'generate_predictions',
    'format_draws',
    'model',
]
```

---

#### Task 6: Kept `models/` unchanged ✓

These files required no changes:
- `models/panel_nmf_model.py` - Core Bayesian model
- `models/priors.py` - Prior configurations  
- `models/utils.py` - Missingness adjustment

They operate on arrays, not DataFrames, so standardized column names don't affect them.

---

#### Task 7: Removed deprecated files ✓

The following files and directories were removed:

```bash
# Removed old data modules
src/bayesian_panel_nmf/data/schema.py
src/bayesian_panel_nmf/data/loader.py
src/bayesian_panel_nmf/data/preprocessing.py
src/bayesian_panel_nmf/data/__init__.py
src/bayesian_panel_nmf/data/

# Removed old inference modules  
src/bayesian_panel_nmf/inference/postprocessing.py
src/bayesian_panel_nmf/inference/sampler.py
src/bayesian_panel_nmf/inference/__init__.py
src/bayesian_panel_nmf/inference/
```

**Current structure:**
```
src/bayesian_panel_nmf/
├── __init__.py
├── data.py              # Combined loading + preparation
├── inference.py         # Simplified MCMC execution
├── output.py            # Draw formatting
└── models/
    ├── __init__.py
    ├── panel_nmf_model.py
    ├── priors.py
    └── utils.py
```

### 10.8 What Config Still Controls

The config remains essential for generalization:

| Section | Controls | Why It Matters |
|---------|----------|----------------|
| `data.schema.unit_col` | Which column is the panel unit | Different datasets use `state`, `hospital_id`, `county`, etc. |
| `data.schema.time_col` | Which column is time | Could be `time`, `date`, `quarter`, etc. |
| `data.schema.treatment_col` | Which column is treatment | Could be `exposed`, `treated`, `post_policy`, etc. |
| `data.schema.outcomes` | Outcome/denominator pairs with labels | Defines the K dimension groups |
| `data.start_date/end_date` | Time range filtering | Different analyses need different windows |
| `data.aggregation.period` | Temporal aggregation | Monthly → bimonthly, quarterly, etc. |
| `model.outcome_distribution` | NB or Poisson | Depends on overdispersion in data |
| `model.nb_disp` | Negative binomial dispersion | Tuning parameter |
| `model.adjust_for_missingness` | Enable censoring adjustment | Some datasets have suppressed counts |
| `model.types` | Which groups to run together | Allows running total, by-race, by-age separately |
| `mcmc.*` | Chains, warmup, samples, thinning, seed | Standard MCMC tuning |

### 10.9 Migration Strategy

**Phase 1: Create new files alongside old**
1. Create `src/bayesian_panel_nmf/data.py` with `load_and_prepare()`
2. Create `src/bayesian_panel_nmf/output.py` with `format_draws()`
3. Create `src/bayesian_panel_nmf/inference.py` (simplified from `sampler.py`)

**Phase 2: Update entry point**
4. Modify `scripts/run_analysis.py` to import from new modules
5. Test with: `python scripts/run_analysis.py --config configs/nativity_config.yaml --type total --rank 5`
6. Verify output columns match specification in Section 10.6

**Phase 3: Clean up**
7. Remove deprecated files (see Task 7)
8. Update `__init__.py` files
9. Run full test: `python scripts/run_analysis.py --config configs/nativity_config.yaml`

### 10.10 Testing the Refactor

```bash
# Quick validation (rank 5 for speed)
python scripts/run_analysis.py --config configs/nativity_config.yaml --type total --rank 5

# Verify output columns
head -1 results/NB_total_5.csv
# Expected: .draw,.chain,.iteration,unit,time,group,outcome,denominator,treatment,ypred,mu,mu_treated

# Verify no hardcoded R-specific values in new code
grep -r "state" src/bayesian_panel_nmf/data.py src/bayesian_panel_nmf/output.py  # Should find nothing
grep -r "population" src/bayesian_panel_nmf/output.py  # Should find nothing  
grep -r "exposure_code" src/bayesian_panel_nmf/output.py  # Should find nothing
```

### 10.11 Future Enhancements (Out of Scope)

The following are explicitly **not** part of this refactor but could be added later:

| Enhancement | Description |
|-------------|-------------|
| Long format input | Support data already in long format with `group` column |
| Staggered treatment | Support different treatment dates per unit |
| Spatial aggregation | Aggregate units (e.g., counties → states) |
| Placebo tests | `run_placebos.py` script |
| Sensitivity analysis | `run_sensitivity.py` script |
| ~~Python visualization~~ | ~~Port R plotting to matplotlib/seaborn~~ (DONE - see Phase 2) |
| GPU support | JAX/CUDA acceleration |

## 11. Development Roadmap

### Phase 1: Codebase Simplification ✅ COMPLETED (Feb 9, 2026)
- [x] Created simplified `data.py` with `load_and_prepare()`
- [x] Created simplified `output.py` with `format_draws()`
- [x] Created simplified `inference.py` with config-driven API
- [x] Updated `run_analysis.py` to use new modules
- [x] Removed deprecated `data/` and `inference/` subdirectories
- [x] Line count reduction: ~1,724 → ~735 lines (~57% reduction)

### Phase 2: Python Visualization (PPC Plots) ✅ COMPLETED (Feb 9, 2026)
Ported Posterior Predictive Check (PPC) plots from R to Python using matplotlib/seaborn.

**Module:** `src/bayesian_panel_nmf/visualization.py`

**Implemented functions:**
| Function | Description |
|----------|-------------|
| `make_abs_ppc_plot()` | Maximum absolute residual comparison |
| `make_acf_ppc_plot()` | Autocorrelation of residuals at specified lag |
| `make_rmse_ppc_plot()` | RMSE comparison (observed vs predicted) |
| `make_unit_corr_ppc_plot()` | Cross-unit correlation (spectral norm) |
| `make_all_ppc_plots()` | Convenience function to generate all PPC plots |

**Key features:**
- Handles BOTH standardized (`unit`, `group`, `treatment`) and legacy (`state`, `category`, `exposure_code`) column names via `_standardize_columns()` helper
- All PPC plots compare observed residuals vs predicted residuals in control period
- Residuals: `obs_diff = outcome - exp(mu)`, `pred_diff = ypred - exp(mu)`
- P-values: proportion of draws where observed statistic < predicted statistic
- Faceted histograms by unit and category with p-value annotations
- Returns `(fig, pvals_df)` tuples for programmatic access
- `make_all_ppc_plots()` can optionally save plots to files

**Usage example:**
```python
from bayesian_panel_nmf import make_all_ppc_plots, make_rmse_ppc_plot

# Generate all PPC plots and save to figs/
results = make_all_ppc_plots(draws_df, output_dir='figs/')

# Generate individual plot
fig, pvals = make_rmse_ppc_plot(draws_df, categories=['total'])
fig.savefig('rmse_check.png')
```

### Phase 3: Debugging & Optimization
- [ ] Add comprehensive logging (replace print statements)
- [ ] Profile MCMC inference for bottlenecks
- [ ] Optimize array operations in `format_draws()`
- [ ] Add input validation with descriptive error messages
- [ ] Memory optimization for large datasets

### Phase 4: Testing with Synthetic Data
- [ ] Create synthetic data generator for panel data
- [ ] Unit tests for `data.py` functions
- [ ] Unit tests for `output.py` functions
- [ ] Integration tests with known treatment effects
- [ ] Convergence tests (R-hat, ESS validation)
- [ ] Edge case tests (missing data, single group, etc.)

### Phase 5: Future Enhancements
- [ ] GPU support via JAX/CUDA
- [ ] Staggered treatment timing support
- [ ] Spillover analysis scripts
- [ ] Sensitivity analysis (donor pool variations)
- [ ] Long-format input support (skip wide_to_long)

## 12. Related Resources

- **Reference Implementations**: `reference_directories/` - Original dobbs_fertility and nativity_analysis code
- **Data Source**: Monthly birth counts by state from vital statistics

## 13. Typical Workflow for Changes

1. **Read this guide**: Understand the data flow and standardized formats
2. **Check Section 10**: See active refactoring tasks and their status
3. **Understand the model**: Review `panel_nmf_model.py` mathematical structure
4. **Make changes**: Preserve plate structure and prior relationships
5. **Test locally**: Run with `--rank 5` for quick validation
6. **Check convergence**: Verify R-hat < 1.01 in MCMC output
7. **Verify output format**: Ensure standardized column names in output CSVs

## 14. Agent Configuration

Custom agents are defined in `.opencode/agent/` (excluded from git):

| Agent | File | Purpose |
|-------|------|---------|
| `master-planner` | `master-planner.md` | Project coordination, planning, documentation updates |
| `code-implementer` | `code-implementer.md` | Code implementation following project patterns |
| `documentation-writer` | `documentation-writer.md` | Documentation creation and updates |

### Creating New Agents

For specialized tasks, create new agent configuration files:

**Example: `debugger.md`**
```yaml
---
description: Debug and optimize Bayesian Panel NMF code
mode: primary
---
You are a debugging specialist for the Bayesian Panel NMF project...
```

**Example: `test-writer.md`**
```yaml
---
description: Write tests for Bayesian Panel NMF using pytest
mode: primary
---
You are a test engineer for the Bayesian Panel NMF project...
```
