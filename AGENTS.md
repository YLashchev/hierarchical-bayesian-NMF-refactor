# Bayesian Panel NMF - Developer Guide

## 1. Project Overview

This repository implements **Bayesian hierarchical panel models with low-rank factorization** for causal inference using JAX and NumPyro. The primary application is estimating causal effects of abortion policy changes (post-Dobbs decision) on birth rates, stratified by demographic groups (nativity status, age, race, etc.).

### Research Context
- **Goal:** Estimate counterfactual birth rates under abortion bans using difference-in-differences with staggered treatment timing
- **Method:** Low-rank matrix factorization (NMF) captures state-time interactions; hierarchical treatment effects allow heterogeneity across states and demographic categories
- **Data:** Monthly/bimonthly birth counts by state, time, and demographic group (2016-2024)

### Core Statistical Model
```
log(births) = state_fe + time_fe + low_rank_factors + treatment_effect + error
```
Where treatment effects are hierarchical: `te = state_te + category_te + state×category_te + time_te`

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

from bayesian_panel_nmf.data import load_panel_data, DataSchema
from bayesian_panel_nmf.models import model
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
Use **NumPy/SciPy style**. See `src/bayesian_panel_nmf/inference/sampler.py` for examples:
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
| Array dimensions | Capital letters | `K` (groups), `D` (states/units), `N` (time periods) |

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
    with numpyro.plate('D', D):       # States
        state_fe = numpyro.sample('state_fe', dist.ImproperUniform(...))
```

### Broadcasting
Use explicit `None` indexing for broadcasting:
```python
fixed_effects = state_fe[:, :, None] + time_fe[:, None, :]  # (K, D, N)
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
│   ├── nativity_config.yaml    # Full nativity analysis
│   └── test_config.yaml        # Quick testing (rank 5)
├── data/raw/                   # Input CSV files
├── results/                    # Output: CSVs with posterior draws
├── figs/                       # Output: Generated figures
├── scripts/
│   └── run_analysis.py         # Main entry point
├── src/bayesian_panel_nmf/
│   ├── data/
│   │   ├── loader.py           # CSV loading, validation
│   │   ├── preprocessing.py    # Wide→Long, aggregation
│   │   └── schema.py           # DataSchema class
│   ├── models/
│   │   ├── panel_nmf_model.py  # Core Bayesian model
│   │   ├── priors.py           # Prior configurations
│   │   └── utils.py            # Missingness adjustment
│   └── inference/
│       ├── sampler.py          # MCMC execution
│       └── postprocessing.py   # Predictions, merging draws
├── nativity_analysis.qmd       # R/Quarto visualization
└── plot_utilities.R            # R plotting functions
```

## 6. Data Flow

```
Raw CSV (wide) → load_panel_data() → wide_to_long() → preprocess_pipeline()
                                                              ↓
                                              data_dict {Y, denominators, masks}
                                                              ↓
                                                    model() [NumPyro]
                                                              ↓
                                                    run_mcmc_inference()
                                                              ↓
                                              MCMC object {posterior samples}
                                                              ↓
                                                    generate_predictions()
                                                              ↓
                                                    merge_draws_and_data()
                                                              ↓
                                              Results CSV (draws + observed)
```

## 7. Key Data Structures

### data_dict (model input)
```python
data_dict = {
    'Y': np.ndarray,                # (K, D, N) - birth counts
    'denominators': np.ndarray,     # (K, D, N) - population
    'control_idx_array': np.ndarray,  # (K, D, N) - bool, True=control period
    'missing_idx_array': np.ndarray,  # (K, D, N) - bool, True=observed
    'groups': List[str],            # Group names
    'units': List[str],             # State/unit names
    'times': List[str],             # Time period labels
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
  input_file: "data/raw/nativity_analyticdata.csv"
  output_dir: "results"
  start_date: "2016-01-01"
  end_date: "2024-01-01"
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
    groups:
      groups: ["hisp_usborn", "hisp_foreign", "nh_usborn", "nh_foreign"]
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
Birth counts <10 are suppressed for privacy. The model explicitly adjusts for this censoring:
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
     state_treatment_effect +          # State-specific
     category_treatment_effect +       # Category-specific (e.g., nativity)
     state_category_te                 # Interaction
```

### Model Refactoring Rules
When modifying `panel_nmf_model.py`:
1. **Preserve mathematical consistency** - the low-rank factorization structure must remain valid
2. **Keep plate structure intact** - independence assumptions are encoded in plates
3. **Test convergence** - any prior changes should be validated with R-hat < 1.01, ESS > 400

## 10. Roadmap / Future Work

- [ ] **GPU Support**: JAX/CUDA acceleration for faster MCMC
- [ ] **Test Suite**: pytest with unit and integration tests
- [ ] **Spillover Analysis**: Scripts for cross-state spillover effects
- [ ] **Sensitivity Analysis**: Donor pool sensitivity (exclude neighboring states)
- [ ] **Python Visualization**: Port R plotting to matplotlib/seaborn
- [ ] **Type Hints**: Complete coverage across all modules
- [ ] **Logging**: Replace print statements with proper logging

## 11. Related Resources

- **Memory Bank**: `../memory-bank/` - Project context, decisions, progress tracking
- **Original dobbs_fertility**: `../dobbs_fertility/` - Parent project with age/race/education analyses
- **Data Source**: Monthly birth counts by state from vital statistics

## 12. Typical Workflow for Changes

1. **Read context**: Check `../memory-bank/activeContext.md` for current status
2. **Understand the model**: Review `panel_nmf_model.py` mathematical structure
3. **Make changes**: Preserve plate structure and prior relationships
4. **Test locally**: Run with `--rank 5` for quick validation
5. **Check convergence**: Verify R-hat < 1.01 in MCMC output
6. **Update memory bank**: Document significant changes in `../memory-bank/progress.md`
