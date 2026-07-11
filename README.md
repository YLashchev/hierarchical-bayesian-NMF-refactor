# Bayesian Panel NMF

Estimate causal treatment effects on panel data via a Bayesian hierarchical model with low-rank factorization (JAX + NumPyro).

Works with any panel dataset that can be described by a YAML schema: one unit column, one time column, a binary treatment indicator, and one or more outcome columns (with optional denominators for rates).

## Installation

```bash
git clone <repo-url> bayesian_panel_nmf
cd bayesian_panel_nmf
uv sync --all-extras --dev
```

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). All other dependencies are pinned in `pyproject.toml`.

## Quick Start

The repo ships with `data/raw/fertility_data.csv` (state-level US birth counts 2016–2024) and two configs:

- `configs/fertility_smoke_test.yaml` — 2 chains × 200+200 samples, rank 3, one model type. Runs in ~4 min on a laptop. For verifying the pipeline works, not for inference.
- `configs/fertility_config.yaml` — 4 chains × 2000+2000 samples across 4 model types. Full production settings. Takes hours and writes large CSVs (hundreds of MB to ~1 GB per multi-group type).

Smoke test first:

```bash
uv run scripts/run_analysis.py --config configs/fertility_smoke_test.yaml
```

When that finishes cleanly, run the full analysis:

```bash
uv run scripts/run_analysis.py --config configs/fertility_config.yaml
```

Full fertility runs default to `parallel.analysis_workers: 1` so progress remains visible and only one large draws/reporting job sits in memory at a time. You can set it to `-1` to auto-cap model-type workers against `mcmc.num_chains` and CPU count. Progress bars are disabled for multi-model parallel runs so subprocess output does not interleave; the parent process logs a line each time a model type finishes.

Both write posterior draws + preprocessed data to `results/<type>/` (and figures under `results/<type>/figs/` when `output.figures: true`).

Regenerate figures from an existing draws CSV without re-running MCMC:

```bash
uv run scripts/generate_full_viz.py --results results/total/NB_births_total_3.csv
```

An ArviZ-based convergence gate (rank-normalized R-hat, bulk/tail ESS,
divergences) always runs after MCMC and is written to `*_convergence.json`
next to the draws CSV.

To compute full post-hoc diagnostics from saved trace sidecars (all latent
parameters), or limited diagnostics from saved draws (mu / mu_treated / ypred only):

```bash
uv run python scripts/compute_posthoc_diagnostics.py results/total/NB_births_total_5.csv --param-filter mu
```

## Using Your Own Data

### 1. Format the CSV

Long-panel layout (one row per unit × time) with columns for:

- **unit** — panel identifier (state, firm, county)
- **time** — date/timestamp (any format pandas can parse)
- **treatment** — `0`/`1` indicator
- **outcome(s)** — one or more numeric columns
- **denominator(s)** — optional, for rate models

### 2. Write a config

Minimal example (`configs/my_config.yaml`):

```yaml
data:
  input_file: "data/my_panel.csv"
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
  outcome_distribution: "NB" # "NB" or "Poisson"
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
```

### 3. Run

```bash
uv run scripts/run_analysis.py --config configs/my_config.yaml
```

See `configs/base_config.yaml` for every supported option with inline comments. `configs/nativity_config.yaml` and `configs/fertility_config.yaml` are working end-to-end examples.

## Configuration Highlights

### Schema: explicit or prefix-based

Define outcomes one of two ways (pick one, not both):

```yaml
# Option A: explicit list
outcomes:
  - outcome_col: "y1"
    denominator_col: "pop1"
    label: "group1"

# Option B: prefix shortcut for many similarly-named groups
outcomes_from_prefixes:
  outcome_prefix: "births_"
  denominator_prefix: "pop_"
  include: ["nhwhite", "hisp", "nhblack"]
```

### Temporal aggregation

```yaml
data:
  aggregation:
    enabled: true
    period: "bimonthly" # monthly | bimonthly | quarterly | yearly
```

### Synthetic `"total"` group

If your config requests `groups: ["total"]` but no outcome is labelled `"total"`, declare how to build it per model type:

```yaml
model:
  types:
    total:
      groups: ["total"]
      total_from: ["nhwhite", "hisp", "nhblack"] # sum these labels
      # OR
      total_all: true # sum every defined outcome
```

### Parallelism

```yaml
parallel:
  analysis_workers: 1 # 1 = sequential, -1 = auto-cap

mcmc:
  num_chains: 4 # chains within one fit
```

`analysis_workers × num_chains` is capped against CPU count automatically. With `analysis_workers: -1`, the runner uses the largest safe model-type worker count. For easier progress visibility or lower memory/disk pressure, set `analysis_workers: 1`. Parallel runs disable worker progress bars; each model type logs a line when it finishes.

### Figures, diagnostics + cleanup

```yaml
output:
  figures: true # auto-generate PPC + summary plots at the end of a run
  clean: false # wipe <output_dir>/<type>/ before writing
```

Output filenames follow a fixed `{distribution}_{outcome}_{type}_{rank}` scheme (e.g. `NB_births_total_5.csv`), plus a `_convergence.json` sidecar with R-hat/ESS/divergence diagnostics written after every fit.

### Reporting-only aggregate units and PPC selection

Aggregate units are created after model fitting, only for reporting/PPC. They are not added to model input.

```yaml
output:
  aggregate_units:
    - unit: "Treated units excluding Texas" # dataset-specific name example
      include_treated_units: true
      exclude_units: ["Texas"]

  ppc_units:
    - "Texas"
    - "Treated units excluding Texas"

  ppc_acf_lags: [6, 3, 1]
  ppc_unit_corr_max_time: "2022-04-01" # optional time < cutoff for unit-correlation PPC
  ppc_exclude_units: [] # default: do not silently exclude aggregate units
```

Selectors are generic: use exactly one of `include_treated_units: true`, `include_all_units: true`, or `include_units: [...]`; add optional `exclude_units`, `strict`, or `overwrite` per aggregate spec.

## Outputs

Per model type, under `<output_dir>/<type>/`:

| File                               | Contents                                                                               |
| ---------------------------------- | -------------------------------------------------------------------------------------- |
| `{distribution}_{type}_{rank}.csv` | Tidy posterior draws                                                                   |
| `df_{type}.csv`                    | Preprocessed observed data (standardized columns)                                      |
| `*_convergence.json`               | Always-on ArviZ convergence gate: R-hat, bulk/tail ESS, divergences                     |
| `figs/` (subdir)                   | Fit/gap plots, PPC panels, interval plot, summary tables (when `output.figures: true`) |

Posterior draws schema:

| Column                                | Meaning                                                            |
| ------------------------------------- | ------------------------------------------------------------------ |
| `.draw`, `.chain`, `.iteration`       | Per-draw indices (compatible with arviz / tidy posterior tooling)  |
| `unit`, `time`, `group`               | Panel coordinates                                                  |
| `outcome`, `denominator`, `treatment` | Observed data                                                      |
| `ypred`                               | Posterior predictive draw (counterfactual, untreated)              |
| `mu`                                  | Log-rate under control                                             |
| `mu_treated`                          | Log-rate under treatment (equals `mu` when `model_treated: false`) |

## Python API

```python
from bayesian_panel_nmf import (
    load_and_prepare,
    run_mcmc_inference,
    generate_predictions,
    format_draws,
    model,
)

data_dict = load_and_prepare("data/my_panel.csv", config, groups=["total"])
mcmc = run_mcmc_inference(data_dict, model, rank=10, config=config)
predictions = generate_predictions(mcmc, data_dict, model, rank=10, config=config)
draws_df = format_draws(mcmc.get_samples(group_by_chain=True), predictions, data_dict)
```

## Development

```bash
uv run pytest              # 155 regression + integration tests, ~15s
uv run ruff check .
uv run ruff format .
```

Current coverage is regression and integration: synthetic CSVs exercised through `load_and_prepare`, YAML config validation, and subprocess runs of `scripts/run_analysis.py`. True unit tests of individual functions (especially inside `src/bayesian_panel_nmf/models/`) are still an open roadmap item.

See `CHANGELOG.md` for release history and the **Roadmap** section below for open items.

## Credits

Developed by Yan Lashchev with [Alex Franks](https://afranks.com/) (UCSB, Statistics & Applied Probability). Derived from the reference implementation in [afranks86/dobbs_fertility](https://github.com/afranks86/dobbs_fertility).

## License

TBD

## Roadmap

Built-in but still being hardened:
- [ ] **Add MCMC diagnostics** - trace plots (log postperior), ESS, Rhats
- [ ] **Unit test coverage** — current suite is mostly integration / regression against synthetic CSVs; add targeted unit tests for functions in `models/`, `inference.py`, and `output.py`
- [ ] **GPU support** — JAX already runs on GPU; surface a config flag + verify chain parallelism against `numpyro.set_host_device_count`
- [ ] **Server/HPC parallel profiles** — add config profiles for laptop vs server runs, and optional reference-style task granularity (`model_type × rank`) so server runs can parallelize many independent fits safely. Guard cleanup/output races, keep diagnostics off for full parallel runs, and cap `analysis_workers × mcmc.num_chains` by CPU/RAM.
- [ ] **Reference-style post-hoc diagnostics** — optionally save selected latent draws (`te`, treatment effects, `unit_weight`, `time_fac`, `disp`) so R-hat/ESS can be computed after a run without calling NumPyro `summary()` during production. Prefer Parquet or compact sidecar files over widening the main CSV; keep full-run diagnostics off by default.
- [ ] **Spillover analysis** — diagnostics for contamination between treated and neighboring control units
- [ ] **Donor-pool sensitivity** — systematic leave-one-out / leave-region-out robustness checks
- [ ] **Additional outcome distributions** — Gaussian, Student-t for continuous outcomes
- [ ] **Alternative latent structures** — GPLVM, linear factor model alongside the current NMF formulation
