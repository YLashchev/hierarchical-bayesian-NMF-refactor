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
uv run bpnmf run --config configs/fertility_smoke_test.yaml
```

When that finishes cleanly, run the full analysis:

```bash
uv run bpnmf run --config configs/fertility_config.yaml
```

Chain-level parallelism is chosen automatically from the visible JAX devices (`mcmc.auto_parallelism: true`, the default): a single-CPU host runs `mcmc.max_chains` chains sequentially, a single GPU runs them vectorized on that device, and a multi-device host (multiple CPUs exposed via `numpyro.set_host_device_count`, or multiple GPUs/TPUs) runs them in parallel across devices, capped at the visible device count. See `src/bayesian_panel_nmf/parallelism.py::choose_mcmc_parallelism` for the exact rules. Model types configured under `model.types` always run sequentially, one after another, in a single process.

Both write posterior draws + preprocessed data to `results/<type>/` (and figures under `results/<type>/figs/` when `output.figures: true`).

Regenerate figures from an existing draws CSV without re-running MCMC:

```bash
uv run bpnmf viz --results results/total/NB_births_total_3.csv

# effects tables only (no figures) — fast re-inspection of the numbers
uv run bpnmf viz --results results/total/NB_births_total_3.csv --tables-only
```

The draws-file path you pass is also what selects the results directory (tables
and figures are written to its parent); omit `--results` in a terminal for an
interactive picker over draws found under `./*results*/`. See
`documentation.md` → CLI reference for `--tables-only`, group selection, and
how the results directory is resolved.

An ArviZ-based convergence gate (rank-normalized R-hat, bulk/tail ESS,
divergences) always runs after MCMC and is written to `*_convergence.json`
next to the draws CSV.

To inspect saved trace sidecars (written with `--save-traces`) after a run:

```bash
uv run bpnmf traces results/total/NB_births_total_3_traces.nc          # R-hat/ESS pass-fail table
uv run bpnmf traces results/total/NB_births_total_3_traces.nc --plots  # visual trace plots
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
uv run bpnmf run --config configs/my_config.yaml
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
mcmc:
  auto_parallelism: true  # default: pick num_chains/chain_method from devices
  max_chains: 4             # upper bound on chain count
```

By default, `choose_mcmc_parallelism` picks the chain count and execution method from `jax.devices()`: sequential on a single CPU, vectorized on a single GPU, parallel (capped at device count) on multiple CPUs/GPUs/TPUs.

To pin exact values manually instead (e.g. to force `vectorized` on a CPU for testing, or guarantee a specific chain count regardless of detected hardware), set `auto_parallelism: false` and provide both `num_chains` and `chain_method` explicitly:

```yaml
mcmc:
  auto_parallelism: false
  num_chains: 4
  chain_method: "sequential"  # "sequential", "parallel", or "vectorized"
```

### Cut-mode inference (two-stage posterior)

By default the treatment effect and untreated baseline are fit jointly. Set
`model.inference_mode: "cut"` to fit them in two stages instead: Stage 1 fits
the untreated baseline on control cells only, then each of a chain-stratified
subset of Stage-1 draws conditions a complete Stage-2 treatment-effect fit.
Exposed outcomes never feed back into the baseline.

```yaml
model:
  inference_mode: "cut" # default: "joint"

cut: # all optional
  num_stage1_draws: 25 # baseline draws carried to Stage 2 (>= 50 for publication runs)
  stage2_draws_per_component: 100 # output thinning; null keeps all retained draws
  stage2_mcmc: # overlay on mcmc: for the cheaper conditional fits
    num_warmup: 500
```

Seeds: Stage-1 MCMC uses `mcmc.random_seed`, the Stage-1 PPC stream uses
`+1`, draw selection defaults to `+2` (`cut.selection_seed`), and Stage-2
uses `+3` (`cut.stage2_seed`). The streams must stay distinct so which draws
get selected is independent of the draws themselves; an explicit
`cut.stage2_mcmc.random_seed` is rejected for the same reason.

Cut runs write `{stem}_cut.csv` (combined draws with `cut_component` +
`stage1_*` provenance columns), `{stem}_cut_stage1_ppc.csv` (full Stage-1
posterior-predictive draws), `{stem}_cut_convergence.json` (per-stage
manifest), and with `--save-traces` also `{stem}_cut_stage1_traces.nc` +
`{stem}_cut_stage2_traces/component_*.nc`. See
`configs/fertility_cut_config.yaml` for a working example.

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
    Config,
    load_and_prepare,
    run_mcmc_inference,
    generate_predictions,
    format_draws,
    model,
)

config = Config.from_yaml("configs/my_config.yaml")
data_dict = load_and_prepare("data/my_panel.csv", config, groups=["total"])
mcmc = run_mcmc_inference(data_dict, model, rank=10, config=config)
predictions = generate_predictions(mcmc, data_dict, model, rank=10, config=config)
draws_df = format_draws(mcmc.get_samples(group_by_chain=True), predictions, data_dict)
```

## Development

```bash
uv run pytest              # 333 regression + integration tests
uv run ruff check .
uv run ruff format .
```

Current coverage is regression and integration: synthetic CSVs exercised through `load_and_prepare`, YAML config validation, and subprocess runs of `bpnmf run`. True unit tests of individual functions (especially inside `src/bayesian_panel_nmf/models/`) are still an open roadmap item.

See `CHANGELOG.md` for release history and the **Roadmap** section below for open items.

## Credits

Developed by Yan Lashchev with [Alex Franks](https://afranks.com/) (UCSB, Statistics & Applied Probability). Derived from the reference implementation in [afranks86/dobbs_fertility](https://github.com/afranks86/dobbs_fertility).

## License

TBD

## Roadmap

Built-in but still being hardened:

- [x] **Add MCMC diagnostics** — done: `bpnmf traces` (R-hat/ESS table + trace plots) and the per-run convergence gate (`*_convergence.json`)
- [ ] **Unit test coverage** — current suite is mostly integration / regression against synthetic CSVs; add targeted unit tests for functions in `models/`, `inference.py`, and `results.py`
- [ ] **GPU support** — JAX already runs on GPU; surface a config flag + verify chain parallelism against `numpyro.set_host_device_count`
- [ ] **Server/HPC multi-model-type parallelism** — model types currently always run sequentially in one process (chain-level parallelism via `mcmc.auto_parallelism` is automatic within each fit). If running many independent model types on a server/HPC host becomes a bottleneck, revisit process-level parallelism across model types, with the same CPU/RAM oversubscription guards the removed `analysis_workers` mechanism had.
- [ ] **Reference-style post-hoc diagnostics** — optionally save selected latent draws (`te`, treatment effects, `unit_weight`, `time_fac`, `disp`) so R-hat/ESS can be computed after a run without calling NumPyro `summary()` during production. Prefer Parquet or compact sidecar files over widening the main CSV; keep full-run diagnostics off by default.
- [ ] **Spillover analysis** — diagnostics for contamination between treated and neighboring control units
- [ ] **Donor-pool sensitivity** — systematic leave-one-out / leave-region-out robustness checks
- [ ] **Additional outcome distributions** — Gaussian, Student-t for continuous outcomes
- [ ] **Alternative latent structures** — GPLVM, linear factor model alongside the current NMF formulation
