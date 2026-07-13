# AGENTS.md — hierarchical-bayesian-NMF-refactor

## Project and methodology boundary

This repository implements a Bayesian hierarchical panel count model with a
low-rank nonnegative factorization. The local methodology supplement,
`jama-e2428527-s001.pdf`, describes the analysis as a low-rank factor-model
approach: the key structural assumption is that untreated outcomes are well
approximated by a small number of latent factors.

Do not describe this model as a difference-in-differences or parallel-trends
analysis. The supplement reports diverging pre-trends for this application and
uses factor-model structure instead. The paper's main text emphasizes
associations; do not strengthen that language to causal claims without an
approved methodological change.

Treat the paper, current implementation, configuration, and tests as evidence
sources. If they conflict or do not support a proposed scientific assertion,
ask before changing model structure or documenting the assertion as fact.

---

## Session and environment

1. **Engram health check.** Verify the Engram daemon:

   ```bash
   curl -s http://localhost:7437/health -o /tmp/h.json && cat /tmp/h.json
   ```

   If not healthy: `nohup engram serve 7437 &>/tmp/engram.log & disown` — wait ~2 s, then re-check.
   Do not proceed until healthy — `mem_*` tools silently no-op otherwise.

2. **Load prior context.** Call `mem_context("<keyword>")` for the subsystem you're about to touch.

3. **Load relevant skills.** Review available Superpowers skills and load every applicable one before any tool calls or code changes.

4. **Install / sync environment first.** `uv run pytest` fails with
   `ModuleNotFoundError: No module named 'bayesian_panel_nmf'` unless the
   environment is synced first:

   ```bash
   uv sync --all-extras --dev
   ```

---

## Verified commands

```bash
# Run a full analysis
uv run scripts/run_analysis.py --config configs/fertility_config.yaml

# Quick smoke test (fast, ~5 min)
uv run scripts/run_analysis.py --config configs/fertility_smoke_test.yaml

# Single type + rank override
uv run scripts/run_analysis.py --config configs/nativity_config.yaml --type total --rank 5

# Debug logging
uv run scripts/run_analysis.py --config configs/<cfg>.yaml --verbose

# Re-render figures from an existing draws CSV (no re-MCMC)
uv run scripts/generate_full_viz.py --results results/total/NB_births_total_5.csv

# Trace plots
uv run scripts/analyze_traces.py --results-dir results/
uv run python scripts/make_trace_plots.py results/total/NB_births_total_3_traces.nc

# Tests
uv run pytest                                        # full suite
uv run pytest -x --ff                                # fail-fast, failed-first
uv run pytest tests/test_total_aggregation.py        # single file

# Lint / type check
uv run ruff check . --fix
uv run ruff format .
uv run mypy src/bayesian_panel_nmf/
```

Run `uv run pytest` **and** `uv run ruff check .` before marking any task complete.

---

## Repository map

```
src/bayesian_panel_nmf/
  data.py                  # CSV ingestion, panel reshaping → model arrays
  models/
    panel_nmf_model.py     # NumPyro model definition — NMF factors, FE, treatment
    utils.py               # missingness_adjustment(), log-space helpers
  inference.py              # NUTS config, MCMC run, convergence gate (ArviZ)
  output.py                  # Draws serialisation → NetCDF / ArviZ InferenceData
  visualization.py           # PPC plots, trace plots — matplotlib optional
  validation.py               # Config + data validation, DataError / ConfigError
  aggregate_units.py           # Post-hoc unit aggregation of posterior draws
  reporting.py                  # Terminal summary tables (Rich)
  logging_config.py              # Loguru setup — non-destructive, re-entrant

scripts/
  run_analysis.py           # Main entrypoint — load config, run pipeline
  analyze_traces.py           # Trace plot generation from saved draws
  generate_full_viz.py          # Full PPC visualization suite
  make_trace_plots.py            # Batch trace plot helper

configs/                     # Flat — no `configs/priors/` subdirectory
tests/                       # Flat — no subdirectories under `tests/`
notebooks/                   # Exploratory only — not source of truth
```

---

## Data and model invariants

- `data.load_and_prepare()` standardizes input through pandas and returns model
  arrays with shape `(K, D, N)` for groups, units, and time periods.
- A `missing_idx_array` cell represents a present but suppressed small count;
  `missingness_adjustment()` models its 1–9 censoring interval. A structurally
  absent cell is governed separately by `allow_unbalanced_panel`.
- JAX arrays are immutable: use `.at[...].set()` or `.at[...].add()`. Split a
  PRNG key before each independent random consumer.
- In NumPyro, `numpyro.sample(..., obs=y)` contributes likelihood and its return
  value is intentionally unused. Treat plate structure, priors, factor
  construction, likelihood family, and treatment model as approval-required
  scientific changes.
- Keep reporting-only aggregate units out of model arrays:
  `add_aggregate_units()` runs on posterior draws after fitting.
- In the current model, NB dispersion is constructed only on the NB path; the
  Poisson path uses no dispersion (`panel_nmf_model.py` guards `dispersion`
  behind `if outcome_dist == "NB"`).

---

## Configuration and output safeguards

- YAML booleans must be unquoted. `"false"` (string) is truthy in Python.
  `validation._require_bool()` rejects quoted-string booleans at load time for:
  `allow_unbalanced_panel`, `aggregation.enabled`, `model.sample_disp`,
  `model.adjust_for_missingness`, `model.model_treated`,
  `model.types.<name>.total_all`, `mcmc.progress_bar`, `output.figures`,
  `output.clean`.
- Chain-level parallelism (`mcmc.auto_parallelism`, default true) is
  resolved by `choose_mcmc_parallelism()` in
  `src/bayesian_panel_nmf/mcmc_utils.py` from the visible JAX devices --
  sequential on 1 CPU device, vectorized on 1 GPU device, parallel
  (capped at device count) on multiple devices. Model types always run
  sequentially, one process, one after another -- there is no
  analysis-level process pool (removed; was `parallel.analysis_workers`).
- Every MCMC run always writes `<draws-stem>_convergence.json` next to the
  draws CSV (rank-normalized R-hat, bulk/tail ESS, divergences via
  `inference.convergence_summary()`; thresholds: R-hat < 1.01, ESS > 400, 0
  divergences). A failed gate logs a warning; it is not gated behind a flag.
- `visualization.py` auto-maps legacy column names (`state→unit`,
  `category→group`, `population→denominator`, `exposure_code→treatment`) via
  `_standardize_columns()`. Do not remove this — reference upstream CSVs rely
  on it. Any new plotting function must lazily import matplotlib
  (`try/except ImportError`), never at module top level.
- `output.clean: true` removes `<output_dir>/<type>/` before writing. Use
  `_safe_rmtree()` in `scripts/run_analysis.py` for any guarded deletion — never
  call `shutil.rmtree()` directly.

---

## Approval boundaries and verification

### Ask first

- Changes to `models/panel_nmf_model.py` (plate structure, priors, likelihood
  family, factor construction, treatment model).
- Introducing a new data source or changing outcome/denominator definitions.
- Changing the exposure/treatment timing model — exposure is staggered in the
  paper's setup; there is no single universal pre/post cutoff to preserve.
- Breaking changes to the standard column names / `data_dict` shape.
- Deleting or renaming files in `src/bayesian_panel_nmf/`.
- Removing legacy-column support from `visualization.py`.
- Adding new package dependencies in `pyproject.toml`.
- Changing MCMC sampler defaults (`target_accept`, `max_tree_depth`, warmup length).
- Strengthening the paper's association framing into a stronger causal claim.

### Never

- Describe the model as difference-in-differences or as relying on parallel
  trends — the supplement reports evidence against that assumption here.
- Commit `.pi-subagents/` or local scratch notes.
- Commit contents of `results/`, `figs/`, `.env`.
- Force-push to main.
- Break mathematical consistency of the low-rank factorization.
- Run analysis without the `uv run` prefix — breaks environment isolation.
- Import matplotlib (or any viz library) at module top level outside `visualization.py`.
- Use `shutil.rmtree()` directly — use `_safe_rmtree()`.

### Verification

- Run `uv run pytest` and `uv run ruff check .` before declaring any task complete.
- Verify MCMC convergence: R-hat < 1.01 and ESS > 400 (thresholds in `inference.py`).
- New features require tests. Add fixtures locally within the relevant test
  file; `tests/conftest.py` currently defines none. Tests are flat — add new
  test files at `tests/test_<module>.py`.
- All tests must be deterministic. Set `jax.random.PRNGKey(0)` explicitly; never
  use `random.random()`.

---

## Maintenance

Every instruction in this file must be traceable to `jama-e2428527-s001.pdf`,
current repository code/configuration/tests, or an explicit project workflow
requirement. When a cited source changes, update or remove the corresponding
instruction rather than leaving it stale.
