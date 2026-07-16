# bayesian_panel_nmf — Documentation

Central reference for the Bayesian hierarchical panel model with low-rank
factorization. This document is built up alongside the codebase; sections are
added as features land.

> **Methodology note.** This is a low-rank factor-model approach: untreated
> outcomes are approximated by a small number of latent factors. It is **not**
> a difference-in-differences / parallel-trends design — the analysis relies on
> factor structure, not on matched pre-trends.

---

## Installation

Requires **Python 3.12–3.14** and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> bayesian_panel_nmf
cd bayesian_panel_nmf
uv sync --all-extras --dev
```

This installs the full stack, pinned in `pyproject.toml` and locked in
`uv.lock`:

| Package | Floor | Notes |
| --------- | ------- | ------- |
| JAX / jaxlib | `>=0.10.2,<0.11` | jaxlib resolves transitively via jax — not pinned directly |
| NumPyro | `>=0.21,<0.22` | |
| ArviZ | `>=1.2,<1.3` | 1.x uses `xarray.DataTree` (not the legacy `InferenceData`) |
| NumPy | `>=2.0` | required by JAX 0.10 |
| pandas | `>=2.2.3` | |
| matplotlib / seaborn | `>=3.8` / `>=0.13.2` | hard dependencies — figures are a core feature |

### GPU / accelerator

JAX selects the accelerator wheel via its own extras (e.g. `jax[cuda13]`).
The default install is CPU-only; see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for CUDA/ROCm/TPU wheels. Chain parallelism is chosen automatically from the
visible devices (see `mcmc.auto_parallelism`).

### Verify the install

```bash
uv run python -c "import jax, numpyro, arviz; print(jax.__version__, numpyro.__version__, arviz.__version__)"
# expected: 0.10.x 0.21.x 1.2.x
```

---

## Configuration reference

Configs are YAML files validated by a typed schema (`bayesian_panel_nmf.config.Config`,
pydantic v2). Unknown keys are **rejected** (typos fail fast), and booleans must be
unquoted (`true`/`false`, not `"true"`). Every default below lives in one place —
the schema — so a value you omit resolves identically everywhere.

### `data` (required)

| Key | Type | Default | Notes |
| ----- | ------ | --------- | ------- |
| `input_file` | str | — (required) | CSV path |
| `output_dir` | str | — (required) | results root |
| `schema` | object | — (required) | column roles (below) |
| `date_format` | str | `"auto"` | `"auto"` = detect; else a strptime format |
| `start_date` / `end_date` | str/null | null | filter window |
| `aggregation.enabled` | bool | `false` | temporal aggregation |
| `aggregation.period` | str | `"bimonthly"` | monthly/bimonthly/quarterly/yearly |
| `allow_unbalanced_panel` | bool | `false` | true = absent cells treated as suppressed |
| `outcome` | str/null | null | filename label override (else derived) |

`data.schema`: `unit_col`, `time_col`, `treatment_col` (all required) plus **exactly one**
of `outcomes` (explicit list of `{outcome_col, label, denominator_col?}`) or
`outcomes_from_prefixes` (`{outcome_prefix, denominator_prefix?, include?}`).

### `model`

| Key | Type | Default | Notes |
| ----- | ------ | --------- | ------- |
| `outcome_distribution` | str | `"NB"` | `"NB"` or `"Poisson"` |
| `nb_disp` | float | `1e-4` | fixed NB dispersion (NB path only) |
| `sample_disp` | bool | `false` | sample dispersion per unit instead |
| `adjust_for_missingness` | bool | `true` | model 1–9 censored small counts |
| `model_treated` | bool | `true` | include the treatment effect |
| `inference_mode` | str/null | null→joint | `"joint"` or `"cut"` (see Cut mode) |
| `types` | map | `{}` | `name → {groups[], ranks_to_test[], total_from?[], total_all?, exclude_units?[]}` |

### `mcmc`

| Key | Default | Notes |
| ----- | --------- | ------- |
| `auto_parallelism` | `true` | pick chains/method from devices |
| `max_chains` | `4` | cap under auto_parallelism |
| `num_chains` / `chain_method` | null | only used when auto_parallelism=false |
| `num_warmup` / `num_samples` | `1000` / `2500` | |
| `thinning` | `10` | |
| `random_seed` | `8675309` | |
| `progress_bar` | `true` | |

### `output`

`figures` (false — see [Visualization](#visualization) for the full
`bool | list[str] | "all"/"none"` selection semantics), `clean` (false),
`save_traces` (false), `print_tables` (true), `print_target_table` (true);
optional reporting filters `target_unit`, `report_groups`,
`aggregate_units`, `ppc_units`, `ppc_exclude_units`, `ppc_acf_lags` (unset →
`[6]` at report time), `ppc_unit_corr_max_time`. `draws_format` (`"csv"` default, or `"parquet"`) controls
only the large draws artifact (joint draws / cut combined draws); human-facing
tables (`df_{type}.csv`, `summary_table*.csv`, `expected_vs_observed.csv`,
`post_treatment_summary.csv`, `ppc_pvalues.csv`, `stage1_ppc.csv`) always stay CSV.

### `cut` (only read when `model.inference_mode: "cut"`)

`num_stage1_draws` (25), `stage2_draws_per_component` (100), `selection_seed`
(default `mcmc.random_seed + 2`), `stage2_seed` (default `+ 3`), `stage2_mcmc`
(overlay on `mcmc` for the cheaper conditional fits). The distinct seed offsets keep
draw *selection* independent of the draws themselves; setting `stage2_mcmc.random_seed`
is rejected (`cut.stage2_seed` is the authority).

---

## Visualization

`bayesian_panel_nmf.plots.PLOT_REGISTRY` maps a stable name to each
figure-producing `make_*` function. `output.figures` selects which registry
entries `reports.generate_reports()` renders; `summary_table` (and the
other CSV tables) is not in the registry and always renders regardless of
this selection.

Figure/table orchestration is split across three modules: `plots.py` (matplotlib
plotting primitives), `tables.py` (pure pandas/numpy table computation plus rich
terminal rendering — no matplotlib), and `reports.py` (`generate_reports()`,
the orchestration entry point that calls into both).

After each rank's artifacts are written, `scripts/run_analysis.py` calls
`tables.print_run_summary_panel()`: a rich terminal panel echoing that run's
config (model type, rank, chains/method, outcome distribution), the
convergence gate verdict (green PASS / red FAIL, plus R-hat/ESS/divergences
when present), the selected figures, and the written artifact paths. Purely
additive terminal output — no file or data side effects, so it never touches
a golden-checked artifact.

| Registry name | Function | Artifact |
| --- | --- | --- |
| `unit_fit` | `make_unit_fit_plot` | `fit_<target>.png` |
| `unit_gap` | `make_unit_gap_plot` | `gap_<target>.png` |
| `raw_rate` | `make_raw_rate_plot` | `raw_rate.png` |
| `interval` | `make_interval_plot` | `interval.png` |
| `group_comparison` | `make_group_comparison_plot` | `group_comparison.png` |
| `ppc` | `make_all_ppc_plots` | `ppc/*.png`, `ppc/ppc_pvalues.csv` |

`output.figures` accepts:

- `true` (default-equivalent "render everything") or `false` (render no
  figures) — the original boolean spelling still works.
- `"all"` / `"none"` — explicit string spellings of the same two extremes.
- a list of registry names, e.g. `figures: ["interval", "ppc"]` — renders
  only those. Unknown names are rejected at config-load time.

All spellings normalize to a canonical `list[str]` (the names to render).
`scripts/run_analysis.py` skips reporting entirely (no figures *and* no
tables) when the normalized selection is empty, matching the pre-existing
`figures: false` behavior. Calling `reports.generate_reports(..., figures=
["interval"])` directly always still writes the always-on tables — only the
PLOT_REGISTRY figures are gated at that level.

---

## Output artifacts

Each model type writes to `<output_dir>/<type>/`. Draws filenames follow
`{distribution}_{outcome}_{type}_{rank}` (e.g. `NB_births_total_3`); cut mode
appends `_cut`.

### Joint mode

| File | Contents |
| ---- | -------- |
| `{stem}.csv` or `.parquet` | Tidy posterior draws (one row per draw × group × unit × time): `.draw/.chain/.iteration`, `unit/time/group`, `outcome/denominator/treatment`, `ypred` (counterfactual untreated), `mu` (log-rate control), `mu_treated`. Format set by `output.draws_format`. This is the large artifact (100 MB–1 GB for multi-group types). |
| `{stem}_convergence.json` | Always-on gate: `{rhat_max, ess_bulk_min, ess_tail_min, divergences, converged}` (R-hat<1.01, bulk ESS>400, 0 divergences). |
| `df_{type}.csv` | Preprocessed observed data (standardized columns). |
| `{stem}_traces.nc` | Full posterior NetCDF sidecar (only with `--save-traces`). |
| `figs/` | PPC panels, fit/gap/interval/raw-rate plots, summary tables (when `output.figures: true`). |

### Cut mode (two-stage) — additional files

| File | Contents |
| ---- | -------- |
| `{stem}_cut.csv`/`.parquet` | Combined Stage-2 draws with provenance columns `cut_component, stage1_draw, stage1_chain, stage1_iteration`. `.draw` is globally unique across components; `.chain`/`.iteration` are the real Stage-2 chain/subsample index. |
| `{stem}_cut_stage1_ppc.csv` | Full Stage-1 posterior-predictive draws (feeds the PPC suite only). Always CSV. |
| `{stem}_cut_convergence.json` | Per-stage manifest: Stage-1 gate + every conditional Stage-2 fit's gate; top-level `converged` true only if Stage 1 and all components passed. |
| `{stem}_cut_stage1_traces.nc`, `{stem}_cut_stage2_traces/component_*.nc` | Trace sidecars (only with `--save-traces`). |

Regenerate figures from a saved draws file without re-running MCMC:

```bash
uv run scripts/generate_full_viz.py --results results/total/NB_births_total_3.csv
```

(`--results` accepts either the `.csv` or `.parquet` draws file.)

---

## Interpreting diagnostics

Every MCMC run writes a `*_convergence.json` gate next to the draws
(`bayesian_panel_nmf.diagnostics.convergence_summary`). It reports four numbers
and a verdict:

| Field | Meaning | Pass threshold |
| ----- | ------- | -------------- |
| `rhat_max` | worst rank-normalized R-hat across parameters | `< 1.01` |
| `ess_bulk_min` | smallest bulk effective sample size | `> 400` |
| `ess_tail_min` | smallest tail effective sample size | (reported; not gated) |
| `divergences` | total divergent transitions | `== 0` |
| `converged` | `true` only if R-hat, bulk ESS, and divergences all pass | — |

A failed gate logs a warning and the run continues — it never silently drops
output. On failure: increase `mcmc.num_warmup`/`num_samples`, or investigate
with the trace sidecars (run with `--save-traces`, then
`scripts/analyze_traces.py` for the numeric table or `scripts/make_trace_plots.py`
for visual traces). In cut mode the manifest reports Stage-1 and every Stage-2
fit separately; diagnostics are never pooled across conditional targets.

---

## Design decisions

Rationale for choices that are surprising without context.

### Why cut mode uses separate RNG seeds per step

The base `mcmc.random_seed` drives the Stage-1 MCMC; `+1` is the Stage-1
posterior-predictive stream; `+2` (`cut.selection_seed`) draws the
chain-stratified subset of Stage-1 draws carried into Stage 2; `+3`
(`cut.stage2_seed`) drives the Stage-2 fits. The streams must be **distinct**
so that *which* Stage-1 draws get selected is statistically independent of the
draws themselves and of the Stage-2 sampling — reusing a stream would correlate
selection with the values being selected and bias the nested Monte Carlo
approximation. An explicit `cut.stage2_mcmc.random_seed` is therefore rejected:
`cut.stage2_seed` is the single Stage-2 seed authority.

### Why Stage-1 and Stage-2 model code is duplicated, not shared

The cut posterior is `p(φ|Z)·p(θ|Y,φ)` — exposed outcomes must never feed back
into the untreated baseline. `models/joint.py`, `models/cut_baseline.py`, and
`models/cut_treatment.py` deliberately duplicate the factor/treatment blocks
rather than import shared helpers, so an edit to one cannot silently change the
other's isolation guarantee. Parity is enforced by `tests/test_cut_model_parity.py`
instead of by code sharing.

### Why single-CPU hosts run chains sequentially

`choose_mcmc_parallelism` (in `parallelism.py`) picks `chain_method` from the
visible JAX devices: sequential on one CPU, vectorized on one GPU, parallel
across multiple devices. XLA's CPU backend shares one thread pool across
logical devices, so forcing multiple CPU devices for `parallel` gives no clean
speedup — sequential is the honest default rather than false parallelism.
