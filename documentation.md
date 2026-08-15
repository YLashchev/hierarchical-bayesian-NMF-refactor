# bayesian_panel_nmf — Documentation

Reference for the Bayesian hierarchical panel model with low-rank
factorization. Grows alongside the codebase.

> **Methodology note.** This is a low-rank factor-model approach: untreated
> outcomes are approximated by a small number of latent factors. It is **not**
> a difference-in-differences / parallel-trends design — it relies on
> factor structure, not matched pre-trends.

---

## The model

### What it estimates

The data is a panel of counts indexed by group (`K`), unit (`D`), and time
period (`N`) — shape `(K, D, N)` everywhere in the model. For each cell the model builds an untreated log-rate surface `mu_ctrl`
from a low-rank factorization plus fixed effects, then (optionally) adds a
treatment effect `te` for cells past exposure. The observed count is drawn
from that combined rate `mu = mu_ctrl + te`.

`mu_ctrl` decomposes as:

```
mu_ctrl[k, d, n] = time_factor[k, d, n] + state_fe[k, d] + time_fe[k, n] + log(denominator[k, d, n])
time_factor[k, d, n] = log( sum_f exp(time_fac[k, f, n] + unit_weight[d, f]) )
```

`time_fac` is a set of `rank` latent time-factor curves per group; `unit_weight`
gives each unit a convex-combination-like mixture over those `rank` factors
(rows sum to 1 in linear space, via the Dirichlet prior below). `state_fe` and
`time_fe` are unit- and time-level fixed effects, both in log-rate space.
`denominator` (population/exposure) converts the log-rate surface to a
log-count rate.

### Likelihood

Two outcome families, chosen by `model.outcome_distribution` (`"NB"` or
`"Poisson"`):

- **Poisson**: `y_obs ~ Poisson(rate = exp(mu))`.
- **Negative Binomial** (`NegativeBinomial2`, mean/dispersion form): `y_obs ~ NB2(mean = exp(mu), concentration = 1 / nb_disp)`.
  Dispersion is either a fixed constant `model.nb_disp` (default `1e-4`,
  giving concentration `1/nb_disp = 10000` — nearly Poisson) or, if
  `model.sample_disp` is true, a per-unit value sampled from `Uniform(0, 1)`
  with a custom log-density penalty favoring small `disp` (`-0.5*log(disp) -
  100*sqrt(disp)`), then inverted to a concentration.
  `sample_disp` is ignored under Poisson.

**Censored small counts.** The source data suppresses small nonzero counts
(values 1–9) to protect privacy. When `model.adjust_for_missingness` is true
(default), cells flagged in `missing_idx_array` get a likelihood adjustment
instead of an observed value: `missingness_adjustment()` adds `log P(y in
{1..9})` for those cells and `log P(y not in {1..9})` for observed control
cells, integrating over the censored range rather than treating it as a
point observation.
This is separate from a cell being structurally absent from the panel
(`data.allow_unbalanced_panel`), which the array-building step handles, not
the likelihood.

### Sample sites and priors

| Site | Prior / form | Notes |
|---|---|---|
| `time_fac` | `Gamma(20, 20)`, then logged | shape `(K, rank, N)`; `time_fac_alpha=20` is fixed in code, not configurable |
| `state_fe_mu` | `ImproperUniform(real)` | shared mean, one scalar |
| `state_fe_sigma` | `HalfNormal(0.5)` | shared scale |
| `state_fe_z` | `Normal(0, 1)`, shape `(D,)` per group | `state_fe = state_fe_mu + state_fe_sigma * state_fe_z` |
| `time_fe` | `Gamma(1, 1)`, then logged | shape `(N,)` per group |
| `unit_weight` | `Dirichlet(ones(rank))`, then logged | shape `(D, rank)`; rows are a simplex over the `rank` factors |
| `disp` | fixed (`nb_disp`, default `1e-4`) or `Uniform(0,1)` if `sample_disp` | NB only; deterministic under Poisson (not constructed) |
| `treatment_it_scale` | `HalfNormal(0.1)` | scale for `treatment_kt` |
| `treatment_state_scale` | `HalfNormal(1)` | scale for `state_treatment_effect` |
| `treatment_category_scale` | `HalfNormal(1)` | scale for `category_treatment_effect` |
| `state_category_scale` | `HalfNormal(1)` | scale for `state_category_te` |
| `treatment_kt_z` | `Normal(0, 1)`, shape `(num_treated,)` | non-centered; `treatment_kt = treatment_kt_z * treatment_it_scale` |
| `state_treatment_effect_z` | `Normal(0, 1)`, shape `(D,)` | non-centered; `state_treatment_effect = state_treatment_effect_z * treatment_state_scale` |
| `state_category_te_z` | `Normal(0, 1)`, shape `(K, D)` | non-centered; `state_category_te = state_category_te_z * state_category_scale` |
| `category_treatment_effect` | `Normal(0, treatment_category_scale)` | centered (not `_z`); scale is well-identified from data |

`state_fe`, `treatment_kt`, `state_treatment_effect`, and `state_category_te`
use a **non-centered** parameterization: sample a standard-normal `_z`, then
scale it deterministically. A directly-centered `Normal(mean, scale)` prior
on these sites creates a funnel geometry when the scale is weakly identified
(sparse exposed-cell data, tight `HalfNormal(0.1)`/`HalfNormal(1)` scale
priors) — NUTS then diverges and ESS drops below 30. Non-centering removes
that funnel. `category_treatment_effect` is the exception: its scale is
data-informed (ESS around 1000), so centered sampling already mixes well and
non-centering isn't needed.

### Treatment / exposure structure

Exposure is **staggered**, not a single global cutoff: `control_idx_array`
is a per-cell boolean array (`True` = untreated), so different units can
cross into treatment at different times. `model_treated=True` requires
`control_idx_array` (raises `ValueError` otherwise) and builds `te` only on
treated cells (`~control_idx_array`): a per-treated-cell term
(`treatment_kt`), plus unit-level, group-level, and unit×group interaction
terms broadcast across the treated mask. `model_treated=False` yields
`mu = mu_ctrl` — the counterfactual/baseline path used for prediction and for
cut-model Stage 1.

### Interpreting the factor sites

`time_fac` and `unit_weight` are not identifiable on their own: any rotation
that preserves their product `time_factor` gives the same likelihood, so
individual factor values and their per-chain orderings can vary across MCMC
runs without meaning anything is wrong. This is a standard property of
low-rank factorizations, not a bug in this model. The quantities that are
identifiable and meaningful to report are the **combined** deterministic
sites: `mu_ctrl` (untreated log-rate), `mu` (log-rate including treatment),
and `te` (treatment effect) — these are what diagnostics, plots, and tables
should be read from, not the raw factor sites.

---

## Cut mode (two-stage inference)

### The problem it solves

In joint inference, one likelihood covers both the untreated baseline (factors, fixed effects) and the treatment effect. Gradients from post-exposure data flow back into the baseline factors — exposed outcomes can pull the "what would have happened without treatment" estimate toward the observed treated outcome. The cut posterior removes that path:

```
p_cut(phi, theta | Z, Y) = p(phi | Z) * p(theta | Y, phi)
```

`phi` is the baseline (factors, fixed effects); `theta` is the treatment effect; `Z` is untreated data; `Y` is the full outcome data. `phi`'s posterior depends only on `Z`.

### Stage 1: untreated baseline

`stage1_model` (`models/cut_baseline.py`) fits the same low-rank factor model as the joint model's `model_treated=False` branch — same plates, same priors, same site names (`time_fac`, `state_fe_mu`/`state_fe_sigma`/`state_fe_z`, `time_fe`, `unit_weight`, optional `disp`). It is a separate copy of that code by design — no shared helpers with `models/joint.py` — and parity between the two is pinned by `tests/test_cut_model_parity.py`.

The likelihood covers only untreated, non-missing cells (`control_idx_array & ~missing_idx_array`); censored untreated counts (present but suppressed, 1–9) are integrated over when `adjust_for_missingness=True`. It deterministically records `mu_ctrl` (log-rate surface including `log(denominators)`) and `mu` (identical to `mu_ctrl` in Stage 1).

Stage 1 runs as an ordinary multi-chain NUTS fit (`run_stage1_mcmc`, `cut.py`), gated by the same R-hat/ESS/divergence checks as joint mode. It also produces a full posterior-predictive draw of untreated counts for every retained Stage-1 sample, using an independent RNG stream (`mcmc.random_seed + 1`), written as `{stem}_stage1_ppc.csv`.

### Draw selection

`select_stage1_draws` (`cut.py`) picks `cut.num_stage1_draws` (default 25) draws from the retained Stage-1 posterior without replacement, split evenly across chains (`selection_seed`, default `mcmc.random_seed + 2`). Each pick becomes one "component," numbered 1..M in `(chain, iteration)` order — reproducible regardless of RNG draw order. Each `Stage1DrawRef` carries that one draw's `mu_ctrl` array and matching NB concentration (`1/disp` if dispersion was sampled, `1/nb_disp` if fixed; `None` for Poisson).

### Stage 2: treatment effect, one component at a time

`stage2_model` (`models/cut_treatment.py`) takes `mu_ctrl` and `nb_concentration` as plain array arguments — not sample sites. There is nothing to condition on with `numpyro.handlers.condition`; the baseline is a fixed number. This is a valid boundary because the exposed-cell likelihood depends on the Stage-1 parameters only through `mu_ctrl` and the matched concentration — nothing else about Stage 1 reaches Stage 2.

Stage 2 samples the treatment-effect hierarchy only: `treatment_it_scale`, `treatment_state_scale`, `treatment_category_scale`, `state_category_scale` (all `HalfNormal`), then `treatment_kt`, `state_treatment_effect`, `state_category_te` via non-centered `*_z ~ Normal(0,1)` scaled deterministically (funnel avoidance under sparse exposed-cell data), and `category_treatment_effect` sampled directly as `Normal(0, treatment_category_scale)` (kept centered — its scale is data-informed and mixes fine centered). `mu = mu_ctrl + te`; the likelihood covers exposed, non-missing cells only, with the same 1–9 censoring treatment as Stage 1.

Each component is its own complete multi-chain NUTS fit (`run_stage2_mcmc`), using `cut.stage2_mcmc` — every joint-mode `mcmc.*` setting except `random_seed`, overridable per key (`{**mcmc_cfg, **stage2_overlay}`). Setting `stage2_mcmc.random_seed` explicitly is rejected at config-load time; `cut.stage2_seed` (default `mcmc.random_seed + 3`) is the only Stage-2 seed. Every component reuses the same model function and array shapes so XLA does not recompile per component.

### Combined output

For each component, `subsample_component_draws` thins the retained Stage-2 draws to `cut.stage2_draws_per_component` (default 100) per chain, evenly strided — diagnostics still use the full retained draws, only the CSV/parquet output is thinned. The combined draws file carries provenance columns `cut_component`, `stage1_draw`, `stage1_chain`, `stage1_iteration` so every row traces back to the exact Stage-1 draw it was conditioned on. All components must produce the same output-draw count, or the run raises `DataError` — unequal counts would give components unequal Monte Carlo weight when pooled.

Each component gets its own convergence entry; nothing is pooled across components. These are assembled into one manifest (`build_cut_convergence_manifest`) written to `{stem}_cut_convergence.json`, holding the Stage-1 diagnostics plus a per-component list. With `--save-traces`, Stage 1 writes `{stem}_cut_stage1_traces.nc` and Stage 2 writes one `component_NNNN.nc` per component under `{stem}_cut_stage2_traces/`.

### Practical notes

- **Memory**: after each MCMC run (Stage 1 or any Stage-2 component), `_extract_fit` copies samples/diverging to host NumPy arrays, then clears NumPyro's cached JAX state off the `MCMC` object (`_states`, `_last_state`, `_cache`, etc.). Without this, each Stage-2 fit leaks roughly 1 GB of device buffers that `gc.collect()`/`jax.clear_caches()` cannot reclaim while the object stays reachable.
- **Cost**: total work scales with `cut.num_stage1_draws` — each one is a full Stage-2 multi-chain MCMC run, not a cheap conditional update.
- **Diagnostics**: a Stage-2 component is an ordinary MCMC run with its own chains, R-hat, and ESS — there's no special-cased single-draw diagnostic path.

---

## Data pipeline

Input is a single CSV in wide or prefix-wide format, one row per `(unit, time)` (or per `(unit, time, subgroup)` if outcomes come from column prefixes). `load_and_prepare()` is the one entry point: it reads the CSV, resolves the schema, converts to a standardized long DataFrame, and builds the `(K, D, N)` model arrays.

**Schema.** `data.schema` names the columns that carry meaning: `unit_col`, `time_col`, `treatment_col`, plus outcomes. Outcomes come from exactly one of two places — an explicit `outcomes: [{outcome_col, denominator_col, label}, ...]` list, or `outcomes_from_prefixes: {outcome_prefix, denominator_prefix, include}` (every column starting with `outcome_prefix` becomes one group, its label is the column name with the prefix stripped, `include` restricts which labels are kept). Exactly one of the two must be set — config load fails otherwise. A `denominator_col` is optional per outcome; when present it must be non-null and positive everywhere or load fails.

**Time.** `data.date_format` is `"auto"` by default: it tries `None` (pandas infer), then `%Y-%m-%d`, `%m/%d/%y`, `%m/%d/%Y`, `%d-%m-%Y` in that order and keeps the first that parses the whole column. Set an explicit `strftime` format to skip the guessing. `data.start_date` / `data.end_date` filter rows to `time >= start_date` and `time < end_date` (end is exclusive) after parsing.

**Optional temporal aggregation.** `data.aggregation.enabled` (default `false`) collapses rows into `monthly`, `bimonthly` (default), `quarterly`, or `yearly` buckets: outcome sums, denominator means, treatment takes `max` (a period counts as treated if any sub-period in it is).

**What `load_and_prepare()` does, in order:** parse schema → load CSV and parse time → resolve `"total"` synthetic group if requested (sums specified outcome labels; needs `total_from` or `total_all` from the model-type config) → pivot wide to long with fixed columns `unit, time, group, outcome, denominator, treatment` → reject duplicate `(group, unit, time)` rows → filter by date range → drop `exclude_units` → optional temporal aggregation → `build_model_arrays()`. Duplicate `(group, unit, time)` combinations raise `DataError` immediately; the model assumes exactly one row per cell.

**Arrays.** `build_model_arrays()` (in `arrays.py`) turns the long DataFrame into dense `(K, D, N)` NumPy arrays — `K` groups, `D` units, `N` time points, axes ordered by sorted label — using a vectorized categorical-code scatter, not a per-row loop. Denominators are divided by `denominator_scale` (default `1e4`, i.e. rates per 10,000) wherever present and positive; otherwise they default to `1`. `control_idx_array` is `True` wherever `treatment == 0`.

Two different kinds of "missing" exist and must not be confused:
- **Present but suppressed**: the outcome cell has a row but a null/NaN value (e.g. a small count masked for privacy). `missing_idx_array` is `True` there; the likelihood treats it as a 1–9 censored interval rather than an exact count.
- **Structurally absent**: no row exists for that `(group, unit, time)` at all. This only happens when `allow_unbalanced_panel: true`; otherwise `build_model_arrays` raises `DataError`. When allowed, absent cells are OR'd into the same `missing_idx_array` and logged as "structurally absent."

The convention for `missing_idx_array` is: **unset (all `False`) means nothing is missing.** A cell is only "missing" if a row said so (null value) or the panel is unbalanced and the cell has no row.

**Aggregate units (`aggregate_units.py`).** Purely post-hoc: `add_aggregate_units()` runs on a fitted model's posterior draws DataFrame, after `bpnmf run` has already produced `mu`/`ypred`/etc. It is never part of the array build or the likelihood — the model never sees or fits an aggregate unit. Each `AggregateUnitSpec` under `output.aggregate_units` picks source units via one selector — `include_units: [...]` (explicit list), `include_treated_units: true` (units with `treatment == 1` anywhere), or `include_all_units: true` — optionally minus `exclude_units`, and gives the synthetic unit a `unit` name. `strict: true` makes a missing referenced unit an error instead of a warning. `overwrite: true` lets the spec replace an existing unit of that name; otherwise a name collision raises `ConfigError`. Aggregation sums `outcome`/`ypred`/`denominator` across source units per draw, takes `max` of `treatment`, and log-sum-exps `mu`/`mu_treated` (correct pooling for log-rate columns) — each spec aggregates from the *original* draws, so chaining multiple specs never double-counts.

## Architecture

```
bpnmf CLI (cli.py)
  -> Config.from_yaml (config.py)
  -> load_and_prepare (data.py) -> build_model_arrays (arrays.py)
  -> run_mcmc_inference / generate_predictions (inference.py), via pipeline.py
  -> format_draws / format_cut_component_draws (results.py)
  -> generate_reports (reports.py) -> tables.py + plots.py
```

`cli.py` parses arguments and dispatches to `pipeline.py`, which loads config, builds arrays, calls into `inference.py` (joint mode) or `cut.py` (cut mode), writes draws via `results.py`, gates convergence via `diagnostics.py`, and triggers `reports.py` when `output.figures` is non-empty. Everything downstream of the arrays is orchestration; `models/` is where the actual NumPyro model lives.

| Module | Role |
| --- | --- |
| `cli.py` | `bpnmf` entry point: `run`/`viz`/`traces`/`init` subcommands |
| `pipeline.py` | Per-type/per-rank orchestration: load, fit, gate, write, report |
| `config.py` | Pydantic config schema — single source of truth for shape/defaults |
| `data.py` | CSV load, schema resolution, wide-to-long, filtering |
| `arrays.py` | Long DataFrame -> `(K, D, N)` NumPy arrays |
| `models/joint.py` | Joint NumPyro model: factors, fixed effects, treatment, likelihood |
| `models/cut_baseline.py` | Cut Stage 1: untreated-baseline model only |
| `models/cut_treatment.py` | Cut Stage 2: treatment-effect model only |
| `models/likelihood.py` | Shared missingness/censoring likelihood helpers |
| `inference.py` | NUTS/MCMC run + posterior prediction (joint mode) |
| `diagnostics.py` | R-hat/bulk-tail-ESS/divergence convergence gate |
| `cut.py` | Two-stage cut-posterior orchestration: seeds, draw selection, Stage-1/2 runners |
| `results.py` | Posterior draws -> tidy reporting DataFrame |
| `reports.py` | Figure/table orchestration entry point |
| `tables.py` | Pandas-only summary tables + terminal rendering |
| `plots.py` | Matplotlib figures and PPC plots |
| `aggregate_units.py` | Post-hoc synthetic reporting units on draws |
| `parallelism.py` | Picks chain count/method from visible JAX devices |
| `checks.py` | Runtime validators for arrays, samples, filepaths |
| `validation.py` | `ConfigError`/`DataError` exception types |
| `logging_config.py` | Loguru setup |

`models/joint.py` and the `models/cut_baseline.py` + `models/cut_treatment.py` pair are intentionally duplicated, not shared: the cut Stage 1/2 models reimplement the joint model's untreated and treatment blocks independently rather than importing from `joint.py`. `tests/test_cut_model_parity.py` pins the two implementations to identical site names, shapes, sampled values, and log-densities, so any drift between the copies fails the test suite rather than silently diverging.

---

## Installation

Requires **Python 3.12–3.14** and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> bayesian_panel_nmf
cd bayesian_panel_nmf
uv sync --all-extras --dev
```

Installs the full stack, pinned in `pyproject.toml` and locked in `uv.lock`:

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
Default install is CPU-only; see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for CUDA/ROCm/TPU wheels. Chain parallelism picks itself from the visible
devices (see `mcmc.auto_parallelism`).

### Verify the install

```bash
uv run python -c "import jax, numpyro, arviz; print(jax.__version__, numpyro.__version__, arviz.__version__)"
# expected: 0.10.x 0.21.x 1.2.x
```

---

## Configuration reference

Configs are YAML files validated by a typed schema (`bayesian_panel_nmf.config.Config`,
pydantic v2). Unknown keys are **rejected** (typos fail fast), and booleans must be
unquoted (`true`/`false`, not `"true"`). Every default below lives in the schema,
so an omitted value resolves the same way everywhere.

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

#### Model types and `ranks_to_test`

Each entry under `model.types` is an **independent model fit** over a set of
outcome `groups`. The map key is a label you choose (e.g. `total`, `age`,
`race`) — it names the output subdirectory only, nothing else.

**`rank` is the number of latent factors** in the low-rank factorization — the
core modeling assumption. The untreated outcome surface is approximated by
`rank` shared time/unit factors (the "NMF": nonnegative matrix
factorization). A higher rank fits more shared structure but is harder to
identify and slower to converge; a lower rank is more parsimonious but may
underfit. Typical values are small single digits — the shipped
fertility/education configs use `5`.

`ranks_to_test` is a **list**; each value produces its own independent
fit and output files (`{dist}_{outcome}_{type}_{rank}.*`) — a
sweep, **not** automatic model selection. `ranks_to_test: [3, 5, 10]` runs
three full analyses to compare; `ranks_to_test: [5]` runs one.
A mis-set rank usually shows up in the diagnostics
(persistent non-convergence, very low ESS) — see "Interpreting diagnostics".

### `mcmc`

| Key | Default | Notes |
| ----- | --------- | ------- |
| `auto_parallelism` | `true` | pick chains/method from devices |
| `max_chains` | `4` | cap under auto_parallelism |
| `num_chains` / `chain_method` | null | only used when auto_parallelism=false |
| `num_warmup` / `num_samples` | `1000` / `2500` | |
| `thinning` | `10` | |
| `target_accept` | `0.8` | NUTS target acceptance probability. Raise toward `0.9`–`0.99` for funnel/high-curvature geometry (smaller step size → fewer divergences, more leapfrog steps per sample). Overridable per-stage via `cut.stage2_mcmc.target_accept`. |
| `random_seed` | `8675309` | |
| `progress_bar` | `true` | |
| `gate_params` | null | prefixes of sample-site names the convergence gate checks; null = all (see [Interpreting diagnostics](#interpreting-diagnostics)) |
| `convergence.rhat_warn` / `convergence.rhat_fail` | `1.01` / `1.05` | R-hat PASS/WARN and FAIL band edges for the per-site diagnostics status |
| `convergence.ess_min` | `400.0` | ESS warn line — `converged` requires `min(bulk, tail) ESS >= ess_min` |
| `convergence.ess_fail_fraction` | `0.25` | ESS below `ess_min * ess_fail_fraction` (= 100 by default) is a hard FAIL |

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

Figure/table code is split across three modules: `plots.py` (matplotlib
plotting primitives), `tables.py` (pure pandas/numpy table computation plus rich
terminal rendering — no matplotlib), and `reports.py` (`generate_reports()`,
the entry point that calls into both).

After each rank's artifacts are written, `bpnmf run` calls
`tables.print_run_summary_panel()`: a rich terminal panel echoing that run's
config (model type, rank, chains/method, outcome distribution), the
**divergence count** as a plain fact, the selected figures, and the written
artifact paths. The panel carries **no roll-up PASS/FAIL verdict** — the
per-parameter / per-component diagnostics table printed above it shows the
config-driven `ConvergenceThresholds` status (PASS/WARN/FAIL) per site, and the
authoritative gate is the `*_convergence.json` write plus its logged warning
(see [Interpreting diagnostics](#interpreting-diagnostics)). Purely additive
terminal output — no file or data side effects, so it never touches a
golden-checked artifact.

| Registry name | Function | Artifact |
| --- | --- | --- |
| `unit_fit` | `make_unit_fit_plot` | `fit_<target>.png` |
| `unit_gap` | `make_unit_gap_plot` | `gap_<target>.png` |
| `raw_rate` | `make_raw_rate_plot` | `raw_rate.png` |
| `interval` | `make_interval_plot` | `interval.png` |
| `group_comparison` | `make_group_comparison_plot` | `group_comparison.png` |
| `ppc` | `make_all_ppc_plots` | `ppc/*.png`, `ppc/ppc_pvalues.csv` |

`output.figures` accepts:

- `true` ("render everything") or `false` (render no
  figures) — the original boolean spelling still works. The config default
  is `false` (equivalently `[]`), i.e. render nothing.
- `"all"` / `"none"` — explicit string spellings of the same two extremes.
- a list of registry names, e.g. `figures: ["interval", "ppc"]` — renders
  only those. Unknown names are rejected at config-load time.

All spellings normalize to a canonical `list[str]` (the names to render).
`bpnmf run` skips reporting entirely (no figures *and* no
tables) when the normalized selection is empty, matching the original
`figures: false` behavior. Calling `reports.generate_reports(..., figures=
["interval"])` directly still always writes the always-on tables — only the
PLOT_REGISTRY figures are gated at that level.

---

## Output artifacts

Each model type writes to `<output_dir>/<type>/`. Draws filenames follow
`{distribution}_{outcome}_{type}_{rank}` (e.g. `NB_births_total_3`); cut mode
appends `_cut`.

### Joint mode

| File | Contents |
| ---- | -------- |
| `{stem}.csv` or `.parquet` | Tidy posterior draws (one row per draw × group × unit × time): `.draw/.chain/.iteration`, `unit/time/group`, `outcome/denominator/treatment`, `ypred` (counterfactual untreated), `mu` (log-rate control), `mu_treated`. Format set by `output.draws_format`. The large artifact (100 MB–1 GB for multi-group types). |
| `{stem}_convergence.json` | Always-on gate: `{rhat_max, ess_bulk_min, ess_tail_min, divergences, converged}` (defaults: R-hat<1.01, ESS≥400, 0 divergences; bands configurable via `mcmc.convergence`). |
| `df_{type}.csv` | Preprocessed observed data (standardized columns). |
| `{stem}_traces.nc` | Full posterior NetCDF sidecar (only with `--save-traces`). |
| `figs/` | PPC panels, fit/gap/interval/raw-rate plots, summary tables (when `output.figures: true`). |

### Cut mode (two-stage) — additional files

| File | Contents |
| ---- | -------- |
| `{stem}_cut.csv`/`.parquet` | Combined Stage-2 draws with columns `cut_component, stage1_draw, stage1_chain, stage1_iteration`. `.draw` is globally unique across components; `.chain`/`.iteration` are the real Stage-2 chain/subsample index. |
| `{stem}_cut_stage1_ppc.csv` | Full Stage-1 posterior-predictive draws (feeds the PPC suite only). Always CSV. |
| `{stem}_cut_convergence.json` | Per-stage manifest: Stage-1 gate + every conditional Stage-2 fit's gate; top-level `converged` true only if Stage 1 and all components passed. |
| `{stem}_cut_stage1_traces.nc`, `{stem}_cut_stage2_traces/component_*.nc` | Trace sidecars (only with `--save-traces`). |

Regenerate figures from a saved draws file without re-running MCMC:

```bash
uv run bpnmf viz --results results/total/NB_births_total_3.csv
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
| `ess_bulk_min` | smallest bulk effective sample size | `>= 400` |
| `ess_tail_min` | smallest tail effective sample size | gated jointly with bulk: the gate uses `min(bulk, tail) >= 400` |
| `divergences` | total divergent transitions | `== 0` |
| `converged` | `true` only if R-hat, min(bulk, tail) ESS, and divergences all pass the PASS band | — |

The numeric bars above are the **defaults**; the band edges are configurable
via `mcmc.convergence.{rhat_warn, rhat_fail, ess_min, ess_fail_fraction}`
(PASS/WARN/FAIL per site; the divergence `== 0` requirement is fixed).

A failed gate logs a warning and the run continues — it never silently drops
output. On failure: increase `mcmc.num_warmup`/`num_samples`, or check
the trace sidecars (run with `--save-traces`, then
`bpnmf traces <nc_path>` for the numeric table or `bpnmf traces <nc_path> --plots`
for visual traces). In cut mode the manifest reports Stage-1 and every Stage-2
fit separately; diagnostics are never pooled across conditional targets.

### Gating only the parameters you care about (`mcmc.gate_params`)

The low-rank factorization contains sites that are individually
non-identifiable by construction (`state_fe`, `time_fe`, `time_fac`,
`unit_weight` trade off against each other while their sum — the log-rate
surface — is stable). Their R-hat can legitimately fail even when every
quantity you report has mixed, so gating the whole posterior cries wolf
forever. `mcmc.gate_params` restricts the R-hat/ESS gate to the sample-site
name **prefixes** you list:

```yaml
mcmc:
  gate_params:
    - "mu"                  # mu, mu_ctrl
    - "te"
    - "treatment"           # treatment_{it,state,category}_scale, treatment_kt
    - "state_treatment"     # state_treatment_effect
    - "category_treatment"  # category_treatment_effect
    - "state_category"      # state_category_te, state_category_scale
```

Semantics:

- **Prefix match** (`startswith`) on posterior variable names. Beware short
  prefixes: `"state"` would also sweep in `state_treatment_effect` and
  `state_category_*`; use the longest unambiguous prefix.
- **Omit the key (or null) = gate everything** — the original default; the
  convergence JSON is unchanged. When set, the JSON gains a `gate_params`
  key recording what was gated.
- **You can include the non-identifiable sites too** — nothing is
  hard-excluded; add `"state_fe"`, `"time_fe"`, `"time_fac"`,
  `"unit_weight"` back at will, or drop the key for the full gate.
- **Divergences are always counted over the full run** regardless of the
  list — a divergence anywhere invalidates the geometry everywhere.
- **Threshold bands are configurable** via `mcmc.convergence`
  (`rhat_warn` / `rhat_fail` / `ess_min` / `ess_fail_fraction`); the defaults
  reproduce the original gate (R-hat < 1.01, ESS ≥ 400, 0 divergences —
  divergences are always a hard gate).
- **One list serves joint and both cut stages.** Each gate filters the list
  against its own posterior: cut Stage-1 (baseline: `mu`, `time_fe`,
  `time_fac`, `unit_weight`, `disp`) matches only `"mu"` from the list
  above; Stage-2 components match the full treatment block. Prefixes that
  match nothing in a given stage are silently skipped — only a list that
  matches **zero** sites in a stage raises (typo protection). Practical
  rule: always include `"mu"` so Stage-1 has something to gate.

Try it on the smoke tests (fast, ~3-5 min each):

```bash
# 1. Baseline: run the joint smoke config as-is (gates ALL parameters —
#    expect converged: false; the smoke settings are tiny on purpose)
uv run bpnmf run --config configs/fertility_smoke_test.yaml
cat results/total/NB_births_total_3_convergence.json

# 2. Add gate_params to a copy and re-run — rhat_max/ess now reflect
#    only the gated sites, and the JSON records gate_params
sed 's/^  random_seed:/  gate_params: ["mu", "te", "treatment", "state_treatment", "category_treatment", "state_category"]\n  random_seed:/' \
  configs/fertility_smoke_test.yaml > /tmp/smoke_gated.yaml
uv run bpnmf run --config /tmp/smoke_gated.yaml
cat results/total/NB_births_total_3_convergence.json

# 3. Same for cut mode (Stage-1 gates on "mu" alone; every Stage-2
#    component gates on the treatment block):
sed 's/^  random_seed:/  gate_params: ["mu", "te", "treatment", "state_treatment", "category_treatment", "state_category"]\n  random_seed:/' \
  configs/fertility_cut_smoke_test.yaml > /tmp/cut_smoke_gated.yaml
uv run bpnmf run --config /tmp/cut_smoke_gated.yaml
# cut mode writes the same filename; it holds the Stage-1 + per-component manifest
cat results/total/NB_births_total_3_convergence.json
```

Gating fewer parameters is a *reporting* choice, not a fix for the
non-identifiability — the JSON's `gate_params` key keeps the narrowing
visible to anyone reading the artifact.

---

## CLI reference

Installed as the `bpnmf` console script (`bpnmf = "bayesian_panel_nmf.cli:main"`
in `pyproject.toml`); also runnable as `python -m bayesian_panel_nmf.cli`.
Four subcommands:

### `bpnmf run --config <path> [options]`

Runs the analysis pipeline (data load → MCMC → convergence gate → draws +
reporting) for one or all configured model types.

| Flag | Meaning |
| ---- | ------- |
| `--config` | Path to config YAML. **Omit it (in a terminal) for an interactive picker** — choose a config from `./configs/*.yaml`, a model type, and save-traces. Non-TTY without `--config` errors. |
| `--type` | Restrict to one configured model type; default runs all |
| `--rank` | Override the rank(s) to test from the config |
| `--verbose` / `-v` | DEBUG-level logging |
| `--log-file` | Also write logs to this file |
| `--save-traces` | Write a NetCDF posterior sidecar (`<stem>_traces.nc`) |
| `--chains` | Override chain count (`max_chains` under auto_parallelism, or literal `num_chains` with `--chain-method`) |
| `--chain-method` | Force `sequential`/`parallel`/`vectorized`, disabling `mcmc.auto_parallelism` |

### `bpnmf viz [--results <draws>] [--config <cfg>] [options]`

Re-renders figures + tables from an existing draws artifact without
re-running MCMC — the same reporting path `bpnmf run` calls automatically
when `output.figures` resolves to a non-empty figure list.

**To reproduce a configured run's figures, pass the same `--config`.** Without
it, viz renders every figure with defaults and ignores the `output` block
(so PPC lags, `aggregate_units`, `fit_gap_per_unit`, etc. are lost). With
`--config`, viz uses the same reporting path as `bpnmf run`:

```bash
bpnmf viz --config configs/my_config.yaml --results results/total/NB_births_total_5.parquet
```

| Flag | Meaning |
| ---- | ------- |
| `--results` | Path to the draws CSV/parquet. **Omit it (in a terminal) for an interactive picker** — discovers draws under `./*results*/**/*` and, for cut runs, auto-attaches the `*_stage1_ppc.csv` sidecar, then offers an optional config
file to drive figure selection / `target` / `aggregate_units`. Non-TTY without `--results` errors. |
| `--config` | Config YAML whose `output` block drives the re-render (figures, `aggregate_units`, `ppc_*`, `fit_gap_per_unit`, ...). Omit to render every figure with defaults. `--target`/`--group` override the config. |
| `--ppc-results` | Optional Stage-1 PPC draws CSV (cut mode) to route the PPC suite to the full Stage-1 posterior |
| `--target` | Target unit for fit/gap/summary plots (auto-detected if omitted); overrides `--config` |
| `--group` | Group label to render; repeatable; omit to render every group present; overrides `--config` |
| `--tables-only` | Print/write the effect tables (`summary_table*.csv`, `post_treatment_summary.csv`, and their rich terminal render) and skip **all** figures. Fast way to re-inspect effects. Orthogonal to `--group`/`--target` (those still choose *which* group/unit; `--tables-only` only drops figures). |

**Which results directory does viz read?** There is no config-driven default —
the draws-file path you pass *is* the locator, and its parent directory is where
tables/figures are written (`output_dir = <results path>.parent`). Even with
`--config`, the config only drives figure selection / `aggregate_units` /
`target` — **not** where to look; `output.output_dir` is not consulted by viz.
Omit `--results` in a terminal for an interactive picker that globs
`./*results*/**/*` under the current working directory (so `results/`,
`test_results/`, `test_results_cut/`, … all match; `figs/`, `df_*`, and `*_ppc`
are excluded). Non-TTY without `--results` errors.

**Group selection nuance.** `--group` filters the **per-unit** table
(computed per `(unit, group)`) as expected. The **headline** summary table is
unit-anchored and sums over whatever groups are present, so on a multi-group
(K>1) run it pools groups rather than breaking them out; use `--group` to
restrict the input frame for a single group's headline.

```bash
# effects tables only, one group, from an explicit cut draws file
bpnmf viz --results test_results_cut/education/NB_births_education_10_cut.csv \
  --config configs/my_config.yaml --tables-only --group hs
```

### `bpnmf traces [nc_path] [options]`

Default: a color-coded R-hat/ESS table per parameter, worst first
(computed natively from the NetCDF sidecar via ArviZ). With `--plots`:
renders PNG trace plots instead.

Three input forms:

- **Omit `nc_path`** (in a terminal): interactive picker over every trace
  sidecar and cut `*_stage2_traces/` directory found under `./*results*/**/*`.
- **A `.nc` file**: full per-parameter table for that fit (joint sidecar,
  cut Stage-1 sidecar, or a single Stage-2 `component_NNNN.nc`).
- **A `*_stage2_traces/` directory** (cut mode): one summary row per
  component — worst parameter, its R-hat/ESS, status — plus a pass count.
  Diagnostics stay per-component; nothing is pooled across components.

Sites that are **constant in the file** (zero posterior variance) show
dimmed as `fixed` with no R-hat/ESS and never count as failures. This is
detected empirically, so it is always correct for the model at hand: `disp`
under `sample_disp: false` is `fixed` in both models, `mu_ctrl` is `fixed`
only in cut Stage-2 components (where the baseline is frozen by design),
and a sampled `disp` gets real diagnostics.

| Flag | Meaning |
| ---- | ------- |
| `--plots` | Render PNG trace plots instead of the numeric table |
| `--param-filter` | Comma-separated parameter-name prefixes to restrict to |
| `--out-dir` | Output directory for `--plots` (default `<input_dir>/figs/diagnostics/`) |

Exit code is 0 only if no gated parameter FAILs (numeric-table and
directory-summary modes — usable in scripts; `--plots` exits non-zero only
when no plots were written, not on gate FAILs).

### `bpnmf init [path] [--force]`

Writes a starter config (a copy of the commented `configs/base_config.yaml`
template) to `path` (default `config.yaml`). Refuses to overwrite an
existing file unless `--force` is given.

```bash
uv run bpnmf init configs/my_config.yaml
```

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
`models/cut_treatment.py` duplicate the factor/treatment blocks on purpose
rather than import shared helpers, so an edit to one cannot silently change the
other's isolation guarantee. Parity is enforced by `tests/test_cut_model_parity.py`
instead of by code sharing.

### Why single-CPU hosts run chains sequentially

`choose_mcmc_parallelism` (in `parallelism.py`) picks `chain_method` from the
visible JAX devices: sequential on one CPU, vectorized on one GPU, parallel
across multiple devices. XLA's CPU backend shares one thread pool across
logical devices, so forcing multiple CPU devices for `parallel` gives no real
speedup — sequential is the honest default rather than false parallelism.
