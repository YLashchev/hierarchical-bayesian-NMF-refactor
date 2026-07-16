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

`figures` (false), `clean` (false), `save_traces` (false), `print_tables` (true),
`print_target_table` (true); optional reporting filters `target_unit`, `report_groups`,
`aggregate_units`, `ppc_units`, `ppc_exclude_units`, `ppc_acf_lags` (default `[6]`),
`ppc_unit_corr_max_time`. `draws_format` (`"csv"` default, or `"parquet"`) controls
only the large draws artifact (joint draws / cut combined draws); human-facing
tables (`df_{type}.csv`, `summary_table*.csv`, `expected_vs_observed.csv`,
`post_treatment_summary.csv`, `ppc_pvalues.csv`, `stage1_ppc.csv`) always stay CSV.

### `cut` (only read when `model.inference_mode: "cut"`)

`num_stage1_draws` (25), `stage2_draws_per_component` (100), `selection_seed`
(default `mcmc.random_seed + 2`), `stage2_seed` (default `+ 3`), `stage2_mcmc`
(overlay on `mcmc` for the cheaper conditional fits). The distinct seed offsets keep
draw *selection* independent of the draws themselves; setting `stage2_mcmc.random_seed`
is rejected (`cut.stage2_seed` is the authority).
