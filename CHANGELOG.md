# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- `tables.print_run_summary_panel()`: a rich terminal panel summarizing one
  completed rank run (model type, rank, chains/method, outcome distribution,
  convergence gate PASS/FAIL, selected figures, artifact paths). Called from
  `scripts/run_analysis.py` after a rank's artifacts are written, in both the
  joint (`_run_single_rank`) and cut (`_run_cut_rank`) paths. Purely additive
  terminal output — no file or data side effects, so golden output is
  unaffected.
- Typed config schema (`bayesian_panel_nmf.config.Config`, pydantic v2):
  validates configs, fills defaults from one place, and rejects unknown keys
  and quoted-boolean YAML footguns with clear errors. `validate_config` now
  delegates to it.
- Opt-in Parquet output for the large draws artifact via `output.draws_format`
  (`"csv"` default, or `"parquet"`); applies only to the joint/cut combined
  draws files, never the human-facing summary/table CSVs.
- `PLOT_REGISTRY` in `plots.py` (`unit_fit`, `unit_gap`, `raw_rate`,
  `interval`, `group_comparison`, `ppc`) plus config-driven figure
  selection: `output.figures` now accepts `bool | list[str] | "all"/"none"`
  (previously a plain boolean), normalizing to the set of registry names to
  render. `reporting.generate_reports()` gained a `figures: list[str] | None`
  parameter (`None` = render everything, back-compat default) that selects
  which registry entries render; always-on tables (`summary_table.csv` etc.)
  are unaffected by this selection. See documentation.md's Visualization
  section.

### Changed

- Renamed modules for legibility (internal import paths changed):
  `visualization`→`plots`, `mcmc_utils`→`parallelism`, `cut_inference`→`cut`,
  `output`+`cut_output`→`results`, `models/panel_nmf_model`→`models/joint`,
  `models/utils`→`models/likelihood`,
  `models/cut_stage1_model`→`models/cut_baseline`,
  `models/cut_stage2_model`→`models/cut_treatment`. Public API
  (`from bayesian_panel_nmf import ...`) is unchanged.
- Split `reporting.py` into `reports.py` (orchestration entry point,
  `generate_reports()`) and a new `tables.py` (table computation + rich
  terminal rendering, no matplotlib): moved `make_summary_table` out of
  `plots.py` (it had zero matplotlib references) and moved
  `_compute_quantiles`, `_auto_detect_target`, `_slug`,
  `_compute_per_unit_post_treatment`, `_print_rich_tables` out of
  `reporting.py`/`reports.py`. Table math is unchanged (verified
  bit-identical against golden CSVs); import paths changed:
  `bayesian_panel_nmf.reporting` → `.reports`,
  `bayesian_panel_nmf.plots.make_summary_table` →
  `bayesian_panel_nmf.tables.make_summary_table`. Public API
  (`generate_reports`, `make_summary_table`) unchanged otherwise.
- Split `arrays.py` out of `data.py` and vectorized the (K, D, N) array build
  (replaced a per-row `DataFrame.iterrows()` loop with a categorical-codes
  scatter). Output is bit-identical.
- Moved the convergence gate (`convergence_summary`) from `inference.py` into a
  dedicated `diagnostics.py`. Still exported as
  `bayesian_panel_nmf.convergence_summary` (public API unchanged).
- Upgraded the core stack to JAX 0.10 / NumPyro 0.21 / ArviZ 1.2 (DataTree).
  Posterior draws are not bit-comparable to pre-upgrade runs because JAX's
  default PRNG changed (`jax_threefry_partitionable`); statistical results
  (posterior means, convergence) are unchanged. No source migration was
  needed — ArviZ 1.2 kept the `az.from_dict` / `az.summary` / `to_netcdf`
  call surface this package uses.
- Raised `requires-python` to `>=3.12,<3.15`; dropped the direct `jaxlib`
  pin (resolved transitively via `jax`).
- `plots.py` teardown: extracted three private helpers
  (`_new_fig`/`_new_grid_fig`, `_empty_placeholder_fig`, `_finalize`) that
  dedup the repeated `plt.subplots`/empty-data-placeholder/tight_layout
  scaffolding across all 11 `make_*` plotting functions. No data-producing
  computation changed; every `make_*` signature and return type is
  unchanged (verified bit-identical against `tests/test_golden.py`).

## 0.1.0

- Initial package architecture
