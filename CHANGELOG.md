# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed (Phase 10 cleanup)

- `aggregate_units.add_aggregate_units()` now delegates all config-shape
  validation to the existing pydantic `AggregateUnitSpec`
  (`config.py`) instead of re-validating it by hand. Deleted the
  hand-rolled `_as_bool`, `_as_str_list`, `_validate_aggregate_spec`, and
  `_active_selectors` helpers (~54 net lines removed from
  `aggregate_units.py`, 225→171 lines). `add_aggregate_units` now accepts
  `Sequence[AggregateUnitSpec | dict]` and coerces plain dicts (e.g. from
  YAML/tests) via `AggregateUnitSpec.model_validate()` at the top, wrapping
  any `pydantic.ValidationError` as `ConfigError` to preserve the existing
  exception contract. The "exactly one include selector" rule (business
  logic, not shape) moved to a new `AggregateUnitSpec._check_selectors`
  `@model_validator` in `config.py`, so it is now enforced at config-load
  time too. The unit-name/`overwrite` collision check (also business logic)
  stays in `add_aggregate_units`, now reading typed attributes
  (`spec.unit`, `spec.overwrite`, ...) instead of `dict.get(...)`.
  `pipeline.py` and `reports.py` pass typed `AggregateUnitSpec` objects
  through end-to-end instead of `spec.model_dump()`'d dicts. Pure
  validation/plumbing change — aggregation math (`_aggregate_one`,
  `_logsumexp_series`, `_source_units_for_spec`'s selection logic) is
  untouched; golden output remains bit-identical.
- Renamed the missingness-adjustment `numpyro.handlers.scope` prefix
  `low_births` → `suppressed_counts` (domain-neutral; the package models
  suppressed small counts generally, not births specifically). This affects
  only scoped sample-site keys, which are stripped before any draws/table/
  trace serialization — draws CSVs, tables, and convergence JSON are
  bit-identical. The `missing_factors` / `nonmissing_factors` factor names
  were already domain-neutral and were kept.

### Added

- `tables.print_run_summary_panel()`: a rich terminal panel summarizing one
  completed rank run (model type, rank, chains/method, outcome distribution,
  convergence gate PASS/FAIL, selected figures, artifact paths). Called from
  the pipeline after a rank's artifacts are written, in both the
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

- **Hard break**: deleted `scripts/run_analysis.py`, `scripts/generate_full_viz.py`,
  `scripts/analyze_traces.py`, `scripts/make_trace_plots.py` and replaced them
  with one installed console script, `bpnmf` (Phase 9.2 of the legibility
  refactor; `src/bayesian_panel_nmf/cli.py`, `bpnmf = "bayesian_panel_nmf.cli:main"`
  in `pyproject.toml`). No shims — direct invocations of the deleted scripts
  no longer work. New subcommands: `bpnmf run` (was `run_analysis.py`),
  `bpnmf viz` (was `generate_full_viz.py`), `bpnmf traces [--plots]` (folds
  `analyze_traces.py`'s numeric table and `make_trace_plots.py`'s PNGs into
  one subcommand), and a new `bpnmf init` that writes a starter config
  (copy of `configs/base_config.yaml`). Flags, defaults, and behavior for
  `run`/`viz`/`traces` are unchanged — this is a move + repackage, not a
  behavior change; golden output remains bit-identical. The module-top
  `numpyro.set_host_device_count()`-before-jax-import ordering is preserved
  verbatim at the top of `cli.py`. `python -m bayesian_panel_nmf.cli` also
  works. See documentation.md's new "CLI reference" section.
- Extracted the analysis-pipeline orchestration out of `scripts/run_analysis.py`
  into a new `src/bayesian_panel_nmf/pipeline.py` (Phase 9.1 of the legibility
  refactor): `run_model_type`, `_run_sequential`, `_run_single_rank`,
  `_run_cut_rank`, `_publish_cut_artifacts`, `_prepare_type_output_dir`,
  `_run_reporting`, `_clean_scoped_samples`, `_write_draws`, `_read_draws`,
  `_draws_filename`, `_get_outcome_name`, `_validate_run_analysis_config`,
  `_select_types_to_run` moved verbatim. `scripts/run_analysis.py` keeps only
  the module-top `numpyro.set_host_device_count()` call, `_parse_args`,
  `load_config`, `_apply_mcmc_overrides`, `_format_elapsed`, and `main()`
  (argparse/CLI split is Phase 9.2). Also moved `_safe_rmtree` into
  `checks.py` (it is a runtime filesystem-safety guard, not orchestration).
  Behavior, draws/table outputs, and the `set_host_device_count`-before-jax
  import ordering are unchanged; golden output remains bit-identical.
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
