# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Typed config schema (`bayesian_panel_nmf.config.Config`, pydantic v2):
  validates configs, fills defaults from one place, and rejects unknown keys
  and quoted-boolean YAML footguns with clear errors. `validate_config` now
  delegates to it.
- Opt-in Parquet output for the large draws artifact via `output.draws_format`
  (`"csv"` default, or `"parquet"`); applies only to the joint/cut combined
  draws files, never the human-facing summary/table CSVs.

### Changed

- Upgraded the core stack to JAX 0.10 / NumPyro 0.21 / ArviZ 1.2 (DataTree).
  Posterior draws are not bit-comparable to pre-upgrade runs because JAX's
  default PRNG changed (`jax_threefry_partitionable`); statistical results
  (posterior means, convergence) are unchanged. No source migration was
  needed — ArviZ 1.2 kept the `az.from_dict` / `az.summary` / `to_netcdf`
  call surface this package uses.
- Raised `requires-python` to `>=3.12,<3.15`; dropped the direct `jaxlib`
  pin (resolved transitively via `jax`).

## 0.1.0

- Initial package architecture
