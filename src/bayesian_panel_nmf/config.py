"""Pydantic v2 config schema for bayesian_panel_nmf.

Mirrors the shape validated ad hoc by ``validation.validate_config`` /
``_require_bool``, as a strict, self-documenting schema. Every model forbids
unknown keys (``extra="forbid"``) except ``ModelConfig.types``, whose keys are
arbitrary user-chosen model-type names (e.g. "age", "race", "total"). Booleans
use ``StrictBool`` so quoted YAML strings like ``"false"`` are rejected rather
than silently treated as truthy.

This module only builds the schema; wiring it into consumers (run_analysis.py,
cut.py, inference.py) and retiring the corresponding validation.py checks is a
separate, later task.
"""

from __future__ import annotations

from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    model_validator,
)

from bayesian_panel_nmf.validation import ConfigError

_STRICT = ConfigDict(extra="forbid")


class OutcomeSpec(BaseModel):
    """One explicit outcome column definition under ``schema.outcomes``."""

    model_config = _STRICT

    outcome_col: str
    label: str
    denominator_col: str | None = None


class OutcomesFromPrefixes(BaseModel):
    """Shortcut for deriving many outcomes from shared column prefixes."""

    model_config = _STRICT

    outcome_prefix: str
    denominator_prefix: str | None = None
    include: list[str] | None = None


class SchemaConfig(BaseModel):
    """``data.schema`` — column roles and outcome definitions."""

    model_config = _STRICT

    unit_col: str
    time_col: str
    treatment_col: str
    outcomes: list[OutcomeSpec] | None = None
    outcomes_from_prefixes: OutcomesFromPrefixes | None = None

    @model_validator(mode="after")
    def _check_outcomes_xor(self) -> SchemaConfig:
        has_outcomes = self.outcomes is not None
        has_prefixes = self.outcomes_from_prefixes is not None
        if has_outcomes == has_prefixes:
            raise ValueError(
                "schema must define exactly one of 'outcomes' or "
                "'outcomes_from_prefixes'"
            )
        if has_outcomes and len(self.outcomes) == 0:  # type: ignore[arg-type]
            raise ValueError("schema['outcomes'] must be non-empty list")
        return self


class AggregationConfig(BaseModel):
    """``data.aggregation`` — temporal aggregation settings."""

    model_config = _STRICT

    enabled: StrictBool = False
    period: str = "bimonthly"


class DataConfig(BaseModel):
    """``data`` section — input file, schema, filtering, aggregation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    input_file: str
    output_dir: str
    schema_: SchemaConfig = Field(alias="schema")
    # "auto" triggers data.py's format auto-detection branch; matches the
    # legacy .get("date_format", "auto") default. Must stay "auto", not None.
    date_format: str = "auto"
    start_date: str | None = None
    end_date: str | None = None
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    allow_unbalanced_panel: StrictBool = False
    # Explicit override for run_analysis.py's _get_outcome_name() filename
    # placeholder; undocumented but tested (test_run_analysis_parallel.py).
    # Falls back to deriving from outcomes_from_prefixes, then "births".
    outcome: str | None = None


class TypeConfig(BaseModel):
    """One entry under ``model.types.<name>`` — an independent model fit."""

    model_config = _STRICT

    groups: list[str]
    ranks_to_test: list[int]
    total_from: list[str] | None = None
    total_all: StrictBool = False
    exclude_units: list[str] | None = None


class ModelConfig(BaseModel):
    """``model`` section — distribution, model types, dispersion, treatment."""

    model_config = _STRICT

    outcome_distribution: str = "NB"
    types: dict[str, TypeConfig] = Field(default_factory=dict)
    nb_disp: float = 1e-4
    sample_disp: StrictBool = False
    adjust_for_missingness: StrictBool = True
    model_treated: StrictBool = True
    inference_mode: str | None = None

    @model_validator(mode="after")
    def _check_inference_mode(self) -> ModelConfig:
        if self.inference_mode is not None and self.inference_mode not in (
            "joint",
            "cut",
        ):
            raise ValueError(
                "model.inference_mode must be 'joint' or 'cut', got "
                f"{self.inference_mode!r}"
            )
        if self.inference_mode == "cut" and not self.model_treated:
            raise ValueError(
                "model.inference_mode='cut' requires model_treated=true "
                "(the cut model estimates treatment effects)"
            )
        return self


class MCMCConfig(BaseModel):
    """``mcmc`` section — NUTS/chain configuration."""

    model_config = _STRICT

    auto_parallelism: StrictBool = True
    max_chains: int = 4
    num_chains: int | None = None
    chain_method: str | None = None
    num_warmup: int = 1000
    num_samples: int = 2500
    thinning: int = 10
    random_seed: int = 8675309
    progress_bar: StrictBool = True


class AggregateUnitSpec(BaseModel):
    """One synthetic reporting-only aggregate unit under ``output.aggregate_units``."""

    model_config = _STRICT

    unit: str
    include_treated_units: StrictBool = False
    include_all_units: StrictBool = False
    include_units: list[str] | None = None
    exclude_units: list[str] | None = None
    strict: StrictBool = False
    overwrite: StrictBool = False


class OutputConfig(BaseModel):
    """``output`` section — figures, tables, reporting/PPC filters."""

    model_config = _STRICT

    figures: StrictBool = False
    clean: StrictBool = False
    save_traces: StrictBool = False
    target_unit: str | None = None
    report_groups: list[str] | None = None
    print_tables: StrictBool = True
    print_target_table: StrictBool = True
    aggregate_units: list[AggregateUnitSpec] | None = None
    ppc_units: list[str] | None = None
    ppc_exclude_units: list[str] | None = None
    ppc_acf_lags: list[int] | None = None
    ppc_unit_corr_max_time: str | None = None


class CutConfig(BaseModel):
    """``cut`` section — optional two-stage cut-posterior settings.

    ``stage2_mcmc`` is kept as a raw dict (rather than a nested MCMCConfig) to
    preserve the existing shallow-merge-over-``mcmc`` semantics in
    ``cut.py``: ``{**mcmc_cfg, **stage2_mcmc}``.
    """

    model_config = _STRICT

    num_stage1_draws: StrictInt = Field(default=25, gt=0)
    stage2_draws_per_component: StrictInt | None = Field(default=100, gt=0)
    selection_seed: StrictInt | None = None
    stage2_seed: StrictInt | None = None
    stage2_mcmc: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_no_random_seed_in_stage2_mcmc(self) -> CutConfig:
        if self.stage2_mcmc is not None and "random_seed" in self.stage2_mcmc:
            raise ValueError(
                "cut.stage2_mcmc must not set random_seed; "
                "cut.stage2_seed is the Stage-2 seed authority"
            )
        return self


class Config(BaseModel):
    """Top-level bayesian_panel_nmf configuration."""

    model_config = _STRICT

    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    mcmc: MCMCConfig = Field(default_factory=MCMCConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cut: CutConfig | None = None

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load and validate a YAML config file. Raises ConfigError on failure."""
        try:
            with open(path) as f:
                raw = yaml.safe_load(f)
        except OSError as e:
            raise ConfigError(f"Could not read config file {path!r}: {e}") from e
        return cls.model_validate(raw)

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> Config:  # type: ignore[override]
        try:
            return super().model_validate(obj, *args, **kwargs)
        except ValidationError as e:
            raise ConfigError(str(e)) from e

    def to_legacy_dict(self) -> dict:
        """Return a plain dict matching the legacy raw-YAML config shape."""
        return self.model_dump(by_alias=True, exclude_none=True)
