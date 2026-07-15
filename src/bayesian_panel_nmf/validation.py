"""Config and data validation. Raises ConfigError or DataError with concise messages."""

from pathlib import Path
from typing import Any

import numpy as np


class ConfigError(ValueError):
    """Configuration is invalid."""


class DataError(ValueError):
    """Data validation failed."""


def _require_bool(value: Any, path: str) -> None:
    """
    Raise ConfigError if value is not None and not a true boolean.

    YAML quoted strings like "false" parse as truthy Python strings,
    silently flipping flags. This guard catches them at load time.
    """
    if value is None:
        return
    if not isinstance(value, bool):
        raise ConfigError(
            f"config['{path}'] must be boolean (unquoted true/false in YAML), "
            f"got {type(value).__name__}: {value!r}. "
            f"Check for quotes around the value in your YAML config."
        )


def _validate_types(model_config: dict) -> None:
    types = model_config.get("types")
    if types is not None and not isinstance(types, dict):
        raise ConfigError(
            f"config['model']['types'] must be dict, got {type(types).__name__}"
        )


def _validate_schema_outcomes_list(schema: dict) -> None:
    outcomes = schema["outcomes"]
    if not isinstance(outcomes, list) or len(outcomes) == 0:
        raise ConfigError("schema['outcomes'] must be non-empty list")

    for i, outcome in enumerate(outcomes):
        if not isinstance(outcome, dict):
            raise ConfigError(f"outcomes[{i}] must be dict")
        if "outcome_col" not in outcome or "label" not in outcome:
            raise ConfigError(f"outcomes[{i}] must have 'outcome_col' and 'label'")


def _validate_schema_prefixes(schema: dict) -> None:
    prefix_cfg = schema["outcomes_from_prefixes"]
    if not isinstance(prefix_cfg, dict):
        raise ConfigError("schema['outcomes_from_prefixes'] must be dict")

    outcome_prefix = prefix_cfg.get("outcome_prefix")
    if not isinstance(outcome_prefix, str) or not outcome_prefix:
        raise ConfigError(
            "outcomes_from_prefixes['outcome_prefix'] must be non-empty string"
        )

    denominator_prefix = prefix_cfg.get("denominator_prefix")
    if denominator_prefix is not None and (
        not isinstance(denominator_prefix, str) or not denominator_prefix
    ):
        raise ConfigError(
            "outcomes_from_prefixes['denominator_prefix'] must be non-empty string or null"
        )

    include = prefix_cfg.get("include")
    if include is not None:
        if not isinstance(include, list) or len(include) == 0:
            raise ConfigError(
                "outcomes_from_prefixes['include'] must be non-empty list"
            )
        if not all(isinstance(label, str) and label for label in include):
            raise ConfigError(
                "outcomes_from_prefixes['include'] must contain non-empty strings"
            )


def _validate_schema_outcomes(schema: dict) -> None:
    has_outcomes = schema.get("outcomes") is not None
    has_prefixes = schema.get("outcomes_from_prefixes") is not None

    if has_outcomes == has_prefixes:
        raise ConfigError(
            "schema must define exactly one of 'outcomes' or 'outcomes_from_prefixes'"
        )

    if has_outcomes:
        _validate_schema_outcomes_list(schema)
    else:
        _validate_schema_prefixes(schema)


def _validate_config_boolean_flags(
    config: dict, data_config: dict, model_cfg: dict
) -> None:
    _require_bool(
        data_config.get("allow_unbalanced_panel"), "data.allow_unbalanced_panel"
    )
    agg_cfg = data_config.get("aggregation", {}) or {}
    _require_bool(agg_cfg.get("enabled"), "data.aggregation.enabled")

    for flag in ("sample_disp", "adjust_for_missingness", "model_treated"):
        _require_bool(model_cfg.get(flag), f"model.{flag}")

    types_cfg = model_cfg.get("types", {}) or {}
    for type_name, type_def in types_cfg.items():
        if isinstance(type_def, dict):
            _require_bool(
                type_def.get("total_all"), f"model.types.{type_name}.total_all"
            )

    mcmc_cfg = config.get("mcmc", {}) or {}
    _require_bool(mcmc_cfg.get("progress_bar"), "mcmc.progress_bar")

    output_cfg = config.get("output", {}) or {}
    for flag in (
        "figures",
        "clean",
        "save_traces",
        "print_tables",
        "print_target_table",
    ):
        _require_bool(output_cfg.get(flag), f"output.{flag}")

    aggregate_units = output_cfg.get("aggregate_units")
    if aggregate_units is not None:
        if not isinstance(aggregate_units, list):
            raise ConfigError("config['output.aggregate_units'] must be a list")
        for i, spec in enumerate(aggregate_units):
            if isinstance(spec, dict):
                _require_bool(
                    spec.get("include_treated_units"),
                    f"output.aggregate_units.{i}.include_treated_units",
                )
                _require_bool(
                    spec.get("include_all_units"),
                    f"output.aggregate_units.{i}.include_all_units",
                )
                _require_bool(spec.get("strict"), f"output.aggregate_units.{i}.strict")
                _require_bool(
                    spec.get("overwrite"), f"output.aggregate_units.{i}.overwrite"
                )

    ppc_acf_lags = output_cfg.get("ppc_acf_lags")
    if ppc_acf_lags is not None:
        if not isinstance(ppc_acf_lags, list) or not ppc_acf_lags:
            raise ConfigError("config['output.ppc_acf_lags'] must be a non-empty list")
        if not all(isinstance(lag, int) and lag > 0 for lag in ppc_acf_lags):
            raise ConfigError(
                "config['output.ppc_acf_lags'] must contain positive integers"
            )


def _validate_inference_mode_and_cut(config: dict, model_cfg: dict) -> None:
    """Validate model.inference_mode and the optional cut: section."""
    mode = model_cfg.get("inference_mode")
    if mode is not None and mode not in ("joint", "cut"):
        raise ConfigError(
            f"config['model']['inference_mode'] must be 'joint' or 'cut', got {mode!r}"
        )
    if mode == "cut" and model_cfg.get("model_treated") is False:
        raise ConfigError(
            "model.inference_mode='cut' requires model_treated=true "
            "(the cut model estimates treatment effects)"
        )

    cut_cfg = config.get("cut")
    if cut_cfg is None:
        return
    if not isinstance(cut_cfg, dict):
        raise ConfigError(f"config['cut'] must be dict, got {type(cut_cfg).__name__}")

    for key in ("num_stage1_draws", "stage2_draws_per_component"):
        value = cut_cfg.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ConfigError(
                f"config['cut']['{key}'] must be a positive integer, got {value!r}"
            )
    for key in ("selection_seed", "stage2_seed"):
        value = cut_cfg.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ConfigError(
                f"config['cut']['{key}'] must be an integer, got {value!r}"
            )

    stage2_mcmc = cut_cfg.get("stage2_mcmc")
    if stage2_mcmc is not None:
        if not isinstance(stage2_mcmc, dict):
            raise ConfigError(
                f"config['cut']['stage2_mcmc'] must be dict, "
                f"got {type(stage2_mcmc).__name__}"
            )
        if "random_seed" in stage2_mcmc:
            raise ConfigError(
                "config['cut']['stage2_mcmc'] must not set random_seed; "
                "cut.stage2_seed is the Stage-2 seed authority"
            )


def _validate_data_schema(config: dict) -> tuple[dict, dict]:
    if "data" not in config:
        raise ConfigError("config missing 'data' section")

    data_config = config["data"]
    if not isinstance(data_config, dict):
        raise ConfigError(
            f"config['data'] must be dict, got {type(data_config).__name__}"
        )

    schema = data_config.get("schema")
    if not schema:
        raise ConfigError("config['data'] missing 'schema' section")

    if not isinstance(schema, dict):
        raise ConfigError(
            f"config['data']['schema'] must be dict, got {type(schema).__name__}"
        )

    required = ["unit_col", "time_col", "treatment_col"]
    missing = [k for k in required if k not in schema]
    if missing:
        raise ConfigError(f"schema missing: {missing}")

    return data_config, schema


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ConfigError(f"config must be dict, got {type(config).__name__}")

    data_config, schema = _validate_data_schema(config)

    model_config = config.get("model")
    if model_config is not None:
        if not isinstance(model_config, dict):
            raise ConfigError(
                f"config['model'] must be dict, got {type(model_config).__name__}"
            )

        _validate_types(model_config)

    _validate_schema_outcomes(schema)
    _validate_config_boolean_flags(config, data_config, config.get("model", {}) or {})
    _validate_inference_mode_and_cut(config, config.get("model", {}) or {})


def validate_filepath(filepath: str) -> Path:
    """Return resolved Path if file exists. Raises DataError otherwise."""
    if not isinstance(filepath, (str, Path)):
        raise DataError(f"filepath must be str or Path, got {type(filepath).__name__}")

    path = Path(filepath)
    if not path.exists():
        raise DataError(f"File not found: {filepath}")

    return path


def validate_groups(groups: list[str]) -> None:
    """Require groups to be a non-empty list of strings."""
    if not groups or not isinstance(groups, list):
        raise DataError("groups must be non-empty list")

    if not all(isinstance(g, str) for g in groups):
        raise DataError("groups must contain strings")


def _check_arrays_consistency(
    data_dict: dict, shape: tuple, required_arrays: list
) -> None:
    for key in required_arrays:
        arr = data_dict[key]
        if not isinstance(arr, np.ndarray):
            raise DataError(f"{key} must be numpy array, got {type(arr).__name__}")
        if arr.shape != shape:
            raise DataError(f"{key} shape {arr.shape} != Y shape {shape}")


def validate_data_dict(data_dict: dict) -> None:
    """
    Validate data dictionary structure and shapes.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing model data with keys:
        Y, denominators, control_idx_array, missing_idx_array, groups, units, times

    Raises
    ------
    DataError
        If required keys are missing or shapes are inconsistent
    """
    if not isinstance(data_dict, dict):
        raise DataError(f"data_dict must be dict, got {type(data_dict).__name__}")

    required_arrays = ["Y", "denominators", "control_idx_array", "missing_idx_array"]
    required_meta = ["groups", "units", "times"]

    missing = [k for k in required_arrays + required_meta if k not in data_dict]
    if missing:
        raise DataError(f"data_dict missing: {missing}")

    # Check Y is 3D
    shape = data_dict["Y"].shape
    if len(shape) != 3:
        raise DataError(f"Arrays must be 3D (K,D,N), got shape {shape}")

    _check_arrays_consistency(data_dict, shape, required_arrays)

    K, D, N = shape
    if len(data_dict["groups"]) != K:
        raise DataError(f"len(groups)={len(data_dict['groups'])} != K={K}")
    if len(data_dict["units"]) != D:
        raise DataError(f"len(units)={len(data_dict['units'])} != D={D}")
    if len(data_dict["times"]) != N:
        raise DataError(f"len(times)={len(data_dict['times'])} != N={N}")


def validate_rank(rank: Any) -> int:
    """Return `rank` as int if positive; raise DataError otherwise."""
    if not isinstance(rank, (int, np.integer)) or rank <= 0:
        raise DataError(f"rank must be positive integer, got {rank}")
    return int(rank)


def validate_samples(samples: dict[str, np.ndarray]) -> None:
    """
    Validate MCMC samples dictionary.

    Parameters
    ----------
    samples : dict
        MCMC samples from mcmc.get_samples(group_by_chain=True)
        Must contain 'mu_ctrl' key with 5D array (C, S, K, D, N)

    Raises
    ------
    DataError
        If samples are invalid
    """
    if not isinstance(samples, dict):
        raise DataError(f"samples must be dict, got {type(samples).__name__}")

    if "mu_ctrl" not in samples:
        raise DataError("samples missing 'mu_ctrl' key")

    mu_ctrl = samples["mu_ctrl"]
    # Check for array-like with ndim attribute (works for numpy and jax arrays)
    if not hasattr(mu_ctrl, "ndim") or mu_ctrl.ndim != 5:
        raise DataError(
            f"samples['mu_ctrl'] must be 5D array (C,S,K,D,N), got shape {getattr(mu_ctrl, 'shape', 'N/A')}"
        )


def validate_predictions(
    predictions: np.ndarray, samples: dict[str, Any] | None = None
) -> None:
    """
    Validate predictions array.

    Parameters
    ----------
    predictions : np.ndarray
        Posterior predictive samples, shape (C, S, K, D, N)
    samples : dict, optional
        If provided, validates predictions shape matches samples['mu_ctrl']

    Raises
    ------
    DataError
        If predictions have invalid shape
    """
    # Check for array-like with ndim attribute (works for numpy and jax arrays)
    if not hasattr(predictions, "ndim"):
        raise DataError(
            f"predictions must be array-like, got {type(predictions).__name__}"
        )

    if predictions.ndim != 5:
        raise DataError(
            f"predictions must be 5D (C,S,K,D,N), got shape {predictions.shape}"
        )

    if (
        samples is not None
        and "mu_ctrl" in samples
        and predictions.shape != samples["mu_ctrl"].shape
    ):
        raise DataError(
            f"predictions shape {predictions.shape} != "
            f"samples['mu_ctrl'] shape {samples['mu_ctrl'].shape}"
        )
