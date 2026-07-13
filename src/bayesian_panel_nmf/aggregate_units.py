"""Synthetic aggregate reporting units for posterior draws.

Aggregate units are appended to a copy of the draws DataFrame for
reporting and PPC only. They are never fed back into model training.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
import pandas as pd

from bayesian_panel_nmf.validation import ConfigError

_INCLUDE_SELECTORS = (
    "include_treated_units",
    "include_all_units",
    "include_units",
)


def _as_bool(value: Any, path: str) -> bool:
    """Return bool config value, rejecting truthy strings."""
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be boolean, got {type(value).__name__}")
    return value


def _as_str_list(value: Any, path: str, *, allow_empty: bool = False) -> list[str]:
    """Return list[str], raising ConfigError for malformed selectors."""
    if not isinstance(value, list):
        raise ConfigError(f"{path} must be a list of strings")
    if not allow_empty and len(value) == 0:
        raise ConfigError(f"{path} must be a non-empty list")
    if not all(isinstance(v, str) and v for v in value):
        raise ConfigError(f"{path} must contain non-empty strings")
    return value


def _active_selectors(spec: dict) -> list[str]:
    """Return include selectors active in one aggregate spec."""
    active: list[str] = []
    if _as_bool(
        spec.get("include_treated_units"), "aggregate_units.include_treated_units"
    ):
        active.append("include_treated_units")
    if _as_bool(spec.get("include_all_units"), "aggregate_units.include_all_units"):
        active.append("include_all_units")
    if spec.get("include_units") is not None:
        _as_str_list(spec["include_units"], "aggregate_units.include_units")
        active.append("include_units")
    return active


def _validate_aggregate_spec(spec: dict, index: int, existing_units: set[str]) -> None:
    """Validate a single aggregate unit spec."""
    if not isinstance(spec, dict):
        raise ConfigError(f"aggregate_units[{index}] must be dict")

    unit_name = spec.get("unit")
    if not unit_name or not isinstance(unit_name, str):
        raise ConfigError(
            f"aggregate_units[{index}]['unit'] must be a non-empty string"
        )

    active = _active_selectors(spec)
    if len(active) != 1:
        raise ConfigError(
            f"aggregate_units[{index}] ('{unit_name}') must have exactly one "
            f"include selector; got {active or 'none'}"
        )

    if spec.get("exclude_units") is not None:
        _as_str_list(
            spec["exclude_units"], "aggregate_units.exclude_units", allow_empty=True
        )

    strict = _as_bool(spec.get("strict"), "aggregate_units.strict")
    overwrite = _as_bool(spec.get("overwrite"), "aggregate_units.overwrite")
    if strict not in (True, False) or overwrite not in (True, False):
        raise ConfigError("strict/overwrite must be boolean")

    if unit_name in existing_units and not overwrite:
        raise ConfigError(
            f"aggregate unit '{unit_name}' collides with an existing unit. "
            f"Set overwrite: true to allow replacement."
        )


def _source_units_for_spec(spec: dict, source_df: pd.DataFrame) -> list[str]:
    """Return source unit names selected by one spec."""
    all_units = set(source_df["unit"])

    if spec.get("include_all_units"):
        sources = list(all_units)
        missing: list[str] = []
    elif spec.get("include_treated_units"):
        missing = []
        if "treatment" not in source_df.columns:
            sources = []
        else:
            treated_mask = source_df["treatment"] == 1
            sources = source_df.loc[treated_mask, "unit"].unique().tolist()
    else:
        requested = list(spec.get("include_units") or [])
        missing = [u for u in requested if u not in all_units]
        sources = [u for u in requested if u in all_units]

    exclude = spec.get("exclude_units") or []
    sources = [u for u in sources if u not in exclude]
    if missing and spec.get("strict", False):
        raise ConfigError(
            f"aggregate unit '{spec['unit']}' references missing units: {missing}"
        )
    if missing:
        warnings.warn(
            f"aggregate unit '{spec['unit']}' skips missing units: {missing}",
            stacklevel=3,
        )
        sources = [u for u in sources if u in all_units]

    if not sources:
        warnings.warn(
            f"aggregate unit '{spec['unit']}' resolved to empty source set; skipping",
            stacklevel=3,
        )

    return sources


def _logsumexp_series(x: pd.Series) -> float:
    """Compute log(sum(exp(x))) for one grouped column."""
    values = x.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan")
    max_value = np.max(values)
    return float(max_value + np.log(np.sum(np.exp(values - max_value))))


def _aggregate_one(
    source_df: pd.DataFrame,
    unit_name: str,
    sources: list[str],
) -> pd.DataFrame:
    """Aggregate draws for one synthetic unit."""
    sub = source_df[source_df["unit"].isin(sources)].copy()
    if sub.empty:
        return pd.DataFrame()

    group_cols = [".draw", ".chain", ".iteration", "time", "group"]
    group_cols = [c for c in group_cols if c in sub.columns]

    grouped = sub.groupby(group_cols, observed=True)
    pieces: list[pd.DataFrame] = []

    sum_cols = [c for c in ("outcome", "ypred", "denominator") if c in sub.columns]
    if sum_cols:
        pieces.append(cast(pd.DataFrame, grouped[sum_cols].sum()))
    if "treatment" in sub.columns:
        treatment_max = grouped[["treatment"]].max()
        pieces.append(cast(pd.DataFrame, treatment_max))
    for col in ("mu", "mu_treated"):
        if col in sub.columns:
            logsum = cast(pd.Series, grouped[col].apply(_logsumexp_series))
            pieces.append(cast(pd.DataFrame, logsum.to_frame(name=col)))

    if not pieces:
        return pd.DataFrame()

    agg = pd.concat(pieces, axis=1).reset_index()
    agg["unit"] = unit_name

    # Preserve original column order where possible.
    ordered_cols = [c for c in source_df.columns if c in agg.columns]
    extra_cols = [c for c in agg.columns if c not in ordered_cols]
    ordered = agg.reindex(columns=ordered_cols + extra_cols).copy()
    return cast(pd.DataFrame, ordered)


def add_aggregate_units(draws_df: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    """Append synthetic reporting-only aggregate units to a copy of ``draws_df``.

    Aggregates are computed from the original input DataFrame, not from
    aggregates created earlier in the same call. This prevents double-counting
    when multiple aggregate specs are configured.
    """
    if not specs:
        return draws_df.copy()

    source_df = draws_df.copy()
    result_df = draws_df.copy()
    existing_units = set(result_df["unit"])
    aggregate_frames: dict[str, pd.DataFrame] = {}

    for index, spec in enumerate(specs):
        _validate_aggregate_spec(spec, index, existing_units)
        unit_name = spec["unit"]

        if spec.get("overwrite", False):
            filtered_result = result_df.loc[result_df["unit"] != unit_name].copy()
            result_df = cast(pd.DataFrame, filtered_result)
            aggregate_frames.pop(unit_name, None)
            existing_units.discard(unit_name)

        sources = _source_units_for_spec(spec, source_df)
        if not sources:
            continue

        aggregate = _aggregate_one(source_df, unit_name, sources)
        if aggregate.empty:
            continue

        aggregate_frames[unit_name] = aggregate
        existing_units.add(unit_name)

    if aggregate_frames:
        frames: list[pd.DataFrame] = [cast(pd.DataFrame, result_df)]
        frames.extend(aggregate_frames.values())
        result_df = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))

    return cast(pd.DataFrame, result_df)
