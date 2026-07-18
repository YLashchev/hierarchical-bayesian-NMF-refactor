"""Synthetic aggregate reporting units for posterior draws.

Aggregate units are appended to a copy of the draws DataFrame for
reporting and PPC only. They are never fed back into model training.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
from pydantic import ValidationError

from bayesian_panel_nmf.config import AggregateUnitSpec
from bayesian_panel_nmf.validation import ConfigError


def _source_units_for_spec(
    spec: AggregateUnitSpec, source_df: pd.DataFrame
) -> list[str]:
    """Return source unit names selected by one spec."""
    all_units = set(source_df["unit"])

    if spec.include_all_units:
        sources = list(all_units)
        missing: list[str] = []
    elif spec.include_treated_units:
        missing = []
        if "treatment" not in source_df.columns:
            sources = []
        else:
            treated_mask = source_df["treatment"] == 1
            sources = source_df.loc[treated_mask, "unit"].unique().tolist()
    else:
        requested = list(spec.include_units or [])
        missing = [u for u in requested if u not in all_units]
        sources = [u for u in requested if u in all_units]

    exclude = spec.exclude_units or []
    sources = [u for u in sources if u not in exclude]
    if missing and spec.strict:
        raise ConfigError(
            f"aggregate unit '{spec.unit}' references missing units: {missing}"
        )
    if missing:
        warnings.warn(
            f"aggregate unit '{spec.unit}' skips missing units: {missing}",
            stacklevel=3,
        )
        sources = [u for u in sources if u in all_units]

    if not sources:
        warnings.warn(
            f"aggregate unit '{spec.unit}' resolved to empty source set; skipping",
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
    # Period boundaries are constant across the pooled units within each
    # (time) group; carry them onto the aggregate so reporting can compute
    # person-years. Without this they'd be re-added as all-NaN by the reindex
    # below, making years -> NaN and rates -> inf/nan.
    date_cols = [c for c in ("start_date", "end_date") if c in sub.columns]
    if date_cols:
        pieces.append(cast(pd.DataFrame, grouped[date_cols].first()))
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


def add_aggregate_units(
    draws_df: pd.DataFrame, specs: Sequence[AggregateUnitSpec | dict]
) -> pd.DataFrame:
    """Append synthetic reporting-only aggregate units to a copy of ``draws_df``.

    Aggregates are computed from the original input DataFrame, not from
    aggregates created earlier in the same call. This prevents double-counting
    when multiple aggregate specs are configured.
    """
    if not specs:
        return draws_df.copy()

    try:
        typed_specs = [
            spec
            if isinstance(spec, AggregateUnitSpec)
            else AggregateUnitSpec.model_validate(spec)
            for spec in specs
        ]
    except ValidationError as e:
        raise ConfigError(str(e)) from e

    # draws_df is only read (never mutated); avoid copying it. On large cut
    # PPC frames (millions of rows) the old double-copy (source_df +
    # result_df) tripled peak memory and drove the reporting OOM. result_df
    # starts as the input and is only copied lazily when an overwrite spec
    # filters it; the final concat allocates the combined frame once.
    source_df = draws_df
    result_df = draws_df
    existing_units = set(draws_df["unit"])
    aggregate_frames: dict[str, pd.DataFrame] = {}

    for spec in typed_specs:
        unit_name = spec.unit

        if unit_name in existing_units and not spec.overwrite:
            raise ConfigError(
                f"aggregate unit '{unit_name}' collides with an existing unit. "
                f"Set overwrite: true to allow replacement."
            )

        if spec.overwrite:
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
        # concat allocates a fresh combined frame (input not mutated).
        return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))

    # No aggregate matched: result_df may still alias the input, so copy to
    # keep the "returns an independent frame" contract.
    if result_df is draws_df:
        return draws_df.copy()
    return cast(pd.DataFrame, result_df)
