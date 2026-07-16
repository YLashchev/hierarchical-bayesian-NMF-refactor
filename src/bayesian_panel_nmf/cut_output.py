"""Formatting for two-stage cut-posterior outputs. Transforms only -- all
file I/O stays in scripts/run_analysis.py, matching the output.py boundary.

The combined cut CSV keeps the established reporting schema (built by the
EXISTING ``output.format_draws`` one component at a time) plus nested
provenance columns. ``.draw`` is globally unique across components;
``.chain`` is the real Stage-2 chain within the component; ``.iteration``
indexes the component's output subsample (order-preserving -- original
retained ordering lives in the optional trace NetCDFs). Components are never
stacked into a fake chain axis, and convergence is never computed from these
frames.
"""

import numpy as np
import pandas as pd

from .cut import Stage1DrawRef
from .output import format_draws

PROVENANCE_COLUMNS = [
    "cut_component",
    "stage1_draw",
    "stage1_chain",
    "stage1_iteration",
]


def format_stage1_ppc_draws(mu_ctrl, ypred, data_dict: dict) -> pd.DataFrame:
    """Full Stage-1 posterior product (all retained draws, no ``te``).

    This frame feeds ONLY the JAMA PPC suite; effect summaries come from the
    combined cut draws.
    """
    return format_draws({"mu_ctrl": np.asarray(mu_ctrl)}, np.asarray(ypred), data_dict)


def format_cut_component_draws(
    te_draws,
    ypred,
    chain_ids,
    ref: Stage1DrawRef,
    data_dict: dict,
    draw_offset: int,
) -> pd.DataFrame:
    """Format one conditional Stage-2 component into the reporting schema.

    Parameters
    ----------
    te_draws, ypred : arrays (n_out, K, D, N)
        Output-subsampled treatment effects and independently generated
        untreated predictive counts, draws concatenated in ascending
        Stage-2-chain order.
    chain_ids : array (n_out,)
        1-based real Stage-2 chain of each draw, grouped ascending.
    draw_offset : int
        Global ``.draw`` offset (sum of output draws of prior components).
    """
    te_draws = np.asarray(te_draws)
    ypred = np.asarray(ypred)
    chain_ids = np.asarray(chain_ids)
    n_out = te_draws.shape[0]
    cell_count = int(np.prod(te_draws.shape[1:]))
    mu_grid = np.broadcast_to(ref.mu_ctrl, te_draws.shape)

    df = format_draws(
        {"mu_ctrl": mu_grid[None], "te": te_draws[None]},
        ypred[None],
        data_dict,
    )

    # format_draws saw a (1, n_out) grid; restore real per-draw coordinates.
    # Rows are draw-major: each draw occupies cell_count consecutive rows.
    _, counts = np.unique(chain_ids, return_counts=True)
    iter_ids = np.concatenate([np.arange(1, c + 1) for c in counts])
    df[".chain"] = np.repeat(chain_ids.astype(np.int8), cell_count)
    df[".iteration"] = np.repeat(iter_ids.astype(np.int32), cell_count)
    df[".draw"] = (df[".draw"] + draw_offset).astype(np.int32)

    df["cut_component"] = np.int16(ref.component)
    df["stage1_draw"] = np.int32(ref.stage1_draw)
    df["stage1_chain"] = np.int8(ref.stage1_chain)
    df["stage1_iteration"] = np.int32(ref.stage1_iteration)
    assert n_out == 0 or len(df) == n_out * cell_count
    return df


def build_cut_convergence_manifest(
    stage1_diag: dict, component_records: list[dict]
) -> dict:
    """One manifest: Stage-1 gate plus every conditional fit's gate.

    Top-level ``converged`` is true only when Stage 1 AND every component
    passed. Aggregate counts are provided for convenience, but R-hat/ESS are
    never computed across pooled conditional targets.
    """
    all_converged = all(bool(r["converged"]) for r in component_records)
    return {
        "inference_mode": "cut",
        "converged": bool(stage1_diag["converged"]) and all_converged,
        "stage1": stage1_diag,
        "stage2": {
            "all_converged": all_converged,
            "failed_fits": int(sum(not r["converged"] for r in component_records)),
            "fits": component_records,
        },
    }
