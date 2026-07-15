"""Two-stage pure cut-posterior inference (eta = 0).

Orchestration primitives: settings resolution, chain-stratified Stage-1 draw
selection, Stage-1/Stage-2 MCMC runners, untreated posterior predictions,
deterministic per-component output subsampling, and per-fit convergence
summaries. No function here touches the filesystem, and no function retains
or returns NumPyro MCMC objects -- each fit is converted to host NumPy arrays
and diagnostics immediately.

The cut target is p_cut(phi, theta | Z, Y) = p(phi | Z) * p(theta | Y, phi):
exposed outcomes never feed back into the baseline. Stage 2 conditions on a
fixed selected ``mu_ctrl`` surface plus matched NB concentration, which is a
sufficient boundary because the exposed-cell likelihood depends on Stage-1
parameters only through those quantities.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from .validation import ConfigError, DataError

DEFAULT_NUM_STAGE1_DRAWS = 25
DEFAULT_STAGE2_DRAWS_PER_COMPONENT = 100
SELECTION_SEED_OFFSET = 2
STAGE2_SEED_OFFSET = 3


@dataclass(frozen=True)
class CutSettings:
    """Resolved cut-mode configuration (defaults applied)."""

    num_stage1_draws: int
    selection_seed: int
    stage2_seed: int
    stage2_draws_per_component: int | None
    stage2_mcmc: dict


@dataclass(frozen=True)
class Stage1DrawRef:
    """One selected Stage-1 baseline draw with provenance.

    ``stage1_iteration`` is the retained (post-thinning) 1-based index within
    its chain; ``stage1_draw`` is the global retained index
    ``(chain - 1) * retained_per_chain + iteration``.
    """

    component: int
    stage1_draw: int
    stage1_chain: int
    stage1_iteration: int
    mu_ctrl: np.ndarray
    nb_concentration: np.ndarray | None


@dataclass(frozen=True)
class MCMCFit:
    """Host-side extraction of one MCMC run (never the MCMC object itself)."""

    samples: dict
    diverging: np.ndarray
    num_chains: int
    num_retained: int


def resolve_cut_settings(config: dict) -> CutSettings:
    """Overlay cut defaults; derive seeds; warn on stream collisions."""
    mcmc_cfg = config.get("mcmc", {}) or {}
    base_seed = int(mcmc_cfg.get("random_seed", 8675309))
    cut_cfg = config.get("cut", {}) or {}

    selection_seed = cut_cfg.get("selection_seed")
    selection_seed = (
        base_seed + SELECTION_SEED_OFFSET
        if selection_seed is None
        else int(selection_seed)
    )
    stage2_seed = cut_cfg.get("stage2_seed")
    stage2_seed = (
        base_seed + STAGE2_SEED_OFFSET if stage2_seed is None else int(stage2_seed)
    )

    per_component = cut_cfg.get(
        "stage2_draws_per_component", DEFAULT_STAGE2_DRAWS_PER_COMPONENT
    )
    if per_component is not None:
        per_component = int(per_component)

    stage2_mcmc = {**mcmc_cfg, **(cut_cfg.get("stage2_mcmc", {}) or {})}
    # cut.stage2_seed is the only Stage-2 seed authority (validation rejects
    # an explicit stage2_mcmc.random_seed; the inherited base value is unused).
    stage2_mcmc.pop("random_seed", None)

    claimed = {
        base_seed: "mcmc.random_seed",
        base_seed + 1: "the Stage-1 PPC stream (mcmc.random_seed + 1)",
    }
    for name, seed_value in (
        ("cut.selection_seed", selection_seed),
        ("cut.stage2_seed", stage2_seed),
    ):
        if seed_value in claimed:
            logger.warning(
                f"{name}={seed_value} collides with {claimed[seed_value]}; "
                "RNG streams should be distinct"
            )
        claimed[seed_value] = name

    return CutSettings(
        num_stage1_draws=int(cut_cfg.get("num_stage1_draws", DEFAULT_NUM_STAGE1_DRAWS)),
        selection_seed=selection_seed,
        stage2_seed=stage2_seed,
        stage2_draws_per_component=per_component,
        stage2_mcmc=stage2_mcmc,
    )


def validate_cut_data(data_dict: dict) -> None:
    """Fail fast when either stage's likelihood subset is empty."""
    control = data_dict["control_idx_array"]
    missing = data_dict["missing_idx_array"]
    if int((control & ~missing).sum()) == 0:
        raise DataError("cut mode: no observed untreated cells for Stage 1")
    if int((~control & ~missing).sum()) == 0:
        raise DataError("cut mode: no observed exposed nonmissing cells for Stage 2")


def _chain_quotas(total: int, n_chains: int) -> list[int]:
    """Split ``total`` across chains as evenly as possible (first chains get
    the remainder; chains are exchangeable, so this is documented behavior)."""
    base, extra = divmod(total, n_chains)
    return [base + (1 if c < extra else 0) for c in range(n_chains)]


def select_stage1_draws(
    stage1_samples: dict, settings: CutSettings, model_config: dict
) -> list[Stage1DrawRef]:
    """Seeded chain-stratified selection without replacement.

    Components are numbered 1..M after sorting by (chain, iteration), so
    numbering is reproducible and independent of RNG draw order.
    """
    mu_ctrl = stage1_samples["mu_ctrl"]  # (C, S, K, D, N)
    n_chains, n_retained = int(mu_ctrl.shape[0]), int(mu_ctrl.shape[1])
    m = settings.num_stage1_draws
    if m > n_chains * n_retained:
        raise ConfigError(
            f"cut.num_stage1_draws={m} exceeds retained Stage-1 draws "
            f"({n_chains * n_retained})"
        )

    outcome_dist = model_config.get("outcome_distribution", "NB")
    sample_disp = model_config.get("sample_disp", False)
    nb_disp = model_config.get("nb_disp", 1e-4)
    disp = stage1_samples.get("disp") if sample_disp else None
    n_units = int(mu_ctrl.shape[3])

    rng = np.random.default_rng(settings.selection_seed)
    picks: list[tuple[int, int]] = []
    for chain, quota in enumerate(_chain_quotas(m, n_chains)):
        if quota == 0:
            continue
        chosen = np.sort(rng.choice(n_retained, size=quota, replace=False))
        picks.extend((chain, int(i)) for i in chosen)
    picks.sort()

    refs: list[Stage1DrawRef] = []
    for component, (chain, idx) in enumerate(picks, start=1):
        if outcome_dist == "NB":
            if sample_disp and disp is not None:
                concentration = np.array(1.0 / disp[chain, idx], dtype=float)
            else:
                concentration = np.ones(n_units) / nb_disp
        else:
            concentration = None
        refs.append(
            Stage1DrawRef(
                component=component,
                stage1_draw=chain * n_retained + idx + 1,
                stage1_chain=chain + 1,
                stage1_iteration=idx + 1,
                mu_ctrl=np.array(mu_ctrl[chain, idx], dtype=float),
                nb_concentration=concentration,
            )
        )
    return refs


def subsample_component_draws(
    num_chains: int, retained_per_chain: int, per_component: int | None
) -> list[np.ndarray]:
    """Deterministic evenly-strided per-chain output subsample (no RNG).

    Diagnostics always use the full retained draws; this only thins what
    enters the combined output CSV. ``None`` keeps everything.
    """
    if per_component is None:
        return [np.arange(retained_per_chain) for _ in range(num_chains)]
    indices: list[np.ndarray] = []
    for quota in _chain_quotas(per_component, num_chains):
        if quota > retained_per_chain:
            raise DataError(
                f"cut.stage2_draws_per_component={per_component} requires {quota} "
                f"draws from a chain with only {retained_per_chain} retained; "
                "reduce it or increase stage2_mcmc samples"
            )
        if quota == 0:
            indices.append(np.empty(0, dtype=int))
            continue
        indices.append(
            np.linspace(0, retained_per_chain - 1, quota).round().astype(int)
        )
    return indices
