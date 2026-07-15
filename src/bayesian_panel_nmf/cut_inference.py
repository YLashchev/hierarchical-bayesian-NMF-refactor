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

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import random
from loguru import logger
from numpyro.infer import MCMC, NUTS

from .inference import _resolve_model_settings, convergence_summary
from .mcmc_utils import choose_mcmc_parallelism
from .models.cut_stage1_model import stage1_model
from .models.cut_stage2_model import stage2_model
from .validation import ConfigError, DataError, validate_data_dict, validate_rank

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


def _resolve_chains(mcmc_cfg: dict) -> tuple[int, str]:
    """Chain count/method: auto (device-based) or literal config values."""
    if mcmc_cfg.get("auto_parallelism", True):
        return choose_mcmc_parallelism(max_chains=mcmc_cfg.get("max_chains", 4))
    return mcmc_cfg.get("num_chains", 4), mcmc_cfg.get("chain_method", "sequential")


def _extract_fit(mcmc: MCMC) -> MCMCFit:
    """Convert one MCMC run to host arrays; strip scoped low_births/* keys
    (xarray rejects '/' in variable names -- same rule as run_analysis's
    _clean_scoped_samples)."""
    grouped = mcmc.get_samples(group_by_chain=True)
    samples = {k: np.asarray(v) for k, v in grouped.items() if "/" not in k}
    diverging = np.asarray(mcmc.get_extra_fields()["diverging"]).reshape(
        mcmc.num_chains, -1
    )
    num_retained = int(next(iter(samples.values())).shape[1])
    return MCMCFit(
        samples=samples,
        diverging=diverging,
        num_chains=int(mcmc.num_chains),
        num_retained=num_retained,
    )


def run_stage1_mcmc(data_dict: dict, rank: int, config: dict) -> MCMCFit:
    """Fit the untreated baseline to observed untreated cells (Stage 1)."""
    validate_data_dict(data_dict)
    rank = validate_rank(rank)
    mcmc_cfg = config.get("mcmc", {}) or {}
    model_cfg = config.get("model", {}) or {}
    model_settings = _resolve_model_settings(config)
    num_chains, chain_method = _resolve_chains(mcmc_cfg)
    logger.info(f"cut Stage 1: num_chains={num_chains}, chain_method={chain_method!r}")
    mcmc = MCMC(
        NUTS(stage1_model),
        num_warmup=mcmc_cfg.get("num_warmup", 1000),
        num_samples=mcmc_cfg.get("num_samples", 2500),
        num_chains=num_chains,
        thinning=mcmc_cfg.get("thinning", 10),
        progress_bar=mcmc_cfg.get("progress_bar", True),
        chain_method=chain_method,
    )
    _, run_key = random.split(random.PRNGKey(int(mcmc_cfg.get("random_seed", 8675309))))
    mcmc.run(
        run_key,
        extra_fields=("diverging",),
        y=data_dict["Y"],
        denominators=data_dict["denominators"],
        control_idx_array=data_dict["control_idx_array"],
        missing_idx_array=data_dict["missing_idx_array"],
        rank=rank,
        outcome_dist=model_settings["outcome_dist"],
        adjust_for_missingness=model_cfg.get("adjust_for_missingness", True),
        nb_disp=model_settings["nb_disp"],
        sample_disp=model_settings["sample_disp"],
    )
    return _extract_fit(mcmc)


def run_stage2_mcmc(
    data_dict: dict,
    ref: Stage1DrawRef,
    config: dict,
    stage2_mcmc: dict,
    rng_key,
) -> MCMCFit:
    """One complete multi-chain conditional Stage-2 fit for one baseline draw.

    Uses the same ``stage2_model`` callable and constant array shapes for
    every component so XLA can reuse compilation. Never call
    ``jax.clear_caches()`` here.
    """
    model_cfg = config.get("model", {}) or {}
    model_settings = _resolve_model_settings(config)
    num_chains, chain_method = _resolve_chains(stage2_mcmc)
    mcmc = MCMC(
        NUTS(stage2_model),
        num_warmup=stage2_mcmc.get("num_warmup", 1000),
        num_samples=stage2_mcmc.get("num_samples", 2500),
        num_chains=num_chains,
        thinning=stage2_mcmc.get("thinning", 10),
        progress_bar=stage2_mcmc.get("progress_bar", True),
        chain_method=chain_method,
    )
    mcmc.run(
        rng_key,
        extra_fields=("diverging",),
        mu_ctrl=ref.mu_ctrl,
        control_idx_array=data_dict["control_idx_array"],
        missing_idx_array=data_dict["missing_idx_array"],
        y=data_dict["Y"],
        outcome_dist=model_settings["outcome_dist"],
        nb_concentration=ref.nb_concentration,
        adjust_for_missingness=model_cfg.get("adjust_for_missingness", True),
    )
    return _extract_fit(mcmc)


def sample_untreated_predictions(
    mu_ctrl, nb_concentration, outcome_dist: str, rng_key
) -> np.ndarray:
    """Draw untreated posterior-predictive counts for given baseline surfaces.

    ``nb_concentration`` must already be broadcastable against ``mu_ctrl``
    (e.g. ``(D, 1)`` against ``(..., K, D, N)``); ``None`` for Poisson.
    """
    rate = jnp.exp(jnp.asarray(mu_ctrl))
    if outcome_dist == "Poisson":
        draws = dist.Poisson(rate).sample(rng_key)
    else:
        draws = dist.NegativeBinomial2(rate, jnp.asarray(nb_concentration)).sample(
            rng_key
        )
    return np.asarray(draws)


def summarize_mcmc(fit: MCMCFit) -> dict:
    """Per-fit convergence gate via the existing ArviZ-based summary.

    Diagnostics are computed on exactly one real MCMC target -- never across
    pooled cut components.
    """
    import arviz as az

    idata = az.from_dict(
        {"posterior": fit.samples, "sample_stats": {"diverging": fit.diverging}}
    )
    return convergence_summary(idata)
