"""Stage 1 of the two-stage pure cut posterior: independent untreated baseline.

Scientifically identical to the joint model's ``model_treated=False`` branch
in ``models/joint.py``. The duplication is deliberate (approved
independence decision): do NOT import private helpers from
``joint.py``. Parity with the joint model is enforced by
``tests/test_cut_model_parity.py``.
"""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro.handlers import scope

from .likelihood import missingness_adjustment


def _define_time_factors_and_fe(K, D, rank, N, time_fac_alpha):
    """Sample the low-rank time factors and state/time fixed effects.

    Independent copy of the joint model's factor construction; identical
    plate names, priors, and log-space assembly.
    """
    with numpyro.plate("K", K):
        with numpyro.plate("F", rank), numpyro.plate("N", N):
            raw_time_factor = jnp.log(
                numpyro.sample("time_fac", dist.Gamma(time_fac_alpha, time_fac_alpha))
            )
        # Hierarchical, non-centered Normal prior on state fixed effects
        # (real/log-rate scale). Replaces the old flat ImproperUniform on the
        # positive scale, which pulled state effects to -inf for units with
        # zero-count cells. Partial pooling via a shared mean/scale + z-scores
        # regularizes those units; non-centered form also samples cleanly.
        # Kept identical to joint.py (Stage-1/Stage-2 parity).
        state_fe_mu = numpyro.sample(
            "state_fe_mu", dist.ImproperUniform(constraints.real, (), ())
        )
        state_fe_sigma = numpyro.sample("state_fe_sigma", dist.HalfNormal(0.5))
        with numpyro.plate("D", D):
            state_fe_z = numpyro.sample("state_fe_z", dist.Normal(0, 1))
        state_fe = (state_fe_mu + state_fe_sigma * state_fe_z).T

        with numpyro.plate("N", N):
            time_fe_sample = numpyro.sample("time_fe", dist.Gamma(1, 1))
            time_fe = jnp.log(time_fe_sample).T
        with numpyro.plate("D", D):
            unit_weights = jnp.log(
                numpyro.sample("unit_weight", dist.Dirichlet(jnp.ones(rank)))
            )

    time_factor = jnp.log(
        jnp.exp(
            raw_time_factor.transpose(2, 0, 1)[:, None, :, :]
            + unit_weights.transpose(1, 0, 2)[:, :, None, :]
        ).sum(-1)
    )
    return state_fe, time_fe, time_factor


def stage1_model(
    denominators,
    control_idx_array,
    missing_idx_array,
    y=None,
    rank=5,
    outcome_dist="NB",
    adjust_for_missingness=True,
    nb_disp=1e-4,
    sample_disp=False,
):
    """Untreated low-rank baseline fit to observed untreated cells only.

    Direct likelihood on ``control & ~missing`` cells; when
    ``adjust_for_missingness``, censored untreated counts (``control &
    missing``) are integrated over 1..9. ``control_idx_array=None`` means a
    vacuous all-True mask and is reserved for unconditional predictive
    sampling (mirrors the joint model's convention).
    """
    K, D, N = denominators.shape

    if control_idx_array is None:
        control_idx = np.ones_like(denominators, dtype=np.bool_).reshape(-1)
    else:
        control_idx = control_idx_array.reshape(-1)

    if missing_idx_array is None:
        missing_idx = np.zeros_like(denominators, dtype=np.bool_).reshape(-1)
    else:
        missing_idx = missing_idx_array.reshape(-1)

    time_fac_alpha = 20
    state_fe, time_fe, time_factor = _define_time_factors_and_fe(
        K, D, rank, N, time_fac_alpha
    )

    fixed_effects = state_fe[:, :, None] + time_fe[:, None, :]

    f_all = numpyro.deterministic(
        "mu_ctrl", time_factor + fixed_effects + jnp.log(denominators)
    )
    mu = numpyro.deterministic("mu", f_all)

    if outcome_dist == "NB":
        if sample_disp:
            lam = 100
            with numpyro.plate("num_states", D):
                nb_disp = numpyro.sample("disp", dist.Uniform())
            numpyro.factor(
                "nb_disp_log_prob",
                -1.0 / 2.0 * jnp.log(nb_disp) - lam * jnp.sqrt(nb_disp),
            )
            dispersion = 1 / nb_disp
        else:
            nb_disp = numpyro.deterministic("disp", nb_disp)
            dispersion = jnp.ones(D) / nb_disp
    else:
        dispersion = None

    if y is not None:
        mask = ~missing_idx & control_idx
        if adjust_for_missingness:
            disp_broadcast = (
                dispersion[None, :, None] * jnp.ones_like(mu)
                if dispersion is not None
                else None
            )
            scope(missingness_adjustment, "suppressed_counts")(
                mu,
                missing_idx,
                control_idx,
                jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                outcome_dist,
                dispersion=disp_broadcast,
            )
        f = (mu.reshape(-1))[mask]
        y_obs = y.reshape(-1)[mask]
    else:
        y_obs = None
        f = mu.reshape(-1)

    if outcome_dist == "Poisson":
        numpyro.sample("y_obs", dist.Poisson(rate=jnp.exp(f)), obs=y_obs)
    else:
        disp_broadcast = dispersion[None, :, None] * jnp.ones_like(mu)
        if y is not None:
            disp_obs = disp_broadcast.reshape(-1)[mask]
        else:
            disp_obs = disp_broadcast.reshape(-1)
        numpyro.sample(
            "y_obs",
            dist.NegativeBinomial2(jnp.exp(f), disp_obs),
            obs=y_obs,
        )
