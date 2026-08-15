"""Stage 2 of the two-stage pure cut posterior: conditional treatment model.

Samples only the treatment-effect hierarchy (identical priors/site names to
the joint model's treatment block), conditioned on one fixed selected Stage-1
``mu_ctrl`` draw and its matched NegativeBinomial2 concentration. Baseline
quantities are ordinary numeric arguments -- there is no sample site for them
and therefore no feedback path from exposed outcomes into the baseline.

Direct likelihood on exposed nonmissing cells; when
``adjust_for_missingness``, censored exposed counts are integrated over 1..9.
"""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import scope

from .likelihood import missingness_adjustment


def stage2_model(
    mu_ctrl,
    control_idx_array,
    missing_idx_array=None,
    y=None,
    outcome_dist="NB",
    nb_concentration=None,
    adjust_for_missingness=True,
):
    """Treatment-effect model conditioned on a fixed Stage-1 baseline draw.

    Parameters
    ----------
    mu_ctrl : array (K, D, N)
        One selected Stage-1 posterior draw of the untreated log-rate
        surface (already includes ``log(denominators)``). Held fixed.
    control_idx_array : bool array (K, D, N)
        True = untreated cell. Required; the likelihood targets the exposed
        (``~control``) cells.
    nb_concentration : array (D,), required when ``outcome_dist == "NB"``
        Per-unit NegativeBinomial2 concentration matched to the same Stage-1
        draw (``1/disp`` draw when dispersion was sampled, ``1/nb_disp``
        when fixed). Poisson runs pass ``None``.
    """
    if control_idx_array is None:
        raise ValueError(
            "stage2_model requires control_idx_array to identify exposed cells."
        )
    if outcome_dist == "NB" and nb_concentration is None:
        raise ValueError("outcome_dist='NB' requires nb_concentration.")

    K, D, N = control_idx_array.shape
    control_idx = control_idx_array.reshape(-1)
    exposed_idx = ~control_idx
    num_treated = int((~control_idx_array).sum())

    if missing_idx_array is None:
        missing_idx = np.zeros_like(control_idx_array, dtype=np.bool_).reshape(-1)
    else:
        missing_idx = missing_idx_array.reshape(-1)

    f_all = numpyro.deterministic("mu_ctrl", jnp.asarray(mu_ctrl))

    treatment_it_scale = numpyro.sample(
        "treatment_it_scale", dist.HalfNormal(scale=0.1)
    )
    treatment_state_scale = numpyro.sample(
        "treatment_state_scale", dist.HalfNormal(scale=1)
    )
    treatment_category_scale = numpyro.sample(
        "treatment_category_scale", dist.HalfNormal(scale=1)
    )
    state_category_scale = numpyro.sample(
        "state_category_scale", dist.HalfNormal(scale=1)
    )

    # Non-centered: treatment_kt/state_treatment_effect/state_category_te each
    # sample a standard-normal z, then scale deterministically. Their priors
    # are tight/weakly-identified with sparse exposed-cell data, and centered
    # form gives a funnel (divergences, ESS < 30) -- same fix as joint.py.
    # category_treatment_effect stays centered: its scale is data-informed
    # (ESS ~1000, PASS), and centered mixes better there.
    with numpyro.plate("num_treated", num_treated):
        treatment_kt_z = numpyro.sample("treatment_kt_z", dist.Normal(0, 1))
    treatment_kt = numpyro.deterministic(
        "treatment_kt", treatment_kt_z * treatment_it_scale
    )
    with numpyro.plate("num_states", D):
        state_treatment_effect_z = numpyro.sample(
            "state_treatment_effect_z", dist.Normal(0, 1)
        )
        with numpyro.plate("num_cats", K):
            state_category_te_z = numpyro.sample(
                "state_category_te_z", dist.Normal(0, 1)
            )
    state_treatment_effect = numpyro.deterministic(
        "state_treatment_effect", state_treatment_effect_z * treatment_state_scale
    )
    state_category_te = numpyro.deterministic(
        "state_category_te", state_category_te_z * state_category_scale
    )
    with numpyro.plate("num_cats", K):
        category_treatment_effect = numpyro.sample(
            "category_treatment_effect", dist.Normal(scale=treatment_category_scale)
        )

    te = numpyro.deterministic(
        "te",
        jnp.zeros_like(control_idx_array, dtype=float)
        .at[~control_idx_array]
        .add(treatment_kt)
        + (
            (~control_idx_array) * state_treatment_effect[None, :, None]
            + (~control_idx_array) * category_treatment_effect[:, None, None]
            + (~control_idx_array) * state_category_te[:, :, None]
        ),
    )
    mu = numpyro.deterministic("mu", f_all + te)

    if outcome_dist == "NB":
        conc_broadcast = jnp.asarray(nb_concentration)[None, :, None] * jnp.ones_like(
            mu
        )
    else:
        conc_broadcast = None

    if adjust_for_missingness:
        scope(missingness_adjustment, "suppressed_counts")(
            mu,
            missing_idx,
            exposed_idx,
            jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            outcome_dist,
            dispersion=conc_broadcast,
        )

    if y is not None:
        mask = ~missing_idx & exposed_idx
        y_obs = y.reshape(-1)[mask]
    else:
        mask = exposed_idx
        y_obs = None

    f = mu.reshape(-1)[mask]

    if outcome_dist == "Poisson":
        numpyro.sample("y_obs", dist.Poisson(rate=jnp.exp(f)), obs=y_obs)
    else:
        conc_obs = conc_broadcast.reshape(-1)[mask]
        numpyro.sample("y_obs", dist.NegativeBinomial2(jnp.exp(f), conc_obs), obs=y_obs)
