import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from jax import numpy as jnp
from numpyro.handlers import scope

from .likelihood import missingness_adjustment


def _define_time_factors_and_fe(K, D, rank, N, time_fac_alpha):
    """Sample the low-rank time factors and state/time fixed effects.

    Parameters
    ----------
    K, D, N : int
        Group, unit, and time-period dimensions.
    rank : int
        Number of latent factors for the low-rank time/unit decomposition.
    time_fac_alpha : float
        Shape/rate parameter for the Gamma prior on raw time factors.

    Returns
    -------
    state_fe : jnp.ndarray
        Log-space state fixed effects, shape (D, K).
    time_fe : jnp.ndarray
        Log-space time fixed effects, shape (N, K).
    time_factor : jnp.ndarray
        Log-space combined low-rank time/unit factor surface.
    """
    with numpyro.plate("K", K):
        with numpyro.plate("F", rank), numpyro.plate("N", N):
            raw_time_factor = jnp.log(
                numpyro.sample("time_fac", dist.Gamma(time_fac_alpha, time_fac_alpha))
            )
        # Non-centered hierarchical Normal prior on state fixed effects
        # (log-rate scale). A flat prior pulls units with zero-count cells
        # to -inf; partial pooling via a shared mean/scale + z-scores fixes
        # that and samples cleanly.
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


def model(
    denominators,
    control_idx_array,
    missing_idx_array,
    y=None,
    rank=5,
    outcome_dist="NB",
    adjust_for_missingness=True,
    nb_disp=1e-4,
    sample_disp=False,
    model_treated=False,
):
    """NumPyro model: low-rank Bayesian panel NMF with optional treatment effects.

    Decomposes the log-rate surface into a low-rank state/time factor
    structure plus state and time fixed effects, then adds an optional
    treatment-effect term (state x category x time interactions) when
    ``model_treated=True``. The outcome is Poisson or Negative Binomial,
    with an optional adjustment for small suppressed/censored counts.

    Parameters
    ----------
    denominators : array, shape (K, D, N)
        Population/exposure denominators per group, unit, and time period.
    control_idx_array : array of bool, shape (K, D, N), or None
        True where a cell is in the untreated (control) period/unit.
        Required when ``model_treated=True``; may be None for pure
        counterfactual prediction with ``model_treated=False``.
    missing_idx_array : array of bool, shape (K, D, N), or None
        True where a cell's observed count is known to be suppressed/
        censored to a small range (see ``adjust_for_missingness``).
    y : array, shape (K, D, N), optional
        Observed outcome counts. None for prior/posterior predictive
        sampling (no likelihood conditioning).
    rank : int
        Number of latent factors for the low-rank time/unit decomposition.
    outcome_dist : {"NB", "Poisson"}
        Outcome likelihood family.
    adjust_for_missingness : bool
        Whether to add the censored-count log-likelihood adjustment via
        ``missingness_adjustment`` for small suppressed counts.
    nb_disp : float
        Fixed Negative Binomial dispersion, used when ``sample_disp`` is
        False and ``outcome_dist == "NB"``.
    sample_disp : bool
        Whether to sample NB dispersion per unit instead of using the
        fixed ``nb_disp`` value. Ignored for ``outcome_dist == "Poisson"``.
    model_treated : bool
        Whether to add the treatment-effect term. Requires
        ``control_idx_array`` when True.

    Raises
    ------
    ValueError
        If ``model_treated=True`` but ``control_idx_array`` is None.
    """
    if model_treated and control_idx_array is None:
        raise ValueError(
            "model_treated=True requires control_idx_array to identify treated "
            "observations. Pass control_idx_array=None only with model_treated=False "
            "(counterfactual prediction)."
        )

    K, D, N = denominators.shape

    # Vacuous "True" mask when not set.
    if control_idx_array is None:
        control_idx = np.ones_like(denominators, dtype=np.bool_).reshape(-1)
        num_treated = 0
    else:
        control_idx = control_idx_array.reshape(-1)
        num_treated = (~control_idx_array).sum()

    if missing_idx_array is None:
        missing_idx = np.zeros_like(denominators, dtype=np.bool_).reshape(-1)
    else:
        missing_idx = missing_idx_array.reshape(-1)

    time_fac_alpha = 20
    state_fe, time_fe, time_factor = _define_time_factors_and_fe(
        K, D, rank, N, time_fac_alpha
    )

    # state_fe broadcasts over time (K, D, 1); time_fe broadcasts over units (K, 1, N).
    fixed_effects = state_fe[:, :, None] + time_fe[:, None, :]

    f_all = numpyro.deterministic(
        "mu_ctrl",
        time_factor
        + fixed_effects
        # log(denominators) converts the log-rate surface to a log-count rate.
        + jnp.log(denominators),
    )

    if model_treated:
        # Help static analysis know control_idx_array is not None in this path
        assert control_idx_array is not None

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

        # Non-centered: treatment_kt/state_treatment_effect/state_category_te
        # each sample a standard-normal z inside the same plate, then scale
        # deterministically. Their scale priors are tight/weakly-identified
        # (HalfNormal(0.1) / HalfNormal(1) with sparse exposed-cell data),
        # which centered parameterization turns into a funnel (divergences,
        # ESS < 30) -- same fix as state_fe above. category_treatment_effect
        # stays centered: its scale is data-informed (ESS ~1000, PASS), and
        # centered mixes better when the scale is well constrained.
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
    else:
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
        if model_treated:
            mask = ~missing_idx
        else:
            mask = ~missing_idx & control_idx

        if adjust_for_missingness:
            # small nonzero counts are censored in the source data; integrate over them.
            disp_broadcast = (
                dispersion[None, :, None] * jnp.ones_like(mu)
                if dispersion is not None
                else None
            )
            scope(missingness_adjustment, "suppressed_counts")(
                mu,
                missing_idx,
                control_idx
                if not model_treated
                else np.ones_like(control_idx, dtype=np.bool_),
                jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                outcome_dist,
                dispersion=disp_broadcast,
            )
        # subset to nonmissing observations that are unmasked
        f = (mu.reshape(-1))[mask]
        y_obs = y.reshape(-1)[mask]

    else:
        y_obs = None
        f = mu.reshape(-1)

    # Likelihood — must run for both train (y is not None) and predict (y is None)
    if outcome_dist == "Poisson":
        numpyro.sample("y_obs", dist.Poisson(rate=jnp.exp(f)), obs=y_obs)
    else:
        # NegativeBinomial2 dispersion is the gamma-mixture concentration.
        disp_broadcast = dispersion[None, :, None] * jnp.ones_like(mu)
        if y is not None:
            mask_ = ~missing_idx if model_treated else (~missing_idx & control_idx)
            disp_obs = disp_broadcast.reshape(-1)[mask_]
        else:
            disp_obs = disp_broadcast.reshape(-1)

        numpyro.sample(
            "y_obs",
            dist.NegativeBinomial2(jnp.exp(f), disp_obs),
            obs=y_obs,
        )
