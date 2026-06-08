from jax import numpy as jnp
import numpy as np
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
import numpyro
from numpyro.handlers import scope

from .utils import missingness_adjustment


def _define_time_factors_and_fe(K, D, rank, N, time_fac_alpha):
    with numpyro.plate("K", K):
        with numpyro.plate("F", rank):
            with numpyro.plate("N", N):
                raw_time_factor = jnp.log(
                    numpyro.sample(
                        "time_fac", dist.Gamma(time_fac_alpha, time_fac_alpha)
                    )
                )
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


def _define_treatment_effects(control_idx_array):
    """Hierarchical treatment-effect block.

    Returns the ``te`` tensor of shape (K, D, N), non-zero only on treated
    cells. Depends only on the control/treated mask (not on the factor model),
    so it is shared between the joint ``model`` and the Stage-2
    ``treatment_effect_model``.
    """
    K, D, N = control_idx_array.shape
    num_treated = (~control_idx_array).sum()

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

    with numpyro.plate("num_treated", num_treated):
        treatment_kt = numpyro.sample(
            "treatment_kt", dist.Normal(scale=treatment_it_scale)
        )
    with numpyro.plate("num_states", D):
        state_treatment_effect = numpyro.sample(
            "state_treatment_effect", dist.Normal(scale=treatment_state_scale)
        )
        with numpyro.plate("num_cats", K):
            state_category_te = numpyro.sample(
                "state_category_te", dist.Normal(scale=state_category_scale)
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
    return te


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
    # if enforce_joint_consistency and (y_totals is None):
    #     raise Exception("Totals must be passed in for joint consistency.")

    if model_treated and control_idx_array is None:
        raise ValueError(
            "model_treated=True requires control_idx_array to identify treated "
            "observations. Pass control_idx_array=None only with model_treated=False "
            "(counterfactual prediction)."
        )

    # treated time period onward
    K, D, N = denominators.shape

    # Set the masking arrays to just be a vacuous "True" if not set
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

    # create fixed effects, accounting for dimensions of each and broadcasting apropriately
    fixed_effects = state_fe[:, :, None] + time_fe[:, None, :]

    f_all = numpyro.deterministic(
        "mu_ctrl",
        time_factor
        + fixed_effects
        +
        # we want births per 10k
        jnp.log(denominators),  # .sum(0)[None, ...]) #+
    )

    if model_treated:
        # Help static analysis know control_idx_array is not None in this path
        assert control_idx_array is not None
        te = _define_treatment_effects(control_idx_array)
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

    # Do low-rank approximation of the proportion model so:
    # \alpha_i, \alpha_k ~ \theta_i \theta_j where theta i,j are low rank factors drawn over time
    # priors for theta (Beta distributions?)
    # then use \alpha as concentration parameters in the Dirichlet
    # mu ~ treatment_effect + observed
    # each category is ~ mu * Dirichlet(\alpha_it, .. \alpha_ik)
    if y is not None:
        if model_treated:
            mask = ~missing_idx
        else:
            mask = ~missing_idx & control_idx

        if adjust_for_missingness:
            # adjust for the fact that low and nonzero births are masked from the dataset
            disp_broadcast = (
                dispersion[None, :, None] * jnp.ones_like(mu)
                if dispersion is not None
                else None
            )
            scope(missingness_adjustment, "low_births")(
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
        # Dispersion is concentration = alpha of gamma
        # e^f = \alpha / \beta
        # \lambda = Gamma(a, b)

        # Broadcast and mask dispersion for NegBin likelihood
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


def treatment_effect_model(
    mu_ctrl_offset,
    control_idx_array,
    missing_idx_array=None,
    y=None,
    outcome_dist="NB",
    dispersion=None,
    adjust_for_missingness=False,
):
    """Stage 2 of the two-stage ("cut") model.

    Estimates only the hierarchical treatment-effect block, fit to the
    post-treatment treated cells, with a single Stage-1 posterior draw of the
    counterfactual log-rate (``mu_ctrl_offset``) plugged in as a fixed offset.
    ``mu_ctrl_offset`` already includes ``log(denominators)``, so it is used
    directly. No factor model is sampled, so post-treatment data cannot feed
    back into the counterfactual.

    Parameters
    ----------
    mu_ctrl_offset : array, shape (K, D, N)
        One posterior draw of ``mu_ctrl`` from Stage 1, held fixed.
    control_idx_array : bool array, shape (K, D, N)
        True for control cells; the likelihood is evaluated on the treated
        (``~control_idx_array``) post-treatment cells only.
    dispersion : array, shape (D,), optional
        NegativeBinomial2 concentration per unit (required for ``outcome_dist='NB'``).
        Typically carried from the matched Stage-1 draw or derived from a fixed
        ``nb_disp``.
    """
    if control_idx_array is None:
        raise ValueError(
            "treatment_effect_model requires control_idx_array to identify "
            "treated cells."
        )

    control_idx = control_idx_array.reshape(-1)
    treated_idx = ~control_idx

    if missing_idx_array is None:
        missing_idx = np.zeros_like(control_idx_array, dtype=np.bool_).reshape(-1)
    else:
        missing_idx = missing_idx_array.reshape(-1)

    f_all = numpyro.deterministic("mu_ctrl", jnp.asarray(mu_ctrl_offset))
    te = _define_treatment_effects(control_idx_array)
    mu = numpyro.deterministic("mu", f_all + te)

    if outcome_dist == "NB":
        if dispersion is None:
            raise ValueError("outcome_dist='NB' requires a dispersion argument.")
        disp_broadcast = jnp.asarray(dispersion)[None, :, None] * jnp.ones_like(mu)
    else:
        disp_broadcast = None

    # Stage 2 fits the treated post-treatment cells only.
    if adjust_for_missingness:
        scope(missingness_adjustment, "low_births")(
            mu,
            missing_idx,
            treated_idx,
            jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            outcome_dist,
            dispersion=disp_broadcast,
        )

    if y is not None:
        mask = ~missing_idx & treated_idx
        y_obs = y.reshape(-1)[mask]
    else:
        mask = treated_idx
        y_obs = None

    f = mu.reshape(-1)[mask]

    if outcome_dist == "Poisson":
        numpyro.sample("y_obs", dist.Poisson(rate=jnp.exp(f)), obs=y_obs)
    else:
        disp_obs = disp_broadcast.reshape(-1)[mask]
        numpyro.sample(
            "y_obs",
            dist.NegativeBinomial2(jnp.exp(f), disp_obs),
            obs=y_obs,
        )
