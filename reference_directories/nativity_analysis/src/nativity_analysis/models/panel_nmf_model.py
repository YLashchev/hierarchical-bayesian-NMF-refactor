from jax import numpy as jnp
import numpy as np
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
import jax.numpy as jnp
import numpyro
from numpyro.handlers import scope

from .utils import missingness_adjustment


def model(
        denominators, 
        control_idx_array, 
        missing_idx_array,
        y=None,
        rank=5,
        outcome_dist="NB",
        adjust_for_missingness=True,
        nb_disp = 1e-4,
        sample_disp = False,
        model_treated =False
    ):
    # if enforce_joint_consistency and (y_totals is None):
    #     raise Exception("Totals must be passed in for joint consistency.")
    
    # treated time period onward
    K, D, N = denominators.shape
    
    # Set the masking arrays to just be a vacuous "True" if not set
    if(control_idx_array is None):
        control_idx = np.ones_like(denominators, dtype=np.bool_).reshape(-1)
        num_treated = 0
    else:
        control_idx = control_idx_array.reshape(-1)
        num_treated = (~control_idx_array).sum()

    if(missing_idx_array is None):
        missing_idx = np.ones_like(denominators, dtype=np.bool_).reshape(-1)
    else:
        missing_idx = missing_idx_array.reshape(-1)

    

    time_fac_alpha = 20
    with numpyro.plate('K', K):
        with numpyro.plate('F', rank):
            with numpyro.plate('N', N):
                raw_time_factor = jnp.log(numpyro.sample('time_fac', 
                        dist.Gamma(time_fac_alpha, time_fac_alpha)
                        ))
        with numpyro.plate('D', D):
            state_fe = jnp.log(numpyro.sample('state_fe', dist.ImproperUniform(constraints.positive, (), ()))).T   #dist.Gamma(0.5, 0.01)).T) 
        
        with numpyro.plate('N', N):
            time_fe = jnp.log(numpyro.sample('time_fe',
                                dist.Gamma(1, 1)
                                ).T
                             )
        with numpyro.plate('D', D):
            unit_weights = jnp.log(numpyro.sample('unit_weight', dist.Dirichlet(jnp.ones(rank))))
    
    time_factor = jnp.log(jnp.exp(raw_time_factor.transpose(2,0,1)[:, None, :, :] + unit_weights.transpose(1, 0, 2)[:, :, None, :]).sum(-1))

    # create fixed effects, accounting for dimensions of each and broadcasting apropriately
    fixed_effects = (
        state_fe[:, :, None] 
        + time_fe[:, None, :]
    )

    f_all = numpyro.deterministic(
        "mu_ctrl",
        time_factor + 
        fixed_effects + 
        # we want births per 10k
        jnp.log(denominators) #.sum(0)[None, ...]) #+ 
    )

    if model_treated:
        treatment_it_scale = numpyro.sample('treatment_it_scale', dist.HalfNormal(scale=0.1))
        treatment_state_scale = numpyro.sample('treatment_state_scale', dist.HalfNormal(scale=1))
        treatment_category_scale = numpyro.sample('treatment_category_scale', dist.HalfNormal(scale=1))
        state_category_scale = numpyro.sample('state_category_scale', dist.HalfNormal(scale=1))

        with numpyro.plate('num_treated', num_treated):
            treatment_kt = numpyro.sample('treatment_kt', dist.Normal(scale=treatment_it_scale))
        with numpyro.plate('num_states', D):
            state_treatment_effect = numpyro.sample('state_treatment_effect', dist.Normal(scale=treatment_state_scale))
            with numpyro.plate('num_cats', K):
                state_category_te = numpyro.sample('state_category_te', dist.Normal(scale=state_category_scale))
        with numpyro.plate('num_cats', K):
            category_treatment_effect = numpyro.sample('category_treatment_effect', dist.Normal(scale=treatment_category_scale))
            

        #treatment_kt_tensor = numpyro.deterministic('tkt', jnp.zeros_like(control_idx_array))
        #treatment_kt_tensor[~control_idx_array] = treatment_kt
        #print(treatment_kt_tensor.shape)
        te = numpyro.deterministic('te', jnp.zeros_like(control_idx_array, dtype=float).at[~control_idx_array].add(treatment_kt) + ((~control_idx_array) * state_treatment_effect[None, :, None] + (~control_idx_array) * category_treatment_effect[:, None, None] + (~control_idx_array) * state_category_te[:, :, None]))
        mu = numpyro.deterministic('mu', f_all + te)
        #mu = numpyro.deterministic('mu', f_all.at[~control_idx_array].add(treatment_kt) + (~control_idx_array) * state_treatment_effect[None, :, None] + (~control_idx_array) * category_treatment_effect[:, None, None])
    else:
        mu = numpyro.deterministic('mu', f_all)
    
    num_obs = K * D * N
    
    if sample_disp:
        lam = 100
        with numpyro.plate('num_states', D):        
            nb_disp = numpyro.sample("disp", dist.Uniform())
        numpyro.factor('nb_disp_log_prob', -1.0/2.0*jnp.log(nb_disp) - lam*jnp.sqrt(nb_disp))
        dispersion = 1/nb_disp #if outcome_dist == "NB" else None
    else:
        nb_disp = numpyro.deterministic("disp", nb_disp)
        dispersion = jnp.ones(D)/nb_disp #if outcome_dist == "NB" else None
    
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
            scope(missingness_adjustment, "low_births")(
                mu, 
                missing_idx, 
                control_idx if not model_treated else np.ones_like(control_idx, dtype=np.bool_), 
                jnp.array([1,2,3,4,5,6,7,8,9]), 
                outcome_dist, 
                dispersion=(dispersion[None, :, None] * jnp.ones_like(mu))
            )
        # subset to nonmissing observations that are unmasked
        f = (mu.reshape(-1))[mask]
        y_obs = y.reshape(-1)[mask]
        disp_obs = (dispersion[None, :, None] * jnp.ones_like(mu)).reshape(-1)[mask]
        # if we are attempting to enforce a total level consistency 
        # we total the subgroups and take the likelihood w.r.t. the sum
        
    else:
        y_obs = None
        f = mu.reshape(-1)
        disp_obs = (dispersion[None, :, None] * jnp.ones_like(mu)).reshape(-1)
    
    # Step 0: Rerun poisson with the updated priors. 
    # Step 1: compare nb_disp = very small and explicit Poisson parameterization
    # Step 2: change nb_disp manually
    # Step 3: run with PC prior (think about how to set sampling, Uniform?)
    # Step 4: Fix joint consistency issue with proportional parameterization
    if outcome_dist == "Poisson":
        obs = numpyro.sample(
            'y_obs',
            dist.Poisson(rate=jnp.exp(f)),
            obs=y_obs
        )
    else:
        # Dispersion is concentration = alpha of gamma
        # e^f = \alpha / \beta 
        # \lambda = Gamma(a, b)

        obs =  numpyro.sample(
            'y_obs',
            dist.NegativeBinomial2(jnp.exp(f), disp_obs), 
            obs=y_obs,
        )
