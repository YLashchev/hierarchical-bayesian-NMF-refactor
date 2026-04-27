import pytest
import jax.numpy as jnp
import jax.random as random
import numpy as np
from numpyro.infer import MCMC, NUTS
from bayesian_panel_nmf.models.panel_nmf_model import model


@pytest.fixture
def tiny_panel():
    """Tiny panel 2 units, 3 groups, 4 times for quick MCMC."""
    K, D, N = 3, 2, 4
    np.random.seed(42)
    denoms = jnp.array(np.random.randint(100, 1000, size=(K, D, N), dtype=int))
    y = jnp.array(np.random.randint(1, 50, size=(K, D, N), dtype=int))
    ctrl_idx = np.ones((K, D, N), dtype=bool)
    missing_idx = np.zeros((K, D, N), dtype=bool)
    return denoms, ctrl_idx, missing_idx, y


@pytest.mark.parametrize("outcome_dist", ["Poisson", "NB"])
@pytest.mark.parametrize("sample_disp", [True, False])
def test_dispersion_guard(tiny_panel, outcome_dist, sample_disp):
    denoms, ctrl_idx, missing_idx, y = tiny_panel

    mcmc = MCMC(
        NUTS(model), num_warmup=2, num_samples=2, num_chains=1, progress_bar=False
    )
    rng_key = random.PRNGKey(0)

    mcmc.run(
        rng_key,
        denominators=denoms,
        control_idx_array=ctrl_idx,
        missing_idx_array=missing_idx,
        y=y,
        rank=2,
        outcome_dist=outcome_dist,
        sample_disp=sample_disp,
        model_treated=False,
        adjust_for_missingness=False,  # speed up test
    )

    samples = mcmc.get_samples()

    if outcome_dist == "Poisson":
        assert "disp" not in samples, "Poisson should not sample or record 'disp'"
    else:
        assert "disp" in samples, "NB should sample or record 'disp'"
