"""Tiny-MCMC checks for the cut inference runners (seconds, seeded)."""

import jax
import numpy as np
import pandas as pd

from bayesian_panel_nmf.cut_inference import (
    resolve_cut_settings,
    run_stage1_mcmc,
    run_stage2_mcmc,
    sample_untreated_predictions,
    select_stage1_draws,
    summarize_mcmc,
)

K, D, N, RANK = 2, 3, 6, 2


def _data_dict():
    rng = np.random.default_rng(0)
    Y = rng.poisson(50, size=(K, D, N)).astype(float)
    control = np.ones((K, D, N), dtype=bool)
    control[:, 0, 3:] = False
    missing = np.zeros((K, D, N), dtype=bool)
    groups = ["g0", "g1"]
    units = ["u0", "u1", "u2"]
    times = list(pd.date_range("2020-01-01", periods=N, freq="MS"))
    rows = []
    for k, g in enumerate(groups):
        for d_i, u in enumerate(units):
            for n_i, t in enumerate(times):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "group": g,
                        "outcome": Y[k, d_i, n_i],
                        "denominator": 2.0,
                        "treatment": int(not control[k, d_i, n_i]),
                    }
                )
    return {
        "Y": Y,
        "denominators": np.full((K, D, N), 2.0),
        "control_idx_array": control,
        "missing_idx_array": missing,
        "groups": groups,
        "units": units,
        "times": times,
        "df_preprocessed": pd.DataFrame(rows),
    }


CONFIG = {
    "model": {
        "outcome_distribution": "NB",
        "nb_disp": 1e-4,
        "sample_disp": False,
        "adjust_for_missingness": True,
        "model_treated": True,
        "inference_mode": "cut",
    },
    "mcmc": {
        "auto_parallelism": False,
        "num_chains": 2,
        "chain_method": "sequential",
        "num_warmup": 15,
        "num_samples": 16,
        "thinning": 1,
        "random_seed": 0,
        "progress_bar": False,
    },
    "cut": {"num_stage1_draws": 2, "stage2_draws_per_component": 4},
}


def test_stage1_and_stage2_runners_end_to_end():
    dd = _data_dict()
    fit1 = run_stage1_mcmc(dd, RANK, CONFIG)
    assert fit1.num_chains == 2
    assert fit1.num_retained == 16
    assert fit1.samples["mu_ctrl"].shape == (2, 16, K, D, N)
    assert not any("/" in k for k in fit1.samples)
    assert fit1.diverging.shape == (2, 16)

    diag = summarize_mcmc(fit1)
    assert set(diag) >= {
        "rhat_max",
        "ess_bulk_min",
        "ess_tail_min",
        "divergences",
        "converged",
    }

    settings = resolve_cut_settings(CONFIG)
    refs = select_stage1_draws(fit1.samples, settings, CONFIG["model"])
    assert len(refs) == 2

    fit2 = run_stage2_mcmc(
        dd, refs[0], CONFIG, settings.stage2_mcmc, jax.random.PRNGKey(5)
    )
    assert fit2.samples["te"].shape == (2, 16, K, D, N)
    # cut invariant: te is zero on unexposed cells for every draw
    te = fit2.samples["te"]
    control = dd["control_idx_array"]
    assert np.all(te[:, :, control] == 0.0)
    summarize_mcmc(fit2)


def test_sample_untreated_predictions_shapes_and_determinism():
    mu = np.zeros((2, 3, K, D, N))
    conc = (np.ones(D) / 1e-4)[:, None]
    a = sample_untreated_predictions(mu, conc, "NB", jax.random.PRNGKey(1))
    b = sample_untreated_predictions(mu, conc, "NB", jax.random.PRNGKey(1))
    c = sample_untreated_predictions(mu, conc, "NB", jax.random.PRNGKey(2))
    assert a.shape == mu.shape
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    p = sample_untreated_predictions(mu, None, "Poisson", jax.random.PRNGKey(1))
    assert p.shape == mu.shape
