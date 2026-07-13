"""Convergence gate: rank-normalized R-hat / bulk+tail ESS via ArviZ."""

import arviz as az
import numpy as np
from loguru import logger

from bayesian_panel_nmf.inference import convergence_summary


def _idata(chains: np.ndarray, diverging: np.ndarray | None = None):
    # Installed ArviZ (1.1.0) uses a nested-dict from_dict(data) signature
    # (data-as-DataTree), not the older from_dict(posterior=..., sample_stats=...)
    # kwarg form. Adapted here to match the installed API; convergence_summary's
    # contract (az.summary(kind="diagnostics"), idata.sample_stats) is unaffected.
    data = {"posterior": {"theta": chains}}
    if diverging is not None:
        data["sample_stats"] = {"diverging": diverging}
    return az.from_dict(data)


def test_well_mixed_chains_converge():
    rng = np.random.default_rng(0)
    idata = _idata(rng.normal(size=(4, 400)))  # iid -> rhat~1, ESS ~1600
    s = convergence_summary(idata)
    assert s["converged"] is True
    assert s["rhat_max"] < 1.01
    assert s["ess_bulk_min"] > 400
    assert s["divergences"] == 0


def test_separated_chains_fail_rhat():
    rng = np.random.default_rng(0)
    chains = rng.normal(size=(4, 400))
    chains[0] += 10.0  # one chain in a different mode
    s = convergence_summary(_idata(chains))
    assert s["converged"] is False
    assert s["rhat_max"] > 1.01


def test_divergences_fail_gate():
    rng = np.random.default_rng(0)
    diverging = np.zeros((4, 400), dtype=bool)
    diverging[1, 5] = True
    s = convergence_summary(_idata(rng.normal(size=(4, 400)), diverging))
    assert s["divergences"] == 1
    assert s["converged"] is False


# --- Identifiable-variable filtering -----------------------------------------
# The gate assesses ONLY the identifiable deterministic quantities
# (mu, mu_ctrl, te) and ignores the non-identifiable latent factors
# (time_fac, unit_weight, state_fe, time_fe), whose R-hat is meaningless
# under rotation/permutation invariance of the low-rank factorization.


def _idata_with_vars(vars_dict: dict, diverging=None):
    data = {"posterior": vars_dict}
    if diverging is not None:
        data["sample_stats"] = {"diverging": diverging}
    return az.from_dict(data)


def test_convergence_passes_when_identifiable_ok_but_latent_unmixed():
    """A badly-mixed latent factor must NOT fail the gate when mu/mu_ctrl/te
    are well mixed."""
    rng = np.random.default_rng(0)
    good = rng.normal(size=(4, 400))
    bad_latent = rng.normal(size=(4, 400))
    bad_latent[0] += 5.0  # one chain in a different mode -> rhat >> 1
    idata = _idata_with_vars(
        {"mu": good, "mu_ctrl": good, "time_fac": bad_latent, "unit_weight": bad_latent}
    )
    s = convergence_summary(idata)
    assert s["converged"] is True
    assert s["rhat_max"] < 1.01


def test_convergence_fails_when_identifiable_var_unmixed():
    """A badly-mixed mu must fail the gate even if latents are well mixed."""
    rng = np.random.default_rng(1)
    good = rng.normal(size=(4, 400))
    bad_mu = rng.normal(size=(4, 400))
    bad_mu[1] += 7.0  # one chain off -> rhat >> 1
    idata = _idata_with_vars(
        {"mu": bad_mu, "mu_ctrl": good, "time_fac": good, "unit_weight": good}
    )
    s = convergence_summary(idata)
    assert s["converged"] is False
    assert s["rhat_max"] > 1.01


def test_convergence_fails_when_te_unmixed():
    """te is identifiable when present (model_treated); a badly-mixed te fails."""
    rng = np.random.default_rng(2)
    good = rng.normal(size=(4, 400))
    bad_te = rng.normal(size=(4, 400))
    bad_te[2] += 6.0
    idata = _idata_with_vars(
        {"mu": good, "mu_ctrl": good, "te": bad_te, "state_fe": good}
    )
    s = convergence_summary(idata)
    assert s["converged"] is False
    assert s["rhat_max"] > 1.01


def test_convergence_falls_back_when_no_identifiable_vars_present():
    """If none of (mu, mu_ctrl, te) are in the posterior, fall back to
    summarizing all variables and emit a warning."""
    rng = np.random.default_rng(3)
    chains = rng.normal(size=(4, 400))
    chains[0] += 8.0  # badly mixed -> should fail via fallback
    idata = _idata_with_vars({"theta": chains})

    warnings: list[str] = []
    sink_id = logger.add(warnings.append, level="WARNING")
    try:
        s = convergence_summary(idata)
    finally:
        logger.remove(sink_id)

    assert s["converged"] is False  # theta was assessed via fallback
    assert any("none of" in w and "falling back" in w for w in warnings)
