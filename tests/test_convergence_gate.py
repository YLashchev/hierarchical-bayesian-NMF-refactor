"""Convergence gate: rank-normalized R-hat / bulk+tail ESS via ArviZ."""

import arviz as az
import numpy as np

from bayesian_panel_nmf.diagnostics import convergence_summary


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
