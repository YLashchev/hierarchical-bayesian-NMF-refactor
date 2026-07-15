"""Shared synthetic data for cut-mode tests (not collected by pytest)."""

import numpy as np
import pandas as pd

K, D, N, RANK = 2, 3, 6, 2


def make_cut_data_dict() -> dict:
    """Minimal (K, D, N) cut-mode data_dict: unit u0 treated from period 3."""
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
