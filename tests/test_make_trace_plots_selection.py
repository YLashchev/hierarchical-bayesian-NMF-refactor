"""Characterization tests for cli.py's variable/coordinate selection logic
(formerly scripts/make_trace_plots.py, folded into `bpnmf traces --plots`
in Phase 9.2). Pins current behavior before any extraction refactor."""

import arviz as az
import numpy as np

from bayesian_panel_nmf.cli import _make_trace_plots_from_netcdf


def _build_fake_idata(var_shapes: dict[str, tuple[int, ...]]):
    """Build a minimal InferenceData (xarray.DataTree, per installed ArviZ
    1.1.0's own migration off the deprecated InferenceData constructor) with
    given posterior variable shapes. First two dims of every array are
    always (chain, draw). Uses az.from_dict, the same construction idiom
    bpnmf traces already uses for real trace sidecars."""
    rng = np.random.default_rng(0)
    n_chain, n_draw = 2, 50
    posterior = {
        name: rng.normal(size=(n_chain, n_draw, *extra_dims))
        for name, extra_dims in var_shapes.items()
    }
    return az.from_dict({"posterior": posterior})


def test_scalar_priority_vars_selected_first_when_present(tmp_path):
    """When no --param-filter given, scalar_priority vars come before
    matrix_params vars, matching current hardcoded ordering."""
    idata = _build_fake_idata({"disp": (), "te": (3,), "time_fac": (4,)})
    nc_path = tmp_path / "fake_traces.nc"
    idata.to_netcdf(str(nc_path))

    out_dir = tmp_path / "out"
    saved = _make_trace_plots_from_netcdf(nc_path, param_filters=None, out_dir=out_dir)

    saved_names = [p.stem.removeprefix("trace_") for p in saved]
    assert saved_names.index("disp") < saved_names.index("te")
    assert saved_names.index("disp") < saved_names.index("time_fac")


def test_param_filter_selects_matching_prefixes_only(tmp_path):
    idata = _build_fake_idata({"disp": (), "te": (3,), "time_fac": (4,)})
    nc_path = tmp_path / "fake_traces.nc"
    idata.to_netcdf(str(nc_path))

    out_dir = tmp_path / "out"
    saved = _make_trace_plots_from_netcdf(
        nc_path, param_filters=["te"], out_dir=out_dir
    )

    saved_names = {p.stem.removeprefix("trace_") for p in saved}
    assert saved_names == {"te"}


def test_structurally_zero_variable_is_skipped(tmp_path):
    """A variable with zero variance across chain+draw for every cell must
    be skipped entirely (current flatline-skip behavior)."""
    idata = az.from_dict({"posterior": {"zero_var": np.zeros((2, 50, 3))}})
    nc_path = tmp_path / "fake_traces.nc"
    idata.to_netcdf(str(nc_path))

    out_dir = tmp_path / "out"
    saved = _make_trace_plots_from_netcdf(
        nc_path, param_filters=["zero_var"], out_dir=out_dir
    )

    assert saved == []


def test_high_dim_variable_subsamples_deterministically(tmp_path):
    """With a fixed seed (42, hardcoded in the function), coordinate
    subsampling for a >20-cell variable must be reproducible across runs."""
    idata = _build_fake_idata({"big_param": (10, 10)})
    nc_path = tmp_path / "fake_traces.nc"
    idata.to_netcdf(str(nc_path))

    out_dir_a = tmp_path / "out_a"
    out_dir_b = tmp_path / "out_b"
    saved_a = _make_trace_plots_from_netcdf(
        nc_path, param_filters=["big_param"], out_dir=out_dir_a
    )
    saved_b = _make_trace_plots_from_netcdf(
        nc_path, param_filters=["big_param"], out_dir=out_dir_b
    )

    assert len(saved_a) == 1
    assert len(saved_b) == 1
    # Same seed, same input -> byte-identical output image.
    assert saved_a[0].read_bytes() == saved_b[0].read_bytes()
