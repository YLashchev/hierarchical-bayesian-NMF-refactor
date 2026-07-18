"""Traces picker, rich diagnostics table, fixed-site detection, stage2 dirs."""

from pathlib import Path

import arviz as az
import numpy as np
import pytest

from bayesian_panel_nmf import cli
from bayesian_panel_nmf.diagnostics import parameter_diagnostics
from bayesian_panel_nmf.validation import ConfigError


def _write_nc(path: Path, *, fixed_disp: bool, seed: int = 0) -> Path:
    """Sidecar with a healthy param, an unmixed one, and optionally a constant."""
    rng = np.random.default_rng(seed)
    post = {
        "mu": rng.normal(size=(2, 300)),
        "time_fac": np.stack(
            [rng.normal(0, 1, 300), rng.normal(9, 1, 300)]
        ),  # unmixed -> FAIL
    }
    if fixed_disp:
        post["disp"] = np.full((2, 300), 1e-4)  # constant -> fixed
    az.from_dict({"posterior": post}).to_netcdf(str(path), engine="h5netcdf")
    return path


# ---------------------------------------------------------------------------
# diagnostics rows: empirical fixed detection + status
# ---------------------------------------------------------------------------


def test_rows_flag_constant_site_as_fixed(tmp_path):
    nc = _write_nc(tmp_path / "t.nc", fixed_disp=True)
    rows = parameter_diagnostics(az.from_netcdf(nc))
    by_name = {r["param"]: r for r in rows}
    assert by_name["disp"]["status"] == "fixed"
    assert by_name["disp"]["rhat"] is None
    assert by_name["time_fac"]["status"] == "FAIL"
    assert by_name["mu"]["status"] == "PASS"


def test_fixed_sites_excluded_from_overall_verdict(tmp_path):
    nc = _write_nc(tmp_path / "t.nc", fixed_disp=True)
    idata = az.from_netcdf(nc)
    rows = parameter_diagnostics(idata, params=["mu", "disp"])  # only healthy + fixed
    assert all(r["status"] != "FAIL" for r in rows)


def test_rows_sorted_worst_first(tmp_path):
    nc = _write_nc(tmp_path / "t.nc", fixed_disp=False)
    rows = parameter_diagnostics(az.from_netcdf(nc))
    assert rows[0]["param"] == "time_fac"  # worst rhat on top


# ---------------------------------------------------------------------------
# picker
# ---------------------------------------------------------------------------


@pytest.fixture
def traces_tree(tmp_path, monkeypatch):
    joint = tmp_path / "results_j" / "education"
    cut = tmp_path / "results_c" / "education"
    joint.mkdir(parents=True)
    cut.mkdir(parents=True)
    _write_nc(joint / "NB_births_education_5_traces.nc", fixed_disp=True)
    _write_nc(cut / "NB_births_education_5_stage1_traces.nc", fixed_disp=True)
    comp_dir = cut / "NB_births_education_5_stage2_traces"
    comp_dir.mkdir()
    _write_nc(comp_dir / "component_1.nc", fixed_disp=True, seed=1)
    _write_nc(comp_dir / "component_2.nc", fixed_disp=True, seed=2)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_discover_finds_files_and_stage2_dirs(traces_tree):
    found = cli._discover_trace_targets()
    names = [str(p) for p in found]
    assert len(found) == 3  # joint nc, stage1 nc, stage2 dir
    assert any(n.endswith("_stage2_traces") for n in names)


def test_picker_fills_nc_path(traces_tree, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(lambda *a, **k: "1"))
    args = cli._build_parser().parse_args(["traces"])
    args = cli._interactive_traces_setup(args)
    assert args.nc_path is not None


def test_explicit_path_skips_picker(traces_tree, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("prompted despite explicit path")

    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(boom))
    args = cli._build_parser().parse_args(["traces", "some.nc"])
    assert cli._interactive_traces_setup(args).nc_path == "some.nc"


def test_non_tty_without_path_errors(traces_tree, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    args = cli._build_parser().parse_args(["traces"])
    with pytest.raises(ConfigError, match="nc_path"):
        cli._interactive_traces_setup(args)


# ---------------------------------------------------------------------------
# stage2 component dir -> per-component summary
# ---------------------------------------------------------------------------


def test_component_dir_summarized(traces_tree, capsys):
    comp_dir = Path("results_c/education/NB_births_education_5_stage2_traces")
    ok = cli._print_component_summary(comp_dir, None)
    out = capsys.readouterr().out
    assert "component_1" in out and "component_2" in out
    assert ok is False  # time_fac fails in both fixtures
