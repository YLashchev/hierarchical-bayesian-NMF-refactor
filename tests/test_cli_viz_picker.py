"""Interactive viz picker: `bpnmf viz` without --results prompts for a draws file."""

from pathlib import Path

import pytest

from bayesian_panel_nmf import cli
from bayesian_panel_nmf.validation import ConfigError


@pytest.fixture
def results_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two runs: a joint draws CSV and a cut one with a stage1-PPC sidecar."""
    joint = tmp_path / "results_joint" / "education"
    cut = tmp_path / "results_cut" / "education"
    joint.mkdir(parents=True)
    cut.mkdir(parents=True)
    (joint / "NB_births_education_5.csv").write_text("x")
    (cut / "NB_births_education_5.csv").write_text("x")
    (cut / "NB_births_education_5_stage1_ppc.csv").write_text("x")
    # non-draws CSVs that must NOT be offered
    figs = joint / "figs"
    figs.mkdir()
    (figs / "summary_table.csv").write_text("x")
    (joint / "df_education.csv").write_text("x")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _args(**over):
    ns = cli._build_parser().parse_args(["viz"])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_picker_lists_draws_and_autowires_ppc(results_tree, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(lambda *a, **k: "1"))

    args = cli._interactive_viz_setup(_args())
    # sorted: results_cut/... comes first
    assert args.results == str(
        Path("results_cut/education/NB_births_education_5.csv")
    )
    # cut run -> stage1 PPC sidecar auto-attached
    assert args.ppc_results == str(
        Path("results_cut/education/NB_births_education_5_stage1_ppc.csv")
    )


def test_picker_joint_run_has_no_ppc(results_tree, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(lambda *a, **k: "2"))

    args = cli._interactive_viz_setup(_args())
    assert "results_joint" in args.results
    assert args.ppc_results is None


def test_figs_and_df_csvs_not_offered(results_tree, monkeypatch):
    found = cli._discover_draws_files()
    names = [str(p) for p in found]
    assert len(found) == 2
    assert not any("figs" in n or "df_education" in n or "_ppc" in n for n in names)


def test_explicit_results_skips_picker(results_tree, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("prompted despite explicit --results")

    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(boom))
    args = cli._interactive_viz_setup(_args(results="some/path.csv"))
    assert args.results == "some/path.csv"


def test_non_tty_without_results_errors(results_tree, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    with pytest.raises(ConfigError, match="--results"):
        cli._interactive_viz_setup(_args())


def test_no_draws_found_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    with pytest.raises(ConfigError, match="No draws"):
        cli._interactive_viz_setup(_args())
