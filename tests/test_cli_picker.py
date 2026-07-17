"""Interactive run picker: `bpnmf run` without --config prompts for settings."""

from pathlib import Path

import pytest

from bayesian_panel_nmf import cli
from bayesian_panel_nmf.validation import ConfigError

MINIMAL_CFG = """\
data:
  input_file: "unused.csv"
  output_dir: "unused"
  schema:
    unit_col: "state"
    time_col: "time"
    treatment_col: "exposed"
    outcomes: [{outcome_col: "births_total", label: "total"}]
model:
  types:
    total: {groups: ["total"], ranks_to_test: [1]}
    education: {groups: ["total"], ranks_to_test: [1]}
"""


@pytest.fixture
def configs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "configs"
    d.mkdir()
    (d / "alpha.yaml").write_text(MINIMAL_CFG)
    (d / "beta.yaml").write_text(MINIMAL_CFG)
    monkeypatch.chdir(tmp_path)
    return d


def _args(**over):
    ns = cli._build_parser().parse_args(["run"])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_picker_fills_config_type_and_traces(configs_dir, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    answers = iter(["2", "1", True])  # config=beta, type=total, traces=yes

    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(lambda *a, **k: next(answers)))
    monkeypatch.setattr(cli.Confirm, "ask", staticmethod(lambda *a, **k: True))

    args = cli._interactive_run_setup(_args())
    assert args.config == "configs/beta.yaml"
    assert args.type == "total"
    assert args.save_traces is True


def test_picker_all_types_leaves_type_none(configs_dir, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    answers = iter(["1", "3"])  # config=alpha, type=(all) [2 types -> option 3]

    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(lambda *a, **k: next(answers)))
    monkeypatch.setattr(cli.Confirm, "ask", staticmethod(lambda *a, **k: False))

    args = cli._interactive_run_setup(_args())
    assert args.type is None
    assert args.save_traces is False


def test_explicit_config_skips_picker(configs_dir, monkeypatch):
    def boom(*a, **k):  # picker must not prompt at all
        raise AssertionError("prompted despite explicit --config")

    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(boom))
    args = cli._interactive_run_setup(_args(config="configs/alpha.yaml"))
    assert args.config == "configs/alpha.yaml"


def test_non_tty_without_config_errors(configs_dir, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    with pytest.raises(ConfigError, match="--config"):
        cli._interactive_run_setup(_args())


def test_no_configs_dir_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # empty cwd, no configs/
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    with pytest.raises(ConfigError, match="No config files"):
        cli._interactive_run_setup(_args())
