"""viz picker also offers a config (like run), so bare `bpnmf viz` is
self-sufficient: pick draws, then pick a config (or defaults)."""


import pytest

from bayesian_panel_nmf import cli

_CFG = """\
data:
  input_file: "unused.csv"
  output_dir: "unused"
  schema:
    unit_col: "s"
    time_col: "t"
    treatment_col: "e"
    outcomes: [{outcome_col: "y", label: "total"}]
model:
  types:
    total: {groups: ["total"], ranks_to_test: [1]}
output:
  figures: ["interval"]
"""


@pytest.fixture
def tree(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "alpha.yaml").write_text(_CFG)
    (tmp_path / "configs" / "beta.yaml").write_text(_CFG)
    d = tmp_path / "results_j" / "total"
    d.mkdir(parents=True)
    (d / "NB_births_total_5.csv").write_text("x")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _args(**over):
    ns = cli._build_parser().parse_args(["viz"])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_picker_prompts_draws_then_config(tree, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    answers = iter(["1", "2"])  # draws=#1, config=beta (#2 after the (none) option)
    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(lambda *a, **k: next(answers)))
    args = cli._interactive_viz_setup(_args())
    assert args.results.endswith("NB_births_total_5.csv")
    assert args.config == "configs/alpha.yaml" or args.config == "configs/beta.yaml"


def test_picker_config_none_option_keeps_defaults(tree, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    answers = iter(["1", "1"])  # draws=#1, config option #1 = (none - defaults)
    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(lambda *a, **k: next(answers)))
    args = cli._interactive_viz_setup(_args())
    assert args.config is None  # chose defaults


def test_explicit_config_skips_config_prompt(tree, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    # only the draws prompt fires; config already given
    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(lambda *a, **k: "1"))
    args = cli._interactive_viz_setup(_args(config="configs/alpha.yaml"))
    assert args.config == "configs/alpha.yaml"
