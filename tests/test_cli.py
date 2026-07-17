"""Tests for bayesian_panel_nmf.cli's subcommand dispatch and argument
parsing (Phase 9.2 of the legibility refactor -- consolidates the four
former standalone scripts into one `bpnmf` console script)."""

from pathlib import Path

import pytest

from bayesian_panel_nmf import cli
from bayesian_panel_nmf.config import Config


def test_build_parser_dispatches_run(tmp_path):
    parser = cli._build_parser()
    args = parser.parse_args(["run", "--config", "configs/foo.yaml", "--type", "a"])
    assert args.command == "run"
    assert args.func is cli._run_command
    assert args.config == "configs/foo.yaml"
    assert args.type == "a"


def test_run_defaults_match_previous_script_defaults():
    parser = cli._build_parser()
    args = parser.parse_args(["run"])
    # config defaults to None -> interactive picker (TTY) or error (non-TTY);
    # the old silent nativity_config.yaml default was removed with the picker.
    assert args.config is None
    assert args.type is None
    assert args.rank is None
    assert args.verbose is False
    assert args.log_file is None
    assert args.save_traces is False
    assert args.chains is None
    assert args.chain_method is None


def test_run_flags_map_to_expected_values():
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config",
            "cfg.yaml",
            "--type",
            "groups",
            "--rank",
            "7",
            "-v",
            "--log-file",
            "logs/x.log",
            "--save-traces",
            "--chains",
            "4",
            "--chain-method",
            "sequential",
        ]
    )
    assert args.config == "cfg.yaml"
    assert args.type == "groups"
    assert args.rank == 7
    assert args.verbose is True
    assert args.log_file == "logs/x.log"
    assert args.save_traces is True
    assert args.chains == 4
    assert args.chain_method == "sequential"


def test_build_parser_dispatches_viz():
    parser = cli._build_parser()
    args = parser.parse_args(
        ["viz", "--results", "results/total/draws.csv", "--target", "Texas"]
    )
    assert args.command == "viz"
    assert args.func is cli._viz_command
    assert args.results == "results/total/draws.csv"
    assert args.target == "Texas"
    assert args.group is None


def test_viz_group_flag_is_repeatable():
    parser = cli._build_parser()
    args = parser.parse_args(["viz", "--group", "a", "--group", "b"])
    assert args.group == ["a", "b"]


def test_build_parser_dispatches_traces():
    parser = cli._build_parser()
    args = parser.parse_args(["traces", "path/to/traces.nc", "--plots"])
    assert args.command == "traces"
    assert args.func is cli._traces_command
    assert args.nc_path == "path/to/traces.nc"
    assert args.plots is True


def test_traces_defaults():
    parser = cli._build_parser()
    args = parser.parse_args(["traces", "traces.nc"])
    assert args.plots is False
    assert args.param_filter is None
    assert args.out_dir is None


def test_build_parser_dispatches_init():
    parser = cli._build_parser()
    args = parser.parse_args(["init"])
    assert args.command == "init"
    assert args.func is cli._init_command
    assert args.path == "config.yaml"
    assert args.force is False


def test_build_parser_requires_a_subcommand():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_init_writes_a_config_loadable_file(tmp_path):
    target = tmp_path / "myconfig.yaml"
    args = cli._build_parser().parse_args(["init", str(target)])
    cli._init_command(args)

    assert target.exists()
    loaded = Config.from_yaml(str(target))
    assert loaded.model.types  # base_config.yaml ships with a default type


def test_init_default_path_is_config_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = cli._build_parser().parse_args(["init"])
    cli._init_command(args)

    assert (tmp_path / "config.yaml").exists()


def test_init_refuses_to_overwrite_without_force(tmp_path):
    target = tmp_path / "existing.yaml"
    target.write_text("sentinel: true\n")

    args = cli._build_parser().parse_args(["init", str(target)])
    with pytest.raises(SystemExit):
        cli._init_command(args)

    assert target.read_text() == "sentinel: true\n"


def test_init_force_overwrites_existing_file(tmp_path):
    target = tmp_path / "existing.yaml"
    target.write_text("sentinel: true\n")

    args = cli._build_parser().parse_args(["init", str(target), "--force"])
    cli._init_command(args)

    assert target.read_text() != "sentinel: true\n"
    Config.from_yaml(str(target))


def test_main_dispatches_run_via_sys_argv(monkeypatch, tmp_path: Path):
    """`main()` (no args -> parses sys.argv) end-to-end dispatches to
    `_run_command` for `bpnmf run`, matching the old script's `main()`
    entry-point contract."""
    called: dict = {}
    monkeypatch.setattr(
        cli, "_run_command", lambda args: called.setdefault("args", args)
    )
    monkeypatch.setattr("sys.argv", ["bpnmf", "run", "--config", "cfg.yaml"])

    cli.main()

    assert called["args"].config == "cfg.yaml"
