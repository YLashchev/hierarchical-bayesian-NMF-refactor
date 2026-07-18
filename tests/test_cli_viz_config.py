"""bpnmf viz --config: re-render honors the same output.* block as `run`.

Without --config, viz falls back to defaults (back-compat). With --config, the
full OutputConfig (figures, aggregate_units, ppc_*, fit_gap_per_unit, ...) is
threaded through, so a re-render reproduces the configured run. CLI --target /
--group override the config.
"""

from unittest.mock import patch

import pandas as pd

from bayesian_panel_nmf import cli

_CFG = """\
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
output:
  figures: ["interval"]
  fit_gap_per_unit: true
  target_unit: "All ban states"
  aggregate_units:
    - unit: "All ban states"
      include_treated_units: true
  ppc_acf_lags: [1, 3]
"""


def _run_viz(args_list, tmp_path):
    draws = tmp_path / "NB_births_total_1.csv"
    draws.write_text("x")
    args = cli._build_parser().parse_args(["viz", "--results", str(draws), *args_list])
    with (
        patch.object(cli, "_read_draws", return_value=pd.DataFrame({"unit": ["A"]})),
        patch("bayesian_panel_nmf.pipeline._run_reporting") as run_rep,
        patch.object(cli, "generate_reports") as gen_rep,
    ):
        cli._viz_command(args)
    return run_rep, gen_rep


def test_with_config_routes_through_run_reporting(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(_CFG)
    run_rep, gen_rep = _run_viz(["--config", str(cfg)], tmp_path)
    # viz --config must use the SAME reporting path as `run`
    assert run_rep.called
    oc = run_rep.call_args.args[2]  # _run_reporting(draws, output_dir, output_config, ...)
    assert oc.fit_gap_per_unit is True
    assert oc.figures == ["interval"]
    assert oc.aggregate_units[0].unit == "All ban states"
    assert oc.ppc_acf_lags == [1, 3]


def test_cli_target_overrides_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(_CFG)
    run_rep, _ = _run_viz(["--config", str(cfg), "--target", "Texas"], tmp_path)
    oc = run_rep.call_args.args[2]
    assert oc.target_unit == "Texas"  # CLI wins over config's "All ban states"


def test_without_config_uses_defaults(tmp_path):
    # back-compat: no --config -> old behavior (generate_reports directly, defaults)
    run_rep, gen_rep = _run_viz([], tmp_path)
    assert not run_rep.called
    assert gen_rep.called
