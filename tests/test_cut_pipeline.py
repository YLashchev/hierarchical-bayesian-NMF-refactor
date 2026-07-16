"""End-to-end cut pipeline: artifacts, staging, determinism, dispatch.

Runs real tiny NUTS fits -- expect ~1-2 minutes wall time for this file.
"""

import filecmp
import json
from pathlib import Path

import pandas as pd
import pytest
from cut_fixtures import RANK, D, K, N, make_cut_data_dict

from bayesian_panel_nmf import pipeline as _run_analysis_module
from bayesian_panel_nmf.config import Config

ROOT = Path(__file__).resolve().parents[1]


def _load_run_analysis():
    return _run_analysis_module


_data_dict = make_cut_data_dict


def _config():
    return Config.model_validate(
        {
            "data": {
                "input_file": "unused.csv",
                "output_dir": "unused",
                "schema": {
                    "unit_col": "state",
                    "time_col": "time",
                    "treatment_col": "exposed",
                    "outcomes": [{"outcome_col": "births_total", "label": "total"}],
                },
            },
            "model": {
                "inference_mode": "cut",
                "outcome_distribution": "NB",
                "nb_disp": 1e-4,
                "sample_disp": False,
                "adjust_for_missingness": True,
                "model_treated": True,
                "types": {"total": {"groups": ["total"], "ranks_to_test": [1]}},
            },
            "mcmc": {
                "auto_parallelism": False,
                "num_chains": 2,
                "chain_method": "sequential",
                "num_warmup": 15,
                "num_samples": 16,
                "thinning": 1,
                "random_seed": 0,
                "progress_bar": False,
            },
            "cut": {
                "num_stage1_draws": 2,
                "stage2_draws_per_component": 4,
                "stage2_mcmc": {"num_warmup": 15, "num_samples": 16},
            },
            "output": {},
        }
    )


@pytest.fixture(scope="module")
def ra():
    return _load_run_analysis()


def _run(ra, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _config()
    ra._run_cut_rank(
        rank=RANK,
        data_dict=_data_dict(),
        model_type="total",
        config=config,
        type_output_dir=out_dir,
        ranks=[RANK],
        save_traces=True,
        output_config=config.output,
    )
    return f"{ra._draws_filename(config, 'total', RANK)}_cut"


def test_cut_rank_artifacts_and_determinism(ra, tmp_path):
    out1 = tmp_path / "run1"
    stem = _run(ra, out1)

    csv = out1 / f"{stem}.csv"
    assert csv.exists()
    assert (out1 / f"{stem}_stage1_ppc.csv").exists()
    assert (out1 / f"{stem}_convergence.json").exists()
    assert (out1 / f"{stem}_stage1_traces.nc").exists()
    assert (out1 / f"{stem}_stage2_traces" / "component_0001.nc").exists()
    assert (out1 / f"{stem}_stage2_traces" / "component_0002.nc").exists()
    assert not list(out1.glob(".tmp_cut_*"))

    df = pd.read_csv(csv)
    cells = K * D * N
    assert len(df) == 2 * 4 * cells
    assert df[".draw"].nunique() == 8
    assert set(df["cut_component"]) == {1, 2}
    assert set(df[".chain"]) <= {1, 2}
    for col in ["stage1_draw", "stage1_chain", "stage1_iteration"]:
        assert col in df.columns
    # mu repeats the fixed baseline within a component: one unique mu per cell
    one_cell = df[
        (df["group"] == "g0") & (df["unit"] == "u0") & (df["time"] == df["time"].min())
    ]
    assert one_cell.groupby("cut_component")["mu"].nunique().max() == 1

    ppc = pd.read_csv(out1 / f"{stem}_stage1_ppc.csv")
    assert len(ppc) == 2 * 16 * cells  # full retained Stage-1 posterior
    assert "cut_component" not in ppc.columns

    manifest = json.loads((out1 / f"{stem}_convergence.json").read_text())
    assert manifest["inference_mode"] == "cut"
    assert set(manifest["stage1"]) >= {"rhat_max", "converged"}
    fits = manifest["stage2"]["fits"]
    assert [f["component"] for f in fits] == [1, 2]
    assert all(f["output_draws"] == 4 for f in fits)
    assert all(f["retained_draws"] == 32 for f in fits)
    assert isinstance(manifest["converged"], bool)

    out2 = tmp_path / "run2"
    _run(ra, out2)
    assert filecmp.cmp(csv, out2 / f"{stem}.csv", shallow=False)


def test_joint_config_never_reaches_cut_path(ra, monkeypatch):
    marker = RuntimeError("joint path reached")

    def boom(*args, **kwargs):
        raise marker

    monkeypatch.setattr(ra, "run_mcmc_inference", boom)
    called = []
    monkeypatch.setattr(ra, "_run_cut_rank", lambda *a, **k: called.append(True))

    config = _config().model_copy(
        update={"model": _config().model.model_copy(update={"inference_mode": None})}
    )
    with pytest.raises(RuntimeError, match="joint path reached"):
        ra._run_single_rank(
            rank=RANK,
            data_dict=_data_dict(),
            model_type="total",
            config=config,
            type_output_dir=Path("/nonexistent"),
            ranks=[RANK],
            save_traces=False,
            output_config=config.output,
        )
    assert called == []


def test_cut_config_dispatches_before_joint_body(ra, monkeypatch):
    called = []
    monkeypatch.setattr(ra, "_run_cut_rank", lambda *a, **k: called.append(True))

    def boom(*args, **kwargs):
        raise AssertionError("joint body must not run in cut mode")

    monkeypatch.setattr(ra, "run_mcmc_inference", boom)
    config = _config()
    ra._run_single_rank(
        rank=RANK,
        data_dict={},
        model_type="total",
        config=config,
        type_output_dir=Path("/nonexistent"),
        ranks=[RANK],
        save_traces=False,
        output_config=config.output,
    )
    assert called == [True]


def test_execution_failure_publishes_nothing(ra, tmp_path, monkeypatch):
    from bayesian_panel_nmf.validation import DataError

    dd = _data_dict()
    dd["missing_idx_array"] = ~dd["control_idx_array"]  # no exposed nonmissing
    config = _config()
    with pytest.raises(DataError):
        ra._run_cut_rank(
            rank=RANK,
            data_dict=dd,
            model_type="total",
            config=config,
            type_output_dir=tmp_path,
            ranks=[RANK],
            save_traces=False,
            output_config=config.output,
        )
    assert list(tmp_path.iterdir()) == []
