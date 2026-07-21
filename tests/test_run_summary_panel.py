"""print_run_summary_panel: additive rich terminal panel, no file/data output.

Small synthetic args, no MCMC. Verifies the panel renders without error, shows
the model type, and reports the divergence count as a plain fact. The panel
carries NO roll-up PASS/FAIL verdict — per-row status lives in the diagnostics
table and the authoritative gate is the _convergence.json write in the pipeline.
"""

import bayesian_panel_nmf.tables as tables_mod


class _RecordingConsole:
    """Stand-in for rich.console.Console that records printed renderables."""

    last_panels: list = []

    def __init__(self, *args, **kwargs):
        _RecordingConsole.last_panels = []

    def print(self, renderable):
        _RecordingConsole.last_panels.append(renderable)


def test_panel_shows_divergences_and_no_verdict(monkeypatch):
    monkeypatch.setattr("rich.console.Console", _RecordingConsole)

    tables_mod.print_run_summary_panel(
        model_type="nativity",
        rank=5,
        num_chains=4,
        chain_method="parallel",
        outcome_distribution="Poisson",
        convergence={"converged": True, "rhat_max": 1.001, "divergences": 0},
        figures=["interval"],
        artifact_paths="results/nativity",
    )

    assert _RecordingConsole.last_panels, "expected a Panel to be printed"
    text = str(_RecordingConsole.last_panels[-1].renderable)
    assert "nativity" in text
    assert "Divergences:" in text and "0" in text
    # No roll-up verdict in the panel regardless of converged state.
    assert "PASS" not in text
    assert "FAIL" not in text


def test_panel_reports_divergence_count_when_present(monkeypatch):
    monkeypatch.setattr("rich.console.Console", _RecordingConsole)

    tables_mod.print_run_summary_panel(
        model_type="nativity",
        rank=5,
        num_chains=4,
        chain_method="parallel",
        outcome_distribution="Poisson",
        convergence={"converged": False, "rhat_max": 1.5, "divergences": 3},
        figures=[],
        artifact_paths=[],
    )

    assert _RecordingConsole.last_panels, "expected a Panel to be printed"
    text = str(_RecordingConsole.last_panels[-1].renderable)
    assert "nativity" in text
    assert "Divergences:" in text and "3" in text
    # Unconverged run still shows no verdict word in the panel.
    assert "PASS" not in text
    assert "FAIL" not in text
