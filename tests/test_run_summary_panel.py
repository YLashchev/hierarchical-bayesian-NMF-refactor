"""print_run_summary_panel: additive rich terminal panel, no file/data output.

Small synthetic args, no MCMC. Verifies the panel renders without error and
shows the model type plus a PASS/FAIL convergence verdict.
"""

import bayesian_panel_nmf.tables as tables_mod


class _RecordingConsole:
    """Stand-in for rich.console.Console that records printed renderables."""

    last_panels: list = []

    def __init__(self, *args, **kwargs):
        _RecordingConsole.last_panels = []

    def print(self, renderable):
        _RecordingConsole.last_panels.append(renderable)


def test_panel_shows_pass_for_converged_gate(monkeypatch):
    monkeypatch.setattr("rich.console.Console", _RecordingConsole)

    tables_mod.print_run_summary_panel(
        model_type="nativity",
        rank=5,
        num_chains=4,
        chain_method="parallel",
        outcome_distribution="Poisson",
        convergence={"converged": True, "rhat_max": 1.001},
        figures=["interval"],
        artifact_paths="results/nativity",
    )

    assert _RecordingConsole.last_panels, "expected a Panel to be printed"
    text = str(_RecordingConsole.last_panels[-1].renderable)
    assert "nativity" in text
    assert "PASS" in text


def test_panel_shows_fail_for_unconverged_gate(monkeypatch):
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
    assert "FAIL" in text
