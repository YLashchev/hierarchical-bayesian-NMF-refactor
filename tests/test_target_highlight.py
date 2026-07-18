"""Target-row highlight only when target_unit is EXPLICITLY set.

null/auto-detected target still anchors plots + headline table, but no
per-unit row is highlighted (bold green). An explicit target highlights.
"""

import pandas as pd

from bayesian_panel_nmf import tables


def _capture_styles(monkeypatch):
    """Record (unit_cell, style) for every add_row on the per-unit table."""
    seen = []

    class SpyTable:
        def __init__(self, *a, **k):
            self.title = k.get("title", "")

        def add_column(self, *a, **k):
            pass

        def add_row(self, *cells, style=None):
            seen.append((cells[0], style))

    monkeypatch.setattr("rich.table.Table", SpyTable)

    class _Console:
        def print(self, *a, **k):
            pass

    monkeypatch.setattr("rich.console.Console", _Console)
    return seen


def _per_unit():
    return pd.DataFrame(
        {
            "unit": ["A", "B"],
            "group": ["total", "total"],
            "n_periods": [3, 3],
            "observed": [100.0, 200.0],
            "expected_mean": [90.0, 190.0],
            "expected_lower_95": [80.0, 180.0],
            "expected_upper_95": [100.0, 200.0],
            "excess_mean": [10.0, 10.0],
            "excess_lower_95": [5.0, 5.0],
            "excess_upper_95": [15.0, 15.0],
            "excess_pct_mean": [11.0, 5.0],
            "excess_pct_lower_95": [5.0, 2.0],
            "excess_pct_upper_95": [17.0, 8.0],
        }
    )


def _summary():
    return pd.DataFrame({"Group": ["total"], "Observed": [300]})


def test_explicit_target_highlights_its_row(monkeypatch):
    seen = _capture_styles(monkeypatch)
    tables._print_rich_tables(_summary(), _per_unit(), "A", highlight_unit="A")
    styles = dict(seen)
    assert styles["A"] == "bold green"
    assert styles["B"] is None


def test_null_target_highlights_nothing(monkeypatch):
    seen = _capture_styles(monkeypatch)
    # anchor is "A" (auto-detected) but highlight_unit is None -> no bold row
    tables._print_rich_tables(_summary(), _per_unit(), "A", highlight_unit=None)
    styles = dict(seen)
    assert styles["A"] is None
    assert styles["B"] is None
