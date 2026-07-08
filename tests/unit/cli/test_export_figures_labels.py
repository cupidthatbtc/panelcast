"""export-figures labels intervals from interval_level and traces per-element."""

from __future__ import annotations

import arviz as az
import numpy as np
from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.visualization.dashboard import DashboardData

runner = CliRunner()


def test_export_figures_interval_label_and_trace_content(monkeypatch, tmp_path):
    arr = np.arange(2 * 50 * 3, dtype=float).reshape(2, 50, 3)
    data = DashboardData(
        idata=az.from_dict(posterior={"alpha": arr}),
        predictions={
            "y_true": np.array([70.0, 75.0]),
            "y_pred_mean": np.array([72.0, 74.0]),
            "y_pred_lower": np.array([67.0, 69.0]),
            "y_pred_upper": np.array([77.0, 79.0]),
            "interval_level": 0.80,
        },
    )
    captured = {}

    monkeypatch.setattr(
        "panelcast.visualization.dashboard.load_dashboard_data", lambda run_dir: data
    )

    def fake_export(output_dir, figures, formats, width, height, scale):
        captured["figures"] = figures
        return {}

    monkeypatch.setattr("panelcast.visualization.export.export_all_figures", fake_export)

    result = runner.invoke(app, ["export-figures", "--formats", "svg", "--output", str(tmp_path)])
    assert result.exit_code == 0

    figures = captured["figures"]
    # Interval legend reflects the recorded level, not a hardcoded 94%
    assert figures["predictions"].data[0].name == "80% CI"
    # Trace keeps (chain, draw) intact and plots element [.., .., 0]
    trace_fig = figures["trace"]
    assert "alpha[0]" in trace_fig.layout.title.text
    np.testing.assert_array_equal(np.asarray(trace_fig.data[0].y), arr[0, :, 0])
    np.testing.assert_array_equal(np.asarray(trace_fig.data[1].y), arr[1, :, 0])


def test_export_figures_generic_label_when_level_missing(monkeypatch, tmp_path):
    data = DashboardData(
        predictions={
            "y_true": np.array([70.0, 75.0]),
            "y_pred_mean": np.array([72.0, 74.0]),
            "y_pred_lower": np.array([67.0, 69.0]),
            "y_pred_upper": np.array([77.0, 79.0]),
        },
    )
    captured = {}

    monkeypatch.setattr(
        "panelcast.visualization.dashboard.load_dashboard_data", lambda run_dir: data
    )

    def fake_export(output_dir, figures, formats, width, height, scale):
        captured["figures"] = figures
        return {}

    monkeypatch.setattr("panelcast.visualization.export.export_all_figures", fake_export)

    result = runner.invoke(app, ["export-figures", "--formats", "svg", "--output", str(tmp_path)])
    assert result.exit_code == 0
    assert captured["figures"]["predictions"].data[0].name == "CI"
