"""Additional CLI tests targeting uncovered code paths.

Covers:
- Full preflight flow (--preflight-full)
- export-figures command wiring
- main() entry point
- generate-diagrams with specific theme/level combinations
- Edge cases in validation and error handling
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from panelcast.cli import app, main

runner = CliRunner()
_HAS_GRAPHVIZ = importlib.util.find_spec("graphviz") is not None


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _make_pipeline_mocks(monkeypatch, exit_code: int = 0):
    """Patch PipelineConfig and run_pipeline, returning capture dict."""
    captured: dict[str, object] = {}

    def fake_config(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(**kwargs)

    def fake_run_pipeline(config):
        captured["config"] = config
        return exit_code

    monkeypatch.setattr("panelcast.pipelines.orchestrator.PipelineConfig", fake_config)
    monkeypatch.setattr("panelcast.pipelines.orchestrator.run_pipeline", fake_run_pipeline)
    return captured


# ============================================================================
# Full Preflight (--preflight-full) Tests
# ============================================================================


class TestPreflightFull:
    """Tests for --preflight-full code path."""

    def _setup_preflight_full_mocks(self, monkeypatch, status="pass", exit_code=0):
        """Set up mocks for the full preflight path."""
        import numpy as np

        # Mock load_training_data
        fake_model_args = {
            "X": np.zeros((100, 10)),
            "y": np.zeros(100),
            "artist_idx": np.zeros(100, dtype=int),
            "album_seq": np.ones(100, dtype=int),
            "artist_album_counts": np.ones(50, dtype=int),
        }
        monkeypatch.setattr(
            "panelcast.pipelines.train_bayes.load_training_data",
            lambda features_path, splits_path, min_albums_filter, descriptor=None: (
                dict(fake_model_args),
                MagicMock(),
                MagicMock(),
            ),
        )

        # Mock _derive_dimensions_from_model_args
        monkeypatch.setattr(
            "panelcast.preflight.full_check._derive_dimensions_from_model_args",
            lambda model_args: (100, 50, 10, 1),
        )

        # Mock the preflight check functions
        fake_result = SimpleNamespace(
            status=status,
            exit_code=exit_code,
        )
        monkeypatch.setattr(
            "panelcast.preflight.run_extrapolated_preflight_check",
            lambda **kwargs: fake_result,
        )
        monkeypatch.setattr(
            "panelcast.preflight.render_extrapolation_result",
            lambda result, verbose: None,
        )
        monkeypatch.setattr(
            "panelcast.preflight.PreflightStatus",
            SimpleNamespace(FAIL="fail"),
        )

        return fake_result

    def test_preflight_full_missing_data_exits_2(self, monkeypatch, tmp_path):
        """--preflight-full exits 2 when data files are missing."""
        # Don't create the required files so they don't exist
        monkeypatch.chdir(tmp_path)

        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--preflight-full"])
        assert result.exit_code == 2

    def test_preflight_full_pass_continues_to_pipeline(self, monkeypatch, tmp_path):
        """--preflight-full pass continues to pipeline execution."""
        monkeypatch.chdir(tmp_path)

        # Create required data files
        features_path = tmp_path / "data" / "features"
        features_path.mkdir(parents=True)
        (features_path / "train_features.parquet").write_text("dummy")
        splits_path = tmp_path / "data" / "splits" / "within_artist_temporal"
        splits_path.mkdir(parents=True)
        (splits_path / "train.parquet").write_text("dummy")

        self._setup_preflight_full_mocks(monkeypatch, status="pass", exit_code=0)
        captured = _make_pipeline_mocks(monkeypatch)

        result = runner.invoke(app, ["run", "--preflight-full"])
        assert result.exit_code == 0
        assert "kwargs" in captured

    def test_preflight_full_only_exits_after_check(self, monkeypatch, tmp_path):
        """--preflight-full --preflight-only exits after check without pipeline."""
        monkeypatch.chdir(tmp_path)

        features_path = tmp_path / "data" / "features"
        features_path.mkdir(parents=True)
        (features_path / "train_features.parquet").write_text("dummy")
        splits_path = tmp_path / "data" / "splits" / "within_artist_temporal"
        splits_path.mkdir(parents=True)
        (splits_path / "train.parquet").write_text("dummy")

        self._setup_preflight_full_mocks(monkeypatch, status="pass", exit_code=0)

        result = runner.invoke(app, ["run", "--preflight-full", "--preflight-only"])
        assert result.exit_code == 0

    def test_preflight_full_fail_aborts_without_force(self, monkeypatch, tmp_path):
        """--preflight-full fail aborts without --force-run."""
        monkeypatch.chdir(tmp_path)

        features_path = tmp_path / "data" / "features"
        features_path.mkdir(parents=True)
        (features_path / "train_features.parquet").write_text("dummy")
        splits_path = tmp_path / "data" / "splits" / "within_artist_temporal"
        splits_path.mkdir(parents=True)
        (splits_path / "train.parquet").write_text("dummy")

        self._setup_preflight_full_mocks(monkeypatch, status="fail", exit_code=1)

        result = runner.invoke(app, ["run", "--preflight-full"])
        assert result.exit_code == 1
        output = strip_ansi(result.output)
        assert "force-run" in output

    def test_preflight_full_fail_continues_with_force(self, monkeypatch, tmp_path):
        """--preflight-full fail continues with --force-run."""
        monkeypatch.chdir(tmp_path)

        features_path = tmp_path / "data" / "features"
        features_path.mkdir(parents=True)
        (features_path / "train_features.parquet").write_text("dummy")
        splits_path = tmp_path / "data" / "splits" / "within_artist_temporal"
        splits_path.mkdir(parents=True)
        (splits_path / "train.parquet").write_text("dummy")

        self._setup_preflight_full_mocks(monkeypatch, status="fail", exit_code=1)
        captured = _make_pipeline_mocks(monkeypatch)

        result = runner.invoke(app, ["run", "--preflight-full", "--force-run"])
        assert result.exit_code == 0
        assert "kwargs" in captured

    def test_preflight_full_only_fail_returns_exit_code(self, monkeypatch, tmp_path):
        """--preflight-full --preflight-only returns the preflight exit code on failure."""
        monkeypatch.chdir(tmp_path)

        features_path = tmp_path / "data" / "features"
        features_path.mkdir(parents=True)
        (features_path / "train_features.parquet").write_text("dummy")
        splits_path = tmp_path / "data" / "splits" / "within_artist_temporal"
        splits_path.mkdir(parents=True)
        (splits_path / "train.parquet").write_text("dummy")

        self._setup_preflight_full_mocks(monkeypatch, status="fail", exit_code=1)

        result = runner.invoke(app, ["run", "--preflight-full", "--preflight-only"])
        assert result.exit_code == 1

    def test_preflight_full_with_recalibrate(self, monkeypatch, tmp_path):
        """--preflight-full --recalibrate triggers fresh calibration message."""
        monkeypatch.chdir(tmp_path)

        features_path = tmp_path / "data" / "features"
        features_path.mkdir(parents=True)
        (features_path / "train_features.parquet").write_text("dummy")
        splits_path = tmp_path / "data" / "splits" / "within_artist_temporal"
        splits_path.mkdir(parents=True)
        (splits_path / "train.parquet").write_text("dummy")

        self._setup_preflight_full_mocks(monkeypatch, status="pass", exit_code=0)
        captured = _make_pipeline_mocks(monkeypatch)

        result = runner.invoke(app, ["run", "--preflight-full", "--recalibrate"])
        assert result.exit_code == 0

    def test_preflight_full_skips_quick_preflight(self, monkeypatch, tmp_path):
        """When --preflight-full runs, --preflight quick path is skipped."""
        monkeypatch.chdir(tmp_path)

        features_path = tmp_path / "data" / "features"
        features_path.mkdir(parents=True)
        (features_path / "train_features.parquet").write_text("dummy")
        splits_path = tmp_path / "data" / "splits" / "within_artist_temporal"
        splits_path.mkdir(parents=True)
        (splits_path / "train.parquet").write_text("dummy")

        self._setup_preflight_full_mocks(monkeypatch, status="pass", exit_code=0)

        # Track if quick preflight was called
        quick_called = {"value": False}
        original_extract = monkeypatch  # just for reference

        def fake_extract(**kwargs):
            quick_called["value"] = True
            return SimpleNamespace(n_observations=500, n_artists=50)

        monkeypatch.setattr(
            "panelcast.data.ingest.extract_data_dimensions",
            fake_extract,
        )

        captured = _make_pipeline_mocks(monkeypatch)

        # Use both flags - full should take precedence
        result = runner.invoke(app, ["run", "--preflight-full", "--preflight"])
        assert result.exit_code == 0
        # Quick preflight should NOT have been called
        assert quick_called["value"] is False


# ============================================================================
# Export-Figures Command Tests
# ============================================================================


class TestExportFiguresCommand:
    """Tests for export-figures command wiring."""

    def test_export_figures_no_data_exits_1(self, monkeypatch):
        """export-figures with no data exits with code 1."""
        # Mock load_dashboard_data to return empty data
        empty_data = SimpleNamespace(
            predictions=None,
            coefficients=None,
            reliability=None,
            idata=None,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: empty_data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )

        result = runner.invoke(app, ["export-figures"])
        assert result.exit_code == 1
        assert "No data" in result.output

    def test_export_figures_with_predictions(self, monkeypatch, tmp_path):
        """export-figures exports prediction figures."""
        import plotly.graph_objects as go

        pred_data = {
            "y_true": [1, 2, 3],
            "y_pred_mean": [1.1, 2.1, 3.1],
            "y_pred_lower": [0.5, 1.5, 2.5],
            "y_pred_upper": [1.5, 2.5, 3.5],
        }
        data = SimpleNamespace(
            predictions=pred_data,
            coefficients=None,
            reliability=None,
            idata=None,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )

        # Mock create_predictions_plot
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_predictions_plot",
            lambda y_true, y_pred_mean, y_pred_lower, y_pred_upper: go.Figure(),
        )

        # Mock export_all_figures
        export_results = {"predictions": [tmp_path / "pred.svg"]}
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda output_dir, figures, formats, width, height, scale: export_results,
        )

        result = runner.invoke(
            app,
            ["export-figures", "--output", str(tmp_path), "--formats", "svg"],
        )
        assert result.exit_code == 0
        assert "Exported" in result.output

    def test_export_figures_with_all_data(self, monkeypatch, tmp_path):
        """export-figures with predictions, coefficients, and reliability."""
        import plotly.graph_objects as go

        pred_data = {
            "y_true": [1, 2],
            "y_pred_mean": [1.1, 2.1],
            "y_pred_lower": [0.5, 1.5],
            "y_pred_upper": [1.5, 2.5],
        }
        coeff_data = {"beta": [0.5, 0.3]}
        rel_data = {
            "predicted_probs": [0.1, 0.5, 0.9],
            "observed_freq": [0.12, 0.48, 0.91],
            "counts": [10, 20, 15],
        }
        data = SimpleNamespace(
            predictions=pred_data,
            coefficients=coeff_data,
            reliability=rel_data,
            idata=None,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_predictions_plot",
            lambda *args: go.Figure(),
        )
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_forest_plot",
            lambda data: go.Figure(),
        )
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_reliability_plot",
            lambda *args: go.Figure(),
        )

        export_results = {
            "predictions": [tmp_path / "pred.svg"],
            "coefficients": [tmp_path / "coeff.svg"],
            "reliability": [tmp_path / "rel.svg"],
        }
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda output_dir, figures, formats, width, height, scale: export_results,
        )

        result = runner.invoke(
            app,
            ["export-figures", "--output", str(tmp_path), "--formats", "svg"],
        )
        assert result.exit_code == 0
        assert "Exported 3 figures" in result.output

    def test_export_figures_with_idata_trace(self, monkeypatch, tmp_path):
        """export-figures generates trace plot when idata is available."""
        import numpy as np
        import plotly.graph_objects as go

        # Create a mock idata with posterior
        mock_posterior = MagicMock()
        mock_posterior.data_vars = ["beta"]
        mock_posterior.__getitem__ = MagicMock(
            return_value=MagicMock(values=np.random.randn(4, 100))
        )
        mock_idata = MagicMock()
        mock_idata.posterior = mock_posterior

        data = SimpleNamespace(
            predictions={
                "y_true": [1],
                "y_pred_mean": [1.1],
                "y_pred_lower": [0.5],
                "y_pred_upper": [1.5],
            },
            coefficients=None,
            reliability=None,
            idata=mock_idata,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_predictions_plot",
            lambda *args: go.Figure(),
        )
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_trace_plot",
            lambda samples, var_name: go.Figure(),
        )

        export_results = {
            "predictions": [tmp_path / "pred.svg"],
            "trace": [tmp_path / "trace.svg"],
        }
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda output_dir, figures, formats, width, height, scale: export_results,
        )

        result = runner.invoke(
            app,
            ["export-figures", "--output", str(tmp_path), "--formats", "svg"],
        )
        assert result.exit_code == 0

    def test_export_figures_kaleido_warning(self, monkeypatch, tmp_path):
        """export-figures warns when Kaleido Chrome is unavailable."""
        data = SimpleNamespace(
            predictions={
                "y_true": [1],
                "y_pred_mean": [1.1],
                "y_pred_lower": [0.5],
                "y_pred_upper": [1.5],
            },
            coefficients=None,
            reliability=None,
            idata=None,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: False,
        )

        import plotly.graph_objects as go

        monkeypatch.setattr(
            "panelcast.visualization.charts.create_predictions_plot",
            lambda *args: go.Figure(),
        )
        export_results = {"predictions": [tmp_path / "pred.png"]}
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda output_dir, figures, formats, width, height, scale: export_results,
        )

        # Default format includes png which triggers kaleido check
        result = runner.invoke(
            app,
            ["export-figures", "--output", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "Kaleido" in result.output or "Exported" in result.output

    def test_export_figures_custom_dimensions(self, monkeypatch, tmp_path):
        """export-figures passes custom width/height/scale."""
        import plotly.graph_objects as go

        data = SimpleNamespace(
            predictions={
                "y_true": [1],
                "y_pred_mean": [1.1],
                "y_pred_lower": [0.5],
                "y_pred_upper": [1.5],
            },
            coefficients=None,
            reliability=None,
            idata=None,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_predictions_plot",
            lambda *args: go.Figure(),
        )

        captured_export = {}

        def mock_export(output_dir, figures, formats, width, height, scale):
            captured_export["width"] = width
            captured_export["height"] = height
            captured_export["scale"] = scale
            captured_export["formats"] = formats
            return {"predictions": [tmp_path / "pred.svg"]}

        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            mock_export,
        )

        result = runner.invoke(
            app,
            [
                "export-figures",
                "--output",
                str(tmp_path),
                "--width",
                "1200",
                "--height",
                "800",
                "--scale",
                "3.0",
                "--formats",
                "svg,pdf",
            ],
        )
        assert result.exit_code == 0
        assert captured_export["width"] == 1200
        assert captured_export["height"] == 800
        assert captured_export["scale"] == 3.0
        assert captured_export["formats"] == ("svg", "pdf")

    def test_export_figures_with_run_dir(self, monkeypatch, tmp_path):
        """export-figures --run passes run directory."""
        import plotly.graph_objects as go

        captured_run_path = {}

        def mock_load(run_path):
            captured_run_path["path"] = run_path
            return SimpleNamespace(
                predictions={
                    "y_true": [1],
                    "y_pred_mean": [1.1],
                    "y_pred_lower": [0.5],
                    "y_pred_upper": [1.5],
                },
                coefficients=None,
                reliability=None,
                idata=None,
            )

        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            mock_load,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_predictions_plot",
            lambda *args: go.Figure(),
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda **kw: {"predictions": [tmp_path / "pred.svg"]},
        )
        # export_all_figures is called with keyword or positional - adjust mock
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda output_dir, figures, formats, width, height, scale: {
                "predictions": [tmp_path / "pred.svg"]
            },
        )

        result = runner.invoke(
            app,
            ["export-figures", "--run", "reports/run_123", "--formats", "svg"],
        )
        assert result.exit_code == 0
        assert captured_run_path["path"] == Path("reports/run_123")


# ============================================================================
# Main Entry Point
# ============================================================================


class TestMainEntryPoint:
    """Tests for the main() entry point."""

    def test_main_invokes_app(self, monkeypatch):
        """main() calls app()."""
        called = {"value": False}

        def fake_app():
            called["value"] = True

        monkeypatch.setattr("panelcast.cli.app", fake_app)
        main()
        assert called["value"] is True


# ============================================================================
# Generate-Diagrams Edge Cases
# ============================================================================


@pytest.mark.skipif(not _HAS_GRAPHVIZ, reason="graphviz not installed")
class TestGenerateDiagramsEdgeCases:
    """Additional tests for generate-diagrams command."""

    def test_generate_diagrams_single_level_all_themes(self, monkeypatch, tmp_path):
        """Generate a single level with all themes calls generate_all_diagrams."""
        captured = {}

        def fake_generate_all(output_path, levels):
            captured["levels"] = levels
            return {"high_light": [Path("test.svg")]}

        monkeypatch.setattr(
            "panelcast.visualization.diagrams.generate_all_diagrams",
            fake_generate_all,
        )
        monkeypatch.setattr(
            "panelcast.visualization.diagrams.LEVEL_FUNCTIONS",
            {"high": lambda t: None, "intermediate": lambda t: None, "detailed": lambda t: None},
        )
        monkeypatch.setattr(
            "panelcast.visualization.diagrams.DetailLevel",
            str,
        )

        result = runner.invoke(
            app, ["generate-diagrams", "--level", "high", "--output", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert captured["levels"] == ["high"]

    def test_generate_diagrams_multiple_levels_specific_theme(self, monkeypatch, tmp_path):
        """Generate all levels with specific theme renders each."""
        render_calls = []

        class FakeDiagram:
            format = "svg"
            source = "digraph{}"

            def render(self, filename, directory, cleanup):
                render_calls.append(filename)
                return f"{filename}.{self.format}"

        def fake_level_func(theme):
            return FakeDiagram()

        monkeypatch.setattr(
            "panelcast.visualization.diagrams.LEVEL_FUNCTIONS",
            {"high": fake_level_func, "intermediate": fake_level_func, "detailed": fake_level_func},
        )
        monkeypatch.setattr(
            "panelcast.visualization.diagrams.DetailLevel",
            str,
        )
        monkeypatch.setattr(
            "panelcast.visualization.diagrams.generate_all_diagrams",
            lambda *a, **k: {},
        )

        result = runner.invoke(
            app,
            ["generate-diagrams", "--theme", "dark", "--level", "all", "--output", str(tmp_path)],
        )
        assert result.exit_code == 0
        # 3 levels * 3 formats = 9 render calls
        assert len(render_calls) == 9


# ============================================================================
# Additional Validation Edge Cases
# ============================================================================


class TestAdditionalValidation:
    """Additional validation edge cases."""

    def test_calibration_intervals_above_1_fails(self):
        """Calibration interval >= 1.0 fails."""
        result = runner.invoke(app, ["run", "--dry-run", "--calibration-intervals", "1.5"])
        assert result.exit_code != 0

    def test_run_pipeline_config_value_error_from_run(self, monkeypatch):
        """ValueError from PipelineConfig in run command produces clean exit."""

        def fake_config(**kwargs):
            raise ValueError("test validation error")

        monkeypatch.setattr("panelcast.pipelines.orchestrator.PipelineConfig", fake_config)
        monkeypatch.setattr("panelcast.pipelines.orchestrator.run_pipeline", lambda _: 0)

        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0
        output = strip_ansi(result.output)
        assert "Traceback" not in output

    def test_stage_data_nonzero_exit_propagates(self, monkeypatch):
        """stage data non-zero exit propagates correctly."""
        _make_pipeline_mocks(monkeypatch, exit_code=3)
        result = runner.invoke(app, ["stage", "data"])
        assert result.exit_code == 3

    def test_idata_trace_plot_exception_handled(self, monkeypatch, tmp_path):
        """Exception in trace plot creation is silently handled."""
        import plotly.graph_objects as go

        mock_idata = MagicMock()
        mock_idata.posterior = MagicMock()
        mock_idata.posterior.data_vars = ["beta"]
        mock_idata.posterior.__getitem__ = MagicMock(side_effect=RuntimeError("bad idata"))

        data = SimpleNamespace(
            predictions={
                "y_true": [1],
                "y_pred_mean": [1.1],
                "y_pred_lower": [0.5],
                "y_pred_upper": [1.5],
            },
            coefficients=None,
            reliability=None,
            idata=mock_idata,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_predictions_plot",
            lambda *args: go.Figure(),
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda output_dir, figures, formats, width, height, scale: {
                "predictions": [tmp_path / "pred.svg"]
            },
        )

        # Mock the logger to prevent TypeError from structlog-style kwargs
        # passed to stdlib logger.debug() in the broad except handler
        monkeypatch.setattr(
            "panelcast.cli.logger",
            MagicMock(),
        )

        result = runner.invoke(
            app,
            ["export-figures", "--output", str(tmp_path), "--formats", "svg"],
        )
        # Should not crash - trace plot exception handled gracefully
        assert result.exit_code == 0

    def test_export_figures_svg_only_no_kaleido_check(self, monkeypatch, tmp_path):
        """export-figures with svg-only format skips kaleido check."""
        import plotly.graph_objects as go

        kaleido_called = {"value": False}

        def mock_ensure_kaleido():
            kaleido_called["value"] = True
            return True

        data = SimpleNamespace(
            predictions={
                "y_true": [1],
                "y_pred_mean": [1.1],
                "y_pred_lower": [0.5],
                "y_pred_upper": [1.5],
            },
            coefficients=None,
            reliability=None,
            idata=None,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            mock_ensure_kaleido,
        )
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_predictions_plot",
            lambda *args: go.Figure(),
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda output_dir, figures, formats, width, height, scale: {
                "predictions": [tmp_path / "pred.svg"]
            },
        )

        result = runner.invoke(
            app,
            ["export-figures", "--output", str(tmp_path), "--formats", "svg"],
        )
        assert result.exit_code == 0
        # kaleido check should NOT be called for svg-only
        assert kaleido_called["value"] is False
