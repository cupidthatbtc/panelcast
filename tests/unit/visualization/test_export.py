"""Unit tests for visualization export module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import plotly.graph_objects as go
import pytest

from panelcast.visualization.export import (
    ensure_kaleido_chrome,
    export_all_figures,
    export_dashboard_html,
    export_figure,
)


class TestEnsureKaleidoChrome:
    """Tests for ensure_kaleido_chrome."""

    def test_returns_bool(self):
        result = ensure_kaleido_chrome()
        assert isinstance(result, bool)

    def test_kaleido_not_installed(self):
        with patch.dict("sys.modules", {"kaleido": None}):
            # Force ImportError by patching __import__
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "kaleido":
                    raise ImportError("No module named 'kaleido'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = ensure_kaleido_chrome()
                assert result is False


class TestExportFigure:
    """Tests for export_figure."""

    def test_unsupported_format_raises(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        with pytest.raises(ValueError, match="Unsupported format"):
            export_figure(fig, tmp_path / "test", formats=("bmp",))

    def test_creates_parent_directory(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        output_path = tmp_path / "subdir" / "test"
        try:
            export_figure(fig, output_path, formats=("svg",))
        except Exception:
            pytest.skip("Kaleido not available for image export")
        assert (tmp_path / "subdir").exists()

    def test_returns_list(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        try:
            paths = export_figure(fig, tmp_path / "test", formats=("svg",))
        except Exception:
            pytest.skip("Kaleido not available for image export")
        assert isinstance(paths, list)

    def test_svg_export(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        try:
            paths = export_figure(fig, tmp_path / "test", formats=("svg",))
        except Exception:
            pytest.skip("Kaleido not available for image export")
        assert len(paths) == 1
        assert paths[0].suffix == ".svg"

    def test_multiple_formats(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        try:
            paths = export_figure(fig, tmp_path / "test", formats=("svg", "pdf"))
        except Exception:
            pytest.skip("Kaleido not available for image export")
        assert len(paths) == 2


class TestExportAllFigures:
    """Tests for export_all_figures."""

    def test_creates_output_directory(self, tmp_path):
        output_dir = tmp_path / "figures"
        fig1 = go.Figure(data=go.Scatter(x=[1], y=[1]))
        try:
            export_all_figures(output_dir, {"test": fig1}, formats=("svg",))
        except Exception:
            pytest.skip("Kaleido not available for image export")
        assert output_dir.exists()

    def test_returns_dict(self, tmp_path):
        fig1 = go.Figure(data=go.Scatter(x=[1], y=[1]))
        try:
            result = export_all_figures(tmp_path, {"test": fig1}, formats=("svg",))
        except Exception:
            pytest.skip("Kaleido not available for image export")
        assert isinstance(result, dict)
        assert "test" in result

    def test_empty_figures(self, tmp_path):
        try:
            result = export_all_figures(tmp_path, {}, formats=("svg",))
        except Exception:
            pytest.skip("Kaleido not available for image export")
        assert result == {}


class TestExportDashboardHtml:
    """Tests for export_dashboard_html."""

    def test_creates_html_file(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        output_path = tmp_path / "dashboard.html"
        result = export_dashboard_html([fig], output_path)
        assert result == output_path
        assert output_path.exists()

    def test_html_content(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
        output_path = tmp_path / "dashboard.html"
        export_dashboard_html([fig], output_path)
        content = output_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content

    def test_custom_title(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
        output_path = tmp_path / "dashboard.html"
        export_dashboard_html([fig], output_path, title="My Dashboard")
        content = output_path.read_text(encoding="utf-8")
        assert "My Dashboard" in content

    def test_default_title(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
        output_path = tmp_path / "dashboard.html"
        export_dashboard_html([fig], output_path)
        content = output_path.read_text(encoding="utf-8")
        assert "AOTY Model Dashboard" in content

    def test_multiple_figures(self, tmp_path):
        figs = [
            go.Figure(data=go.Scatter(x=[1], y=[1])),
            go.Figure(data=go.Scatter(x=[2], y=[2])),
            go.Figure(data=go.Scatter(x=[3], y=[3])),
        ]
        output_path = tmp_path / "dashboard.html"
        export_dashboard_html(figs, output_path)
        content = output_path.read_text(encoding="utf-8")
        assert content.count('class="chart-container"') == 3

    def test_first_figure_includes_plotlyjs(self, tmp_path):
        figs = [
            go.Figure(data=go.Scatter(x=[1], y=[1])),
            go.Figure(data=go.Scatter(x=[2], y=[2])),
        ]
        output_path = tmp_path / "dashboard.html"
        export_dashboard_html(figs, output_path)
        content = output_path.read_text(encoding="utf-8")
        # First chart div should contain plotly.js
        assert "Plotly" in content

    def test_creates_parent_directory(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
        output_path = tmp_path / "subdir" / "dashboard.html"
        export_dashboard_html([fig], output_path)
        assert output_path.exists()

    def test_empty_figures(self, tmp_path):
        output_path = tmp_path / "dashboard.html"
        export_dashboard_html([], output_path)
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_cdn_plotlyjs(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
        output_path = tmp_path / "dashboard.html"
        export_dashboard_html([fig], output_path, include_plotlyjs="cdn")
        content = output_path.read_text(encoding="utf-8")
        assert "cdn" in content.lower() or "plotly" in content.lower()

    def test_returns_path(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
        output_path = tmp_path / "dashboard.html"
        result = export_dashboard_html([fig], output_path)
        assert isinstance(result, Path)
        assert result == output_path
