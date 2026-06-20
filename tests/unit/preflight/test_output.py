"""Tests for preflight output rendering."""

from __future__ import annotations

import pytest

from panelcast.data.ingest import DataDimensions
from panelcast.gpu_memory import MemoryEstimate
from panelcast.preflight import PreflightResult, PreflightStatus, render_preflight_result


class TestRenderPreflightStatusText:
    """Tests for status text rendering."""

    @pytest.fixture
    def pass_result(self) -> PreflightResult:
        """Create a PASS result for testing."""
        return PreflightResult(
            status=PreflightStatus.PASS,
            estimate=MemoryEstimate(
                base_model_gb=0.1,
                per_chain_gb=0.05,
                jit_buffer_gb=0.12,
                num_chains=4,
            ),
            available_gb=10.0,
            total_gpu_gb=10.0,
            headroom_percent=50.0,
            message="Memory check passed",
            suggestions=[],
            device_name="NVIDIA Test GPU",
        )

    @pytest.fixture
    def fail_result(self) -> PreflightResult:
        """Create a FAIL result for testing."""
        return PreflightResult(
            status=PreflightStatus.FAIL,
            estimate=MemoryEstimate(
                base_model_gb=5.0,
                per_chain_gb=2.0,
                jit_buffer_gb=5.2,
                num_chains=4,
            ),
            available_gb=10.0,
            total_gpu_gb=10.0,
            headroom_percent=-80.0,
            message="Memory check failed: exceeds available",
            suggestions=["Try --num-chains 2"],
            device_name="NVIDIA Test GPU",
        )

    @pytest.fixture
    def warning_result(self) -> PreflightResult:
        """Create a WARNING result for testing."""
        return PreflightResult(
            status=PreflightStatus.WARNING,
            estimate=MemoryEstimate(
                base_model_gb=2.0,
                per_chain_gb=1.0,
                jit_buffer_gb=2.4,
                num_chains=4,
            ),
            available_gb=10.0,
            total_gpu_gb=10.0,
            headroom_percent=12.0,
            message="Memory check warning: low headroom",
            suggestions=["Try reducing samples"],
            device_name="NVIDIA Test GPU",
        )

    @pytest.fixture
    def cannot_check_result(self) -> PreflightResult:
        """Create a CANNOT_CHECK result for testing."""
        return PreflightResult(
            status=PreflightStatus.CANNOT_CHECK,
            estimate=None,
            available_gb=0.0,
            total_gpu_gb=0.0,
            headroom_percent=0.0,
            message="Cannot query GPU memory: No GPU detected",
            suggestions=["Consider running with --device cpu"],
        )

    def test_render_pass_contains_pass(self, capsys, pass_result: PreflightResult):
        """Output contains 'PASS' for PASS status."""
        render_preflight_result(pass_result)
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_render_fail_contains_fail(self, capsys, fail_result: PreflightResult):
        """Output contains 'FAIL' for FAIL status."""
        render_preflight_result(fail_result)
        captured = capsys.readouterr()
        assert "FAIL" in captured.out

    def test_render_warning_contains_warning(self, capsys, warning_result: PreflightResult):
        """Output contains 'WARNING' for WARNING status."""
        render_preflight_result(warning_result)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_render_cannot_check_contains_cannot(
        self, capsys, cannot_check_result: PreflightResult
    ):
        """Output contains 'CANNOT' for CANNOT_CHECK status."""
        render_preflight_result(cannot_check_result)
        captured = capsys.readouterr()
        assert "CANNOT" in captured.out


class TestRenderPreflightVerboseMode:
    """Tests for verbose mode memory breakdown."""

    @pytest.fixture
    def result_with_estimate(self) -> PreflightResult:
        """Create a result with detailed estimate for verbose testing."""
        return PreflightResult(
            status=PreflightStatus.PASS,
            estimate=MemoryEstimate(
                base_model_gb=1.0,
                per_chain_gb=0.5,
                jit_buffer_gb=1.2,
                num_chains=4,
            ),
            available_gb=10.0,
            total_gpu_gb=12.0,
            headroom_percent=50.0,
            message="Memory check passed",
            suggestions=[],
            device_name="NVIDIA Test GPU",
        )

    def test_verbose_shows_breakdown(self, capsys, result_with_estimate: PreflightResult):
        """verbose=True includes 'Base model', 'Chains', 'JIT buffer'."""
        render_preflight_result(result_with_estimate, verbose=True)
        captured = capsys.readouterr()

        assert "Base model" in captured.out
        assert "Chains" in captured.out
        assert "JIT buffer" in captured.out

    def test_non_verbose_no_breakdown(self, capsys, result_with_estimate: PreflightResult):
        """verbose=False does not include detailed breakdown."""
        render_preflight_result(result_with_estimate, verbose=False)
        captured = capsys.readouterr()

        # Without verbose, these detailed labels should not appear
        # (might have GB values but not the labeled breakdown)
        assert "Base model" not in captured.out
        assert "JIT buffer" not in captured.out


class TestRenderPreflightSuggestions:
    """Tests for suggestions rendering."""

    def test_render_suggestions(self, capsys):
        """Suggestions list items appear in output."""
        result = PreflightResult(
            status=PreflightStatus.FAIL,
            estimate=MemoryEstimate(
                base_model_gb=5.0,
                per_chain_gb=2.0,
                jit_buffer_gb=5.6,
                num_chains=4,
            ),
            available_gb=8.0,
            total_gpu_gb=8.0,
            headroom_percent=-90.0,
            message="Memory check failed",
            suggestions=[
                "Try --num-chains 2",
                "Try --num-samples 500",
            ],
            device_name="NVIDIA Test GPU",
        )

        render_preflight_result(result)
        captured = capsys.readouterr()

        assert "--num-chains 2" in captured.out
        assert "--num-samples 500" in captured.out


class TestRenderPreflightDataSource:
    """Tests for data source display in output."""

    @pytest.fixture
    def basic_result(self) -> PreflightResult:
        """Create a basic result for data source testing."""
        return PreflightResult(
            status=PreflightStatus.PASS,
            estimate=MemoryEstimate(
                base_model_gb=0.1,
                per_chain_gb=0.05,
                jit_buffer_gb=0.12,
                num_chains=4,
            ),
            available_gb=10.0,
            total_gpu_gb=10.0,
            headroom_percent=50.0,
            message="Memory check passed",
            suggestions=[],
            device_name="Test GPU",
        )

    def test_render_shows_data_source_when_dimensions_provided(
        self, capsys, basic_result: PreflightResult
    ):
        """Output shows dimension source when dimensions provided."""
        dimensions = DataDimensions(
            n_observations=41234,
            n_artists=7521,
            source="from data: all_albums_full.csv",
        )
        render_preflight_result(basic_result, dimensions=dimensions)
        captured = capsys.readouterr()

        assert "41,234 obs" in captured.out
        assert "7,521 artists" in captured.out
        assert "from data:" in captured.out

    def test_render_shows_defaults_note_when_dimensions_none(
        self, capsys, basic_result: PreflightResult
    ):
        """Output shows fixed defaults note when dimensions is None."""
        render_preflight_result(basic_result, dimensions=None)
        captured = capsys.readouterr()

        assert "fixed defaults" in captured.out

    def test_render_shows_default_dimensions_source(self, capsys, basic_result: PreflightResult):
        """Output shows defaults source when using DataDimensions.from_defaults()."""
        dimensions = DataDimensions.from_defaults()
        render_preflight_result(basic_result, dimensions=dimensions)
        captured = capsys.readouterr()

        # Should show the defaults source string, not "fixed defaults" fallback
        assert "defaults" in captured.out
        assert "1,000 obs" in captured.out
        assert "100 artists" in captured.out
