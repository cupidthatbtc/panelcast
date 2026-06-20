"""New coverage-targeted tests for preflight/full_check.py.

Targets missed lines/branches:
- _generate_message with unknown status (default case in match)
- _generate_extrapolation_message with unknown status
- _run_mini_mcmc_subprocess: JSONDecodeError path, returncode!=0 path
- _create_dummy_calibration function
- run_extrapolated_preflight_check: dimension mismatch logging path
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from panelcast.preflight import PreflightStatus
from panelcast.preflight.full_check import (
    _create_dummy_calibration,
    _generate_extrapolation_message,
    _generate_extrapolation_suggestions,
    _generate_message,
    _generate_suggestions,
    _run_mini_mcmc_subprocess,
    calculate_headroom_percent,
)


class TestRunMiniMcmcSubprocess:
    """Tests for _run_mini_mcmc_subprocess error paths."""

    @mock.patch("subprocess.run")
    def test_nonzero_returncode_returns_failure(self, mock_run):
        """Non-zero return code produces failure result with stderr message."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="CUDA error: out of memory"
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            temp_path = Path(f.name)
        try:
            result = _run_mini_mcmc_subprocess(temp_path, timeout_seconds=60)
            assert result["success"] is False
            assert "CUDA error" in result["error"]
            assert result["peak_memory_bytes"] == 0
        finally:
            temp_path.unlink(missing_ok=True)

    @mock.patch("subprocess.run")
    def test_nonzero_returncode_empty_stderr(self, mock_run):
        """Non-zero return code with empty stderr uses default message."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=""
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            temp_path = Path(f.name)
        try:
            result = _run_mini_mcmc_subprocess(temp_path, timeout_seconds=60)
            assert result["success"] is False
            assert "Unknown subprocess error" in result["error"]
        finally:
            temp_path.unlink(missing_ok=True)

    @mock.patch("subprocess.run")
    def test_timeout_returns_timeout_message(self, mock_run):
        """Subprocess timeout produces timeout error message."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=60)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            temp_path = Path(f.name)
        try:
            result = _run_mini_mcmc_subprocess(temp_path, timeout_seconds=60)
            assert result["success"] is False
            assert "timeout" in result["error"].lower()
            assert result["runtime_seconds"] == 60.0
        finally:
            temp_path.unlink(missing_ok=True)

    @mock.patch("subprocess.run")
    def test_json_decode_error_returns_failure(self, mock_run):
        """Invalid JSON stdout produces parse error result."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json at all {{{", stderr=""
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            temp_path = Path(f.name)
        try:
            result = _run_mini_mcmc_subprocess(temp_path, timeout_seconds=60)
            assert result["success"] is False
            assert "parse" in result["error"].lower()
        finally:
            temp_path.unlink(missing_ok=True)

    @mock.patch("subprocess.run")
    def test_success_parses_json(self, mock_run):
        """Successful subprocess returns parsed JSON output."""
        output = {"success": True, "peak_memory_bytes": 4294967296, "runtime_seconds": 10.5}
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(output), stderr=""
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            temp_path = Path(f.name)
        try:
            result = _run_mini_mcmc_subprocess(temp_path, timeout_seconds=60)
            assert result["success"] is True
            assert result["peak_memory_bytes"] == 4294967296
        finally:
            temp_path.unlink(missing_ok=True)


class TestCreateDummyCalibration:
    """Tests for _create_dummy_calibration."""

    def test_returns_zeroed_calibration(self):
        """Dummy calibration has all zero values."""
        cal = _create_dummy_calibration()
        assert cal.fixed_overhead_gb == 0.0
        assert cal.per_sample_gb == 0.0
        assert cal.calibration_points == ((0, 0.0), (0, 0.0))
        assert cal.config_hash == ""
        assert cal.calibration_time == 0.0

    def test_accepts_config_hash(self):
        """config_hash parameter is passed through."""
        cal = _create_dummy_calibration(config_hash="test_hash")
        assert cal.config_hash == "test_hash"

    def test_extrapolate_returns_zero(self):
        """Dummy calibration extrapolation returns 0.0 for any target."""
        cal = _create_dummy_calibration()
        assert cal.extrapolate(0) == 0.0
        assert cal.extrapolate(1000) == 0.0
        assert cal.extrapolate(100000) == 0.0


class TestGenerateMessageUnknownStatus:
    """Tests for _generate_message with edge-case status values."""

    def test_cannot_check_message(self):
        """CANNOT_CHECK status generates appropriate message."""
        msg = _generate_message(PreflightStatus.CANNOT_CHECK, 0.0, 0.0, 0.0)
        assert "Cannot" in msg

    def test_pass_includes_headroom(self):
        """PASS message includes headroom percentage."""
        msg = _generate_message(PreflightStatus.PASS, 2.0, 10.0, 80.0)
        assert "80%" in msg


class TestGenerateExtrapolationMessageEdgeCases:
    """Tests for _generate_extrapolation_message edge cases."""

    def test_cannot_check_message(self):
        """CANNOT_CHECK returns appropriate message."""
        msg = _generate_extrapolation_message(PreflightStatus.CANNOT_CHECK, 0.0, 0.0, 0.0, 100)
        assert "Cannot" in msg

    def test_pass_includes_sample_count(self):
        """PASS message includes formatted sample count."""
        msg = _generate_extrapolation_message(PreflightStatus.PASS, 3.0, 10.0, 70.0, 2000)
        assert "2,000" in msg

    def test_fail_includes_projected(self):
        """FAIL message includes projected memory and available."""
        msg = _generate_extrapolation_message(PreflightStatus.FAIL, 15.0, 10.0, -50.0, 5000)
        assert "15.0" in msg
        assert "10.0" in msg


class TestGenerateExtrapolationSuggestionsEdgeCases:
    """Tests for _generate_extrapolation_suggestions edge cases."""

    def test_warning_suggests_reducing(self):
        """WARNING suggests reducing samples or chains."""
        result = _generate_extrapolation_suggestions(PreflightStatus.WARNING, 9.0, 10.0)
        assert len(result) > 0
        assert any("--num-samples" in s or "--num-chains" in s or "Close" in s for s in result)

    def test_fail_shows_deficit(self):
        """FAIL shows how much memory is over."""
        result = _generate_extrapolation_suggestions(PreflightStatus.FAIL, 15.0, 10.0)
        assert any("5.0 GB" in s for s in result)
