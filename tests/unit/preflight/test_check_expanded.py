"""Expanded tests for preflight check logic: _generate_suggestions, _generate_message."""

from unittest import mock

import pytest

from panelcast.gpu_memory import GpuMemoryInfo, MemoryEstimate
from panelcast.pipelines.errors import GpuMemoryError
from panelcast.preflight import PreflightStatus, run_preflight_check
from panelcast.preflight.check import _generate_message, _generate_suggestions


class TestGenerateSuggestions:
    """Tests for _generate_suggestions function."""

    def test_pass_returns_empty(self):
        result = _generate_suggestions(
            status=PreflightStatus.PASS,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            jit_buffer_percent=0.40,
        )
        assert result == ()

    def test_fail_suggests_reduce_chains(self):
        result = _generate_suggestions(
            status=PreflightStatus.FAIL,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            jit_buffer_percent=0.40,
        )
        chain_suggestions = [s for s in result if "--num-chains" in s]
        assert len(chain_suggestions) > 0

    def test_fail_suggests_reduce_samples(self):
        result = _generate_suggestions(
            status=PreflightStatus.FAIL,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            jit_buffer_percent=0.40,
        )
        sample_suggestions = [s for s in result if "--num-samples" in s]
        assert len(sample_suggestions) > 0

    def test_fail_suggests_collection_exclusion_not_warmup(self):
        # Warmup draws are not stored, so reducing warmup is no longer a
        # memory suggestion; the in-sampler exclusion is.
        result = _generate_suggestions(
            status=PreflightStatus.FAIL,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            jit_buffer_percent=0.40,
        )
        assert not [s for s in result if "--num-warmup" in s]
        assert [s for s in result if "--exclude-rw-raw-from-collection" in s]

    def test_no_exclusion_suggestion_when_already_on(self):
        result = _generate_suggestions(
            status=PreflightStatus.FAIL,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            jit_buffer_percent=0.40,
            exclude_rw_raw_from_collection=True,
        )
        assert not [s for s in result if "--exclude-rw-raw-from-collection" in s]

    def test_fail_suggests_full_preflight(self):
        result = _generate_suggestions(
            status=PreflightStatus.FAIL,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            jit_buffer_percent=0.40,
        )
        full_suggestions = [s for s in result if "preflight-full" in s]
        assert len(full_suggestions) > 0

    def test_warning_no_full_preflight_suggestion(self):
        result = _generate_suggestions(
            status=PreflightStatus.WARNING,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            jit_buffer_percent=0.40,
        )
        full_suggestions = [s for s in result if "preflight-full" in s]
        assert len(full_suggestions) == 0

    def test_single_chain_no_chain_suggestion(self):
        result = _generate_suggestions(
            status=PreflightStatus.FAIL,
            num_chains=1,
            num_samples=1000,
            num_warmup=1000,
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            jit_buffer_percent=0.40,
        )
        chain_suggestions = [s for s in result if "--num-chains" in s]
        assert len(chain_suggestions) == 0

    def test_low_samples_no_sample_suggestion(self):
        result = _generate_suggestions(
            status=PreflightStatus.FAIL,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            jit_buffer_percent=0.40,
        )
        sample_suggestions = [s for s in result if "--num-samples" in s]
        assert len(sample_suggestions) == 0

    def test_returns_tuple(self):
        result = _generate_suggestions(
            status=PreflightStatus.FAIL,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            jit_buffer_percent=0.40,
        )
        assert isinstance(result, tuple)


class TestGenerateMessage:
    """Tests for _generate_message function."""

    @pytest.fixture
    def estimate(self):
        return MemoryEstimate(
            base_model_gb=1.0,
            per_chain_gb=0.5,
            jit_buffer_gb=0.8,
            num_chains=4,
        )

    def test_pass_message(self, estimate):
        msg = _generate_message(PreflightStatus.PASS, estimate, 10.0, 62.0)
        assert "passed" in msg
        assert "10.0 GB available" in msg
        assert "62%" in msg

    def test_warning_message(self, estimate):
        msg = _generate_message(PreflightStatus.WARNING, estimate, 5.0, 15.0)
        assert "warning" in msg
        assert "low headroom" in msg

    def test_fail_message(self, estimate):
        msg = _generate_message(PreflightStatus.FAIL, estimate, 2.0, -50.0)
        assert "failed" in msg
        assert "exceeds" in msg

    def test_cannot_check_message(self, estimate):
        msg = _generate_message(PreflightStatus.CANNOT_CHECK, estimate, 0.0, 0.0)
        assert "Cannot check" in msg


class TestRunPreflightCheckExpanded:
    """Expanded tests for run_preflight_check."""

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_returns_preflight_result(self, mock_query):
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,
            used_bytes=0,
            free_bytes=10 * 1024**3,
        )
        result = run_preflight_check(
            n_observations=100,
            n_features=10,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        assert result.estimate is not None
        assert result.device_name == "Test GPU"

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_cannot_check_has_suggestion(self, mock_query):
        mock_query.side_effect = GpuMemoryError("no GPU")
        result = run_preflight_check(
            n_observations=100,
            n_features=10,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        assert result.status == PreflightStatus.CANNOT_CHECK
        assert len(result.suggestions) > 0
        assert "--device cpu" in result.suggestions[0]

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_zero_available_memory(self, mock_query):
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=0,
            used_bytes=0,
            free_bytes=0,
        )
        result = run_preflight_check(
            n_observations=100,
            n_features=10,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        assert result.headroom_percent == -100.0
        assert result.status == PreflightStatus.FAIL

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_custom_headroom_target(self, mock_query):
        """Lower headroom target makes PASS easier."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.024 * 1024**3),
            used_bytes=0,
            free_bytes=int(0.024 * 1024**3),
        )
        # Default 20% headroom -> WARNING
        result_default = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
        )
        # Very low headroom target -> might PASS
        result_low = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
            headroom_target=0.01,
        )
        # With lower headroom target, status should be same or better
        status_order = {
            PreflightStatus.FAIL: 0,
            PreflightStatus.WARNING: 1,
            PreflightStatus.PASS: 2,
        }
        assert status_order.get(result_low.status, 0) >= status_order.get(result_default.status, 0)
