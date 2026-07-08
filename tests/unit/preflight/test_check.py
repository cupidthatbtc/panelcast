"""Tests for preflight check logic."""

from __future__ import annotations

from unittest import mock

import pytest

from panelcast.gpu_memory import GpuMemoryInfo, MemoryEstimate, estimate_memory_gb
from panelcast.pipelines.errors import GpuMemoryError
from panelcast.preflight import PreflightStatus, run_preflight_check
from panelcast.preflight.check import _generate_message, _generate_suggestions

TEST_PARAMS: dict[str, int] = {
    "n_observations": 1000,
    "n_features": 20,
    "n_artists": 50,
    "max_seq": 10,
    "num_chains": 4,
    "num_samples": 1000,
    "num_warmup": 1000,
}


def _warning_band_bytes() -> int:
    """Available memory that puts TEST_PARAMS in the WARNING band.

    Sized from the actual estimate so the test tracks formula changes:
    estimate fits (no FAIL) but headroom is below the 20% target.
    """
    estimate_gb = estimate_memory_gb(**TEST_PARAMS).total_gb
    return int(estimate_gb / 0.9 * 1024**3)


class TestPreflightStatusDetermination:
    """Tests for preflight status determination logic."""

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_preflight_pass_with_headroom(self, mock_query):
        """Estimate well under available memory -> PASS."""
        # 10 GB available, small estimate should easily pass
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,  # 10 GB
            used_bytes=0,
            free_bytes=10 * 1024**3,  # 10 GB free
        )

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.PASS

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_preflight_warning_low_headroom(self, mock_query):
        """Estimate close to available memory (low headroom) -> WARNING."""
        available = _warning_band_bytes()
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=available,
            used_bytes=0,
            free_bytes=available,
        )

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.WARNING

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_preflight_fail_exceeds_available(self, mock_query):
        """Estimate exceeds available memory -> FAIL."""
        # Very little available memory
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.01 * 1024**3),  # 0.01 GB
            used_bytes=0,
            free_bytes=int(0.01 * 1024**3),  # 0.01 GB
        )

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.FAIL

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_preflight_cannot_check_no_gpu(self, mock_query):
        """GPU query raises GpuMemoryError -> CANNOT_CHECK."""
        mock_query.side_effect = GpuMemoryError("No GPU detected")

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.CANNOT_CHECK


class TestPreflightExitCodes:
    """Tests for exit codes returned by PreflightResult."""

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_exit_code_pass(self, mock_query):
        """PASS status -> exit_code 0."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,
            used_bytes=0,
            free_bytes=10 * 1024**3,
        )

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.PASS
        assert result.exit_code == 0

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_exit_code_fail(self, mock_query):
        """FAIL status -> exit_code 1."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.001 * 1024**3),  # Tiny
            used_bytes=0,
            free_bytes=int(0.001 * 1024**3),
        )

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.FAIL
        assert result.exit_code == 1

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_exit_code_warning(self, mock_query):
        """WARNING status -> exit_code 2."""
        # Tight fit with low headroom
        available = _warning_band_bytes()
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=available,
            used_bytes=0,
            free_bytes=available,
        )

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.WARNING
        assert result.exit_code == 2

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_exit_code_cannot_check(self, mock_query):
        """CANNOT_CHECK status -> exit_code 2."""
        mock_query.side_effect = GpuMemoryError("No GPU")

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.CANNOT_CHECK
        assert result.exit_code == 2


class TestPreflightSuggestions:
    """Tests for suggestions generated on FAIL/WARNING."""

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_suggestions_on_fail(self, mock_query):
        """FAIL status includes suggestions."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.001 * 1024**3),
            used_bytes=0,
            free_bytes=int(0.001 * 1024**3),
        )

        result = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
        )

        assert result.status == PreflightStatus.FAIL
        assert len(result.suggestions) > 0

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_suggestions_reduce_chains(self, mock_query):
        """Suggestions include reducing chains when chains > 1."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.001 * 1024**3),
            used_bytes=0,
            free_bytes=int(0.001 * 1024**3),
        )

        result = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,  # More than 1 chain
            num_samples=1000,
            num_warmup=1000,
        )

        assert result.status == PreflightStatus.FAIL
        chain_suggestion = [s for s in result.suggestions if "--num-chains" in s]
        assert len(chain_suggestion) > 0

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_suggestions_reduce_samples(self, mock_query):
        """Suggestions include reducing samples when samples > 500."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.001 * 1024**3),
            used_bytes=0,
            free_bytes=int(0.001 * 1024**3),
        )

        result = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,
            num_samples=1000,  # More than 500
            num_warmup=1000,
        )

        assert result.status == PreflightStatus.FAIL
        sample_suggestion = [s for s in result.suggestions if "--num-samples" in s]
        assert len(sample_suggestion) > 0

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_no_suggestions_on_pass(self, mock_query):
        """PASS status has empty suggestions list."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,
            used_bytes=0,
            free_bytes=10 * 1024**3,
        )

        result = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
        )

        assert result.status == PreflightStatus.PASS
        assert result.suggestions == ()


class TestPreflightMessages:
    """Tests for message content in PreflightResult."""

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_message_contains_estimate(self, mock_query):
        """Message includes estimate GB value."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,
            used_bytes=0,
            free_bytes=10 * 1024**3,
        )

        result = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
        )

        # Message should contain "GB" (from estimate value)
        assert "GB" in result.message

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_message_contains_available(self, mock_query):
        """Message includes available GB value."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,
            used_bytes=0,
            free_bytes=10 * 1024**3,  # 10 GB
        )

        result = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
        )

        # Message should reference available memory
        assert "10.0 GB available" in result.message


# --- from unit/preflight/test_check_expanded.py ---


class TestRunPreflightCheckGateFlags:
    """Gate flags passed to run_preflight_check reach the estimate."""

    @mock.patch("panelcast.preflight.check.query_gpu_memory")
    def test_gate_flags_reach_the_estimate(self, mock_query):
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=100 * 1024**3,
            used_bytes=0,
            free_bytes=100 * 1024**3,
        )
        off = run_preflight_check(**TEST_PARAMS)
        on = run_preflight_check(
            **TEST_PARAMS,
            errors_in_variables=True,
            heteroscedastic_entity_obs=True,
        )
        assert on.estimate.total_gb > off.estimate.total_gb


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

    def test_halved_chains_estimate_uses_exclusion_flag(self):
        """The --num-chains suggestion must price the run's own model: with
        the rw_raw exclusion on, recomputing without the flag quotes a ~25x
        bigger collection term (regression: absurd suggested GB)."""
        # Production scale so the wrong and right totals differ at .1f
        params = {
            "num_samples": 5000,
            "num_warmup": 1000,
            "n_observations": 20000,
            "n_features": 32,
            "n_artists": 7562,
            "max_seq": 50,
        }
        result = _generate_suggestions(
            status=PreflightStatus.FAIL,
            num_chains=4,
            jit_buffer_percent=0.10,
            exclude_rw_raw_from_collection=True,
            **params,
        )
        chain_suggestion = next(s for s in result if "--num-chains 2" in s)
        halved = {**params, "num_chains": 2}
        expected = estimate_memory_gb(
            jit_buffer_percent=0.10, exclude_rw_raw_from_collection=True, **halved
        )
        wrong = estimate_memory_gb(jit_buffer_percent=0.10, **halved)
        assert f"{wrong.total_gb:.1f}" != f"{expected.total_gb:.1f}"  # test is meaningful
        assert f"{expected.total_gb:.1f} GB" in chain_suggestion
        assert f"{wrong.total_gb:.1f} GB" not in chain_suggestion

    def test_halved_chains_estimate_uses_gate_flags(self):
        """New gate flags flow into the halved-chains estimate too."""
        params = {
            "num_samples": 5000,
            "num_warmup": 1000,
            "n_observations": 50000,
            "n_features": 32,
            "n_artists": 500,
            "max_seq": 10,
        }
        result = _generate_suggestions(
            status=PreflightStatus.FAIL,
            num_chains=4,
            jit_buffer_percent=0.10,
            errors_in_variables=True,
            **params,
        )
        chain_suggestion = next(s for s in result if "--num-chains 2" in s)
        halved = {**params, "num_chains": 2}
        expected = estimate_memory_gb(jit_buffer_percent=0.10, errors_in_variables=True, **halved)
        wrong = estimate_memory_gb(jit_buffer_percent=0.10, **halved)
        assert f"{wrong.total_gb:.1f}" != f"{expected.total_gb:.1f}"  # test is meaningful
        assert f"{expected.total_gb:.1f} GB" in chain_suggestion

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
