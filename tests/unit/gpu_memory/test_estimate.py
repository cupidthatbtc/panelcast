"""Tests for GPU memory estimation module."""

from __future__ import annotations

import pytest

from panelcast.gpu_memory.estimate import MemoryEstimate, estimate_memory_gb

# Standard test params for memory estimation tests
# Shared across multiple test methods to avoid duplication
STANDARD_PARAMS: dict[str, int] = {
    "n_observations": 1000,
    "n_features": 20,
    "n_artists": 50,
    "max_seq": 10,
    "num_chains": 4,
    "num_samples": 1000,
    "num_warmup": 1000,
}


class TestMemoryEstimate:
    """Tests for MemoryEstimate dataclass."""

    @pytest.fixture
    def sample_estimate(self) -> MemoryEstimate:
        """Create sample estimate: base=1.0, per_chain=0.5, jit=0.8, chains=4."""
        return MemoryEstimate(
            base_model_gb=1.0,
            per_chain_gb=0.5,
            jit_buffer_gb=0.8,
            num_chains=4,
        )

    def test_memory_estimate_total_gb(self, sample_estimate: MemoryEstimate):
        """total_gb = base + (per_chain * chains) + jit_buffer."""
        # Expected: 1.0 + (0.5 * 4) + 0.8 = 1.0 + 2.0 + 0.8 = 3.8
        assert sample_estimate.total_gb == pytest.approx(3.8)

    def test_memory_estimate_chain_memory_gb(self, sample_estimate: MemoryEstimate):
        """chain_memory_gb = per_chain * chains."""
        # Expected: 0.5 * 4 = 2.0
        assert sample_estimate.chain_memory_gb == pytest.approx(2.0)

    def test_memory_estimate_frozen(self, sample_estimate: MemoryEstimate):
        """MemoryEstimate is immutable (frozen=True)."""
        with pytest.raises(AttributeError):
            sample_estimate.base_model_gb = 2.0  # type: ignore[misc]


class TestEstimateMemoryGb:
    """Tests for estimate_memory_gb function."""

    def test_estimate_returns_memory_estimate(self):
        """estimate_memory_gb returns MemoryEstimate type."""
        result = estimate_memory_gb(
            n_observations=100,
            n_features=10,
            n_artists=10,
            max_seq=5,
            num_chains=2,
            num_samples=100,
            num_warmup=100,
        )
        assert isinstance(result, MemoryEstimate)

    def test_estimate_reasonable_range(self):
        """Typical config produces positive, non-trivial estimate."""
        result = estimate_memory_gb(**STANDARD_PARAMS)
        # Estimate should be positive and reasonable (MB to GB scale)
        # The formula estimates conservatively but for this config
        # it's typically in the tens of MB range
        assert result.total_gb > 0.0
        assert result.total_gb < 100.0  # Should not be unreasonably large

    def test_estimate_scales_with_chains(self):
        """Doubling chains roughly doubles chain memory component."""
        result_4_chains = estimate_memory_gb(**STANDARD_PARAMS)
        result_8_chains = estimate_memory_gb(**{**STANDARD_PARAMS, "num_chains": 8})

        # Chain memory should roughly double
        ratio = result_8_chains.chain_memory_gb / result_4_chains.chain_memory_gb
        assert ratio == pytest.approx(2.0)

    def test_estimate_scales_with_samples(self):
        """Doubling samples increases per-chain memory."""
        result_1000_samples = estimate_memory_gb(**STANDARD_PARAMS)
        result_2000_samples = estimate_memory_gb(**{**STANDARD_PARAMS, "num_samples": 2000})

        # More samples = more per-chain memory
        assert result_2000_samples.per_chain_gb > result_1000_samples.per_chain_gb

    def test_estimate_includes_jit_buffer(self):
        """Default 10% allocator-slack buffer is present."""
        result = estimate_memory_gb(**STANDARD_PARAMS)
        # Buffer should be approximately 10% of subtotal (recalibrated from
        # the measured ladder; the old 40% stacked error on error)
        subtotal = result.base_model_gb + result.chain_memory_gb
        expected_jit = subtotal * 0.10
        assert result.jit_buffer_gb == pytest.approx(expected_jit)

    def test_warmup_does_not_change_estimate(self):
        """Warmup draws are not stored (measured: identical peaks)."""
        low = estimate_memory_gb(**{**STANDARD_PARAMS, "num_warmup": 50})
        high = estimate_memory_gb(**{**STANDARD_PARAMS, "num_warmup": 5000})
        assert low.total_gb == pytest.approx(high.total_gb)

    def test_rw_raw_exclusion_removes_dominant_term(self):
        """The in-sampler exclusion drops the n_artists*(max_seq-1) term."""
        production_scale = {
            **STANDARD_PARAMS,
            "n_artists": 7562,
            "max_seq": 50,
            "n_features": 32,
        }
        on = estimate_memory_gb(**production_scale, exclude_rw_raw_from_collection=True)
        off = estimate_memory_gb(**production_scale, exclude_rw_raw_from_collection=False)
        assert on.per_chain_gb < off.per_chain_gb
        # rw_raw dominates the collected-state size at production max_seq
        assert on.per_chain_gb < off.per_chain_gb * 0.05

    def test_estimate_custom_jit_buffer(self):
        """Custom jit_buffer_percent=0.30 produces 30% buffer."""
        result = estimate_memory_gb(**STANDARD_PARAMS, jit_buffer_percent=0.30)
        # JIT buffer should be approximately 30% of subtotal
        subtotal = result.base_model_gb + result.chain_memory_gb
        expected_jit = subtotal * 0.30
        assert result.jit_buffer_gb == pytest.approx(expected_jit)

    def test_estimate_minimal_config(self):
        """Single chain, 100 samples produces small estimate."""
        result = estimate_memory_gb(
            n_observations=100,
            n_features=5,
            n_artists=10,
            max_seq=3,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        # Minimal config should be under 1 GB
        assert result.total_gb < 1.0

    def test_estimate_large_config(self):
        """Many artists (500), long sequences (20) produces larger estimate."""
        small_result = estimate_memory_gb(**STANDARD_PARAMS)
        large_result = estimate_memory_gb(**{**STANDARD_PARAMS, "n_artists": 500, "max_seq": 20})

        # Large config should produce significantly larger estimate
        assert large_result.total_gb > small_result.total_gb
