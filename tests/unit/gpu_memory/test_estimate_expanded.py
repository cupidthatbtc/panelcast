"""Expanded tests for GPU memory estimation: MemoryEstimate and estimate_memory_gb."""

import pytest

from panelcast.gpu_memory.estimate import MemoryEstimate, estimate_memory_gb


class TestMemoryEstimateProperties:
    """Extended tests for MemoryEstimate computed properties."""

    def test_single_chain_memory(self):
        est = MemoryEstimate(base_model_gb=1.0, per_chain_gb=0.5, jit_buffer_gb=0.4, num_chains=1)
        assert est.chain_memory_gb == pytest.approx(0.5)

    def test_many_chains_memory(self):
        est = MemoryEstimate(base_model_gb=1.0, per_chain_gb=0.5, jit_buffer_gb=0.4, num_chains=8)
        assert est.chain_memory_gb == pytest.approx(4.0)

    def test_total_zero_components(self):
        est = MemoryEstimate(base_model_gb=0.0, per_chain_gb=0.0, jit_buffer_gb=0.0, num_chains=1)
        assert est.total_gb == pytest.approx(0.0)

    def test_total_no_jit_buffer(self):
        est = MemoryEstimate(base_model_gb=2.0, per_chain_gb=1.0, jit_buffer_gb=0.0, num_chains=4)
        assert est.total_gb == pytest.approx(6.0)

    def test_total_large_values(self):
        est = MemoryEstimate(base_model_gb=10.0, per_chain_gb=5.0, jit_buffer_gb=12.0, num_chains=4)
        assert est.total_gb == pytest.approx(42.0)

    def test_frozen(self):
        est = MemoryEstimate(base_model_gb=1.0, per_chain_gb=0.5, jit_buffer_gb=0.4, num_chains=4)
        with pytest.raises(AttributeError):
            est.num_chains = 8


class TestEstimateMemoryGbEdgeCases:
    """Edge cases for estimate_memory_gb."""

    def test_zero_observations(self):
        result = estimate_memory_gb(
            n_observations=0,
            n_features=10,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        assert result.total_gb >= 0.0

    def test_zero_features(self):
        result = estimate_memory_gb(
            n_observations=100,
            n_features=0,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        assert result.total_gb >= 0.0

    def test_zero_artists(self):
        result = estimate_memory_gb(
            n_observations=100,
            n_features=10,
            n_artists=0,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        assert result.total_gb >= 0.0

    def test_max_seq_one(self):
        """max_seq=1 should produce rw_raw = artists * max(0, 0) = 0."""
        result = estimate_memory_gb(
            n_observations=100,
            n_features=10,
            n_artists=50,
            max_seq=1,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        assert result.total_gb > 0.0

    def test_max_seq_zero(self):
        """max_seq=0 should handle gracefully."""
        result = estimate_memory_gb(
            n_observations=100,
            n_features=10,
            n_artists=50,
            max_seq=0,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        assert result.total_gb > 0.0

    def test_single_sample(self):
        result = estimate_memory_gb(
            n_observations=100,
            n_features=10,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=1,
            num_warmup=1,
        )
        assert result.total_gb > 0.0

    def test_no_warmup(self):
        result = estimate_memory_gb(
            n_observations=100,
            n_features=10,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=0,
        )
        assert result.total_gb > 0.0

    def test_jit_buffer_zero(self):
        result = estimate_memory_gb(
            n_observations=100,
            n_features=10,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
            jit_buffer_percent=0.0,
        )
        assert result.jit_buffer_gb == pytest.approx(0.0)
        assert result.total_gb == pytest.approx(result.base_model_gb + result.chain_memory_gb)

    def test_scales_with_observations(self):
        small = estimate_memory_gb(
            n_observations=100,
            n_features=10,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        large = estimate_memory_gb(
            n_observations=10000,
            n_features=10,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        assert large.base_model_gb > small.base_model_gb

    def test_base_model_includes_feature_matrix(self):
        """More features = larger feature matrix = larger base_model_gb."""
        few = estimate_memory_gb(
            n_observations=1000,
            n_features=5,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        many = estimate_memory_gb(
            n_observations=1000,
            n_features=50,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        assert many.base_model_gb > few.base_model_gb

    def test_per_chain_scales_with_artists(self):
        """More artists = more parameters = more per-chain memory."""
        few = estimate_memory_gb(
            n_observations=100,
            n_features=10,
            n_artists=10,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        many = estimate_memory_gb(
            n_observations=100,
            n_features=10,
            n_artists=500,
            max_seq=5,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
        )
        assert many.per_chain_gb > few.per_chain_gb

    def test_num_chains_preserved(self):
        result = estimate_memory_gb(
            n_observations=100,
            n_features=10,
            n_artists=10,
            max_seq=5,
            num_chains=7,
            num_samples=100,
            num_warmup=100,
        )
        assert result.num_chains == 7
