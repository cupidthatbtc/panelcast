"""Tests for GPU memory estimation module."""

from __future__ import annotations

import pytest

from panelcast.gpu_memory.estimate import (
    COLLECTION_OVERHEAD_FACTOR,
    ENTITY_OBS_KEEP_MAX,
    MemoryEstimate,
    estimate_memory_gb,
)

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


# --- from unit/gpu_memory/test_estimate_expanded.py ---


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


def _collection_term_gb(n_sites: int, num_samples: int) -> float:
    """Per-chain collected size (GB) of n_sites extra scalars per draw."""
    return n_sites * num_samples * 4 * COLLECTION_OVERHEAD_FACTOR / 1024**3


class TestGatedVectorSites:
    """Gate flags add the gated latent sites to the formula (never-under)."""

    def test_eiv_on_exceeds_off_by_n_obs_collection_term(self):
        off = estimate_memory_gb(**STANDARD_PARAMS)
        on = estimate_memory_gb(**STANDARD_PARAMS, errors_in_variables=True)
        assert on.total_gb > off.total_gb
        expected = _collection_term_gb(
            STANDARD_PARAMS["n_observations"], STANDARD_PARAMS["num_samples"]
        )
        assert on.per_chain_gb - off.per_chain_gb == pytest.approx(expected)

    def test_eiv_latent_rides_the_rw_raw_exclusion(self):
        """With the exclusion on, prev_latent_raw leaves collection but is
        still a parameter (train_bayes wiring)."""
        excl_only = estimate_memory_gb(**STANDARD_PARAMS, exclude_rw_raw_from_collection=True)
        both = estimate_memory_gb(
            **STANDARD_PARAMS,
            exclude_rw_raw_from_collection=True,
            errors_in_variables=True,
        )
        assert both.per_chain_gb == pytest.approx(excl_only.per_chain_gb)
        assert both.base_model_gb > excl_only.base_model_gb

    def test_entity_obs_collected_below_cap_even_with_exclusion(self):
        """entity_obs_raw stays in collection below the keep cap."""
        off = estimate_memory_gb(**STANDARD_PARAMS, exclude_rw_raw_from_collection=True)
        on = estimate_memory_gb(
            **STANDARD_PARAMS,
            exclude_rw_raw_from_collection=True,
            heteroscedastic_entity_obs=True,
        )
        expected = _collection_term_gb(STANDARD_PARAMS["n_artists"], STANDARD_PARAMS["num_samples"])
        assert on.per_chain_gb - off.per_chain_gb == pytest.approx(expected)

    def test_entity_obs_dropped_above_cap_only_with_exclusion(self):
        """Above the cap, entity_obs_raw is dropped from collection only when
        the rw_raw exclusion is also on (train_bayes wiring)."""
        big = {**STANDARD_PARAMS, "n_artists": ENTITY_OBS_KEEP_MAX + 1}
        excl_off = estimate_memory_gb(**big, exclude_rw_raw_from_collection=True)
        excl_on = estimate_memory_gb(
            **big, exclude_rw_raw_from_collection=True, heteroscedastic_entity_obs=True
        )
        assert excl_on.per_chain_gb == pytest.approx(excl_off.per_chain_gb)
        assert excl_on.base_model_gb > excl_off.base_model_gb

        no_excl_off = estimate_memory_gb(**big)
        no_excl_on = estimate_memory_gb(**big, heteroscedastic_entity_obs=True)
        expected = _collection_term_gb(big["n_artists"], big["num_samples"])
        assert no_excl_on.per_chain_gb - no_excl_off.per_chain_gb == pytest.approx(expected)

    def test_entity_obs_kept_at_exact_cap(self):
        """At exactly ENTITY_OBS_KEEP_MAX the site stays in collection — the
        drop condition is strictly greater-than (train_bayes wiring)."""
        at_cap = {**STANDARD_PARAMS, "n_artists": ENTITY_OBS_KEEP_MAX}
        off = estimate_memory_gb(**at_cap, exclude_rw_raw_from_collection=True)
        on = estimate_memory_gb(
            **at_cap, exclude_rw_raw_from_collection=True, heteroscedastic_entity_obs=True
        )
        expected = _collection_term_gb(at_cap["n_artists"], at_cap["num_samples"])
        assert on.per_chain_gb - off.per_chain_gb == pytest.approx(expected)

    def test_group_pooling_adds_n_groups_term(self):
        off = estimate_memory_gb(**STANDARD_PARAMS)
        on = estimate_memory_gb(**STANDARD_PARAMS, entity_group_pooling=True, n_groups=7)
        expected = _collection_term_gb(7, STANDARD_PARAMS["num_samples"])
        assert on.per_chain_gb - off.per_chain_gb == pytest.approx(expected)
        zero = estimate_memory_gb(**STANDARD_PARAMS, entity_group_pooling=True, n_groups=0)
        assert zero.total_gb == pytest.approx(off.total_gb)

    def test_entity_obs_keep_max_matches_train_bayes(self):
        from panelcast.pipelines.train_bayes import _ENTITY_OBS_KEEP_MAX

        assert ENTITY_OBS_KEEP_MAX == _ENTITY_OBS_KEEP_MAX

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
