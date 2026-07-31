"""The memory estimator and the calibration record see the skew gates.

`_count_params` modeled `errors_in_variables`, `heteroscedastic_entity_obs` and
`entity_group_pooling` but nothing for `rw_innovation_type` /
`entity_effect_prior_type`, so its projections systematically understated any
fit that turned them on: under a skew innovation the fit allocates `rw_raw_abs`
alongside `rw_raw`, roughly doubling the dominant collected term, and the entity
skew prior adds its own per-entity latents. Preflight admission under-provisioned
those fits, and their calibration records were indistinguishable from a normal
fit of the same shape, so the per-machine refit was fit on a mixture (#412).
"""

from __future__ import annotations

import pytest

from panelcast.gpu_memory.calibration_store import _linear_terms
from panelcast.gpu_memory.estimate import (
    ENTITY_OBS_KEEP_MAX,
    _count_params,
    estimate_memory_gb,
)

SHAPE = {
    "n_observations": 5_000,
    "n_features": 20,
    "n_artists": 800,
    "max_seq": 30,
}


class TestParameterCounts:
    def test_the_skew_walk_doubles_the_dominant_term(self):
        """rw_raw_abs is the same n_artists*(max_seq-1) shape as rw_raw."""
        normal, _ = _count_params(**SHAPE)
        skew, _ = _count_params(**SHAPE, rw_innovation_type="skew_normal")
        walk = SHAPE["n_artists"] * (SHAPE["max_seq"] - 1)
        assert skew - normal == walk

    def test_the_entity_skew_prior_adds_two_entity_sized_latents(self):
        normal, _ = _count_params(**SHAPE)
        skew, _ = _count_params(**SHAPE, entity_effect_prior_type="skew_normal")
        assert skew - normal == 2 * SHAPE["n_artists"]

    @pytest.mark.parametrize("value", [None, "normal"])
    def test_unset_and_normal_are_the_same_model(self, value):
        """Records that predate the flags must read as gate-off, like the
        boolean gates do."""
        assert _count_params(**SHAPE, rw_innovation_type=value) == _count_params(**SHAPE)
        assert _count_params(**SHAPE, entity_effect_prior_type=value) == _count_params(**SHAPE)

    def test_a_single_event_panel_has_no_walk_to_skew(self):
        """rw_latent_sites reports no walk at max_seq <= 1, for either
        innovation, so the estimator cannot invent one."""
        flat = {**SHAPE, "max_seq": 1}
        assert _count_params(**flat, rw_innovation_type="skew_normal") == _count_params(**flat)

    def test_an_unknown_innovation_is_rejected_rather_than_read_as_gaussian(self):
        with pytest.raises(ValueError, match="rw_innovation_type"):
            _count_params(**SHAPE, rw_innovation_type="laplace")


class TestCollectionExclusions:
    def test_the_skew_walk_sites_leave_collection_together(self):
        """train_bayes excludes every walk latent the innovation creates, so
        the collected count must drop by both."""
        _, collected = _count_params(
            **SHAPE, rw_innovation_type="skew_normal", exclude_rw_raw_from_collection=True
        )
        _, normal_collected = _count_params(**SHAPE, exclude_rw_raw_from_collection=True)
        assert collected == normal_collected

    def test_entity_skew_latents_are_kept_below_the_entity_cap(self):
        """Same rule as entity_obs_raw: kept below the cap, dropped above it."""
        small = {**SHAPE, "n_artists": 100}
        _, kept = _count_params(
            **small, entity_effect_prior_type="skew_normal", exclude_rw_raw_from_collection=True
        )
        _, baseline = _count_params(**small, exclude_rw_raw_from_collection=True)
        assert kept - baseline == 2 * small["n_artists"]

    def test_entity_skew_latents_are_dropped_above_the_entity_cap(self):
        big = {**SHAPE, "n_artists": ENTITY_OBS_KEEP_MAX + 1}
        _, dropped = _count_params(
            **big, entity_effect_prior_type="skew_normal", exclude_rw_raw_from_collection=True
        )
        _, baseline = _count_params(**big, exclude_rw_raw_from_collection=True)
        assert dropped == baseline


class TestProjection:
    def _estimate(self, **gates) -> float:
        return estimate_memory_gb(
            **SHAPE, num_chains=4, num_samples=1000, num_warmup=1000, **gates
        ).total_gb

    def test_a_skew_fit_projects_higher_than_a_normal_one(self):
        """The understatement the issue reports: same shape, ~2x the dominant
        collected term, previously the same number."""
        assert self._estimate(rw_innovation_type="skew_normal") > self._estimate()

    def test_the_default_projection_is_unchanged(self):
        assert self._estimate(rw_innovation_type="normal") == self._estimate()


class TestCalibrationRecords:
    def _inputs(self, **gates) -> dict:
        return {**SHAPE, "num_chains": 4, "num_samples": 1000, **gates}

    def test_a_skew_record_does_not_collide_with_a_normal_one(self):
        """Both halves of a mixed refit used to land on identical terms."""
        normal = _linear_terms(self._inputs())
        skew = _linear_terms(self._inputs(rw_innovation_type="skew_normal"))
        assert normal is not None and skew is not None
        assert skew != normal

    def test_records_that_predate_the_flags_read_as_gate_off(self):
        assert _linear_terms(self._inputs()) == _linear_terms(
            self._inputs(rw_innovation_type=None, entity_effect_prior_type=None)
        )

    @pytest.mark.parametrize("value", [1, True, ["skew_normal"]])
    def test_a_non_name_gate_disqualifies_the_record(self, value):
        assert _linear_terms(self._inputs(rw_innovation_type=value)) is None
