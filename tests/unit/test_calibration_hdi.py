"""HDI sample-mass properties (#368).

A closed highest-density interval must span exactly ``ceil(prob * n_samples)``
draws — the fewest whose empirical mass reaches ``prob``. The previous
implementation used that count as an *offset* on an inclusive upper endpoint,
so every interval carried one extra draw and was wider than the mass asked
for, and it searched one placement too few.
"""

from __future__ import annotations

import math

import arviz as az
import numpy as np
import pytest

from panelcast.evaluation.calibration import (
    _hdi_per_observation,
    _hdi_sample_count,
    compute_coverage,
)

PROBS = (0.05, 0.5, 0.8, 0.9, 0.94, 0.95, 0.99)


def _reference_hdi(column: np.ndarray, prob: float) -> tuple[float, float]:
    """Exhaustive narrowest-window search over one column.

    Deliberately written as a plain loop over every admissible placement, so it
    shares no indexing arithmetic with the vectorized implementation.
    """
    ordered = np.sort(np.asarray(column, dtype=float))
    n = ordered.size
    count = min(max(math.ceil(prob * n - 1e-9), 1), n)
    best_width = np.inf
    best = (float(ordered[0]), float(ordered[-1]))
    for start in range(n - count + 1):
        low = float(ordered[start])
        high = float(ordered[start + count - 1])
        if high - low < best_width:
            best_width = high - low
            best = (low, high)
    return best


def _mass_counts(samples: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Draws inside each closed interval, per observation."""
    inside = (samples >= lower[None, :]) & (samples <= upper[None, :])
    return inside.sum(axis=0)


class TestSampleCount:
    """The requested draw count, including binary-float edges."""

    @pytest.mark.parametrize(
        ("n_samples", "prob", "expected"),
        [
            (1000, 0.95, 950),  # 0.95 * 1000 must not round up to 951
            (100, 0.8, 80),
            (1000, 0.9501, 951),
            (999, 0.95, 950),
            (3, 0.5, 2),
            (10, 0.50000000001, 6),  # genuine excess above five draws is not a roundoff
            (100, 0.07, 7),  # one-ULP decimal multiplication overshoot
            (1, 0.99, 1),
            (10, 0.95, 10),  # ceil(9.5) saturates at the full sample
        ],
    )
    def test_count(self, n_samples, prob, expected):
        assert _hdi_sample_count(n_samples, prob) == expected

    @pytest.mark.parametrize("n_samples", [1, 2, 7, 33, 250, 1000])
    @pytest.mark.parametrize("prob", PROBS)
    def test_count_is_the_smallest_reaching_prob(self, n_samples, prob):
        count = _hdi_sample_count(n_samples, prob)
        assert 1 <= count <= n_samples
        assert count / n_samples >= prob - 1e-12
        if count > 1:
            assert (count - 1) / n_samples < prob


class TestExactSampleMass:
    """The headline property: the interval carries the mass it advertises."""

    @pytest.mark.parametrize("n_samples", [7, 50, 251, 1000])
    @pytest.mark.parametrize("prob", PROBS)
    def test_interval_spans_exactly_the_requested_draws(self, n_samples, prob):
        rng = np.random.default_rng(11)
        samples = rng.normal(size=(n_samples, 6))
        lower, upper = _hdi_per_observation(samples, prob)

        expected = _hdi_sample_count(n_samples, prob)
        np.testing.assert_array_equal(_mass_counts(samples, lower, upper), expected)

    @pytest.mark.parametrize("n_samples", [7, 50, 251, 1000])
    @pytest.mark.parametrize("prob", PROBS)
    def test_mass_reaches_prob_without_a_spare_draw(self, n_samples, prob):
        rng = np.random.default_rng(12)
        samples = rng.gamma(shape=2.0, scale=3.0, size=(n_samples, 5))
        lower, upper = _hdi_per_observation(samples, prob)

        mass = _mass_counts(samples, lower, upper) / n_samples
        assert np.all(mass >= prob - 1e-12)
        assert np.all(mass < prob + 1.0 / n_samples + 1e-12)

    def test_deterministic_endpoints_on_an_evenly_spaced_column(self):
        # 10 evenly spaced draws: every placement ties on width, so argmin takes
        # the first. prob=0.5 asks for 5 draws, i.e. indices 0..4 — the old
        # off-by-one returned index 5 as the upper endpoint.
        samples = np.arange(10.0).reshape(10, 1)
        lower, upper = _hdi_per_observation(samples, 0.5)
        assert (float(lower[0]), float(upper[0])) == (0.0, 4.0)

        lower, upper = _hdi_per_observation(samples, 0.9)
        assert (float(lower[0]), float(upper[0])) == (0.0, 8.0)

    def test_950_of_1000_draws_at_the_publication_level(self):
        rng = np.random.default_rng(13)
        samples = rng.normal(size=(1000, 4))
        lower, upper = _hdi_per_observation(samples, 0.95)
        np.testing.assert_array_equal(_mass_counts(samples, lower, upper), 950)


class TestNarrowest:
    """Still the narrowest admissible window, checked against a naive search."""

    @pytest.mark.parametrize(("n_samples", "prob"), [(9, 0.5), (60, 0.8), (137, 0.95)])
    def test_matches_exhaustive_reference(self, n_samples, prob):
        rng = np.random.default_rng(14)
        samples = rng.standard_t(df=3, size=(n_samples, 8))
        lower, upper = _hdi_per_observation(samples, prob)

        for obs in range(samples.shape[1]):
            ref_low, ref_high = _reference_hdi(samples[:, obs], prob)
            assert float(lower[obs]) == pytest.approx(ref_low, rel=0, abs=0)
            assert float(upper[obs]) == pytest.approx(ref_high, rel=0, abs=0)

    def test_no_narrower_window_of_the_same_mass_exists(self):
        rng = np.random.default_rng(15)
        samples = rng.lognormal(mean=0.0, sigma=0.8, size=(200, 3))
        prob = 0.9
        lower, upper = _hdi_per_observation(samples, prob)
        count = _hdi_sample_count(200, prob)

        for obs in range(samples.shape[1]):
            ordered = np.sort(samples[:, obs])
            widths = ordered[count - 1 :] - ordered[: 200 - count + 1]
            assert float(upper[obs] - lower[obs]) == pytest.approx(float(widths.min()))


class TestArviZReference:
    """Cross-check against ArviZ, whose convention differs by at most one draw.

    ArviZ takes ``floor(prob * n)`` as an inclusive *offset*, so it spans
    ``floor(prob * n) + 1`` draws. Whenever ``prob * n`` is not an integer that
    equals ``ceil(prob * n)`` and the two implementations must agree exactly;
    at integer products ArviZ carries one spare draw.
    """

    @pytest.mark.parametrize(("n_samples", "prob"), [(999, 0.95), (777, 0.8), (137, 0.9)])
    def test_identical_where_the_conventions_coincide(self, n_samples, prob):
        assert not float(prob * n_samples).is_integer()
        rng = np.random.default_rng(16)
        samples = rng.normal(loc=70.0, scale=9.0, size=(n_samples, 5))
        lower, upper = _hdi_per_observation(samples, prob)

        for obs in range(samples.shape[1]):
            ref = np.ravel(az.hdi(np.ascontiguousarray(samples[:, obs]), hdi_prob=prob))
            assert float(lower[obs]) == pytest.approx(float(ref[0]))
            assert float(upper[obs]) == pytest.approx(float(ref[1]))

    @pytest.mark.parametrize(("n_samples", "prob"), [(1000, 0.95), (100, 0.94)])
    def test_arviz_carries_one_spare_draw_at_integer_products(self, n_samples, prob):
        assert float(prob * n_samples).is_integer()
        rng = np.random.default_rng(17)
        samples = rng.normal(size=(n_samples, 4))
        lower, upper = _hdi_per_observation(samples, prob)
        counts = _mass_counts(samples, lower, upper)

        for obs in range(samples.shape[1]):
            column = np.ascontiguousarray(samples[:, obs])
            ref = np.ravel(az.hdi(column, hdi_prob=prob))
            ref_count = int(((column >= ref[0]) & (column <= ref[1])).sum())
            assert ref_count == counts[obs] + 1
            assert float(upper[obs] - lower[obs]) <= float(ref[1] - ref[0]) + 1e-12


class TestTies:
    """Repeated values: index windows stay exact, value counts may exceed them."""

    @pytest.mark.parametrize("prob", PROBS)
    def test_constant_column_collapses_to_a_point(self, prob):
        samples = np.full((40, 3), 7.5)
        lower, upper = _hdi_per_observation(samples, prob)
        np.testing.assert_array_equal(lower, 7.5)
        np.testing.assert_array_equal(upper, 7.5)

    @pytest.mark.parametrize("prob", (0.5, 0.8, 0.95))
    def test_heavily_tied_integer_draws_match_the_reference(self, prob):
        rng = np.random.default_rng(18)
        samples = rng.integers(0, 4, size=(120, 6)).astype(float)
        lower, upper = _hdi_per_observation(samples, prob)

        for obs in range(samples.shape[1]):
            ref_low, ref_high = _reference_hdi(samples[:, obs], prob)
            assert float(lower[obs]) == ref_low
            assert float(upper[obs]) == ref_high

    @pytest.mark.parametrize("prob", (0.5, 0.8, 0.95))
    def test_ties_never_drop_below_the_requested_mass(self, prob):
        rng = np.random.default_rng(19)
        samples = rng.integers(0, 3, size=(90, 5)).astype(float)
        lower, upper = _hdi_per_observation(samples, prob)

        counts = _mass_counts(samples, lower, upper)
        # Duplicates outside the chosen index window still fall inside the
        # closed interval, so the value-based count is a lower bound only.
        assert np.all(counts >= _hdi_sample_count(90, prob))
        assert np.all(counts / 90 >= prob - 1e-12)

    def test_two_valued_column_picks_the_dominant_value(self):
        column = np.array([0.0] * 90 + [10.0] * 10)
        samples = column.reshape(100, 1)
        lower, upper = _hdi_per_observation(samples, 0.8)
        assert (float(lower[0]), float(upper[0])) == (0.0, 0.0)


class TestSmallArrays:
    @pytest.mark.parametrize("prob", PROBS)
    def test_single_draw(self, prob):
        samples = np.array([[3.0, -1.0]])
        lower, upper = _hdi_per_observation(samples, prob)
        np.testing.assert_array_equal(lower, [3.0, -1.0])
        np.testing.assert_array_equal(upper, [3.0, -1.0])

    def test_two_draws_half_mass_is_a_single_point(self):
        samples = np.array([[1.0], [5.0]])
        lower, upper = _hdi_per_observation(samples, 0.5)
        assert (float(lower[0]), float(upper[0])) == (1.0, 1.0)

    def test_three_draws_half_mass_takes_the_closest_pair(self):
        samples = np.array([[0.0], [9.0], [10.0]])
        lower, upper = _hdi_per_observation(samples, 0.5)
        assert (float(lower[0]), float(upper[0])) == (9.0, 10.0)

    @pytest.mark.parametrize("n_samples", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("prob", PROBS)
    def test_small_arrays_keep_exact_mass(self, n_samples, prob):
        rng = np.random.default_rng(20)
        samples = rng.normal(size=(n_samples, 3))
        lower, upper = _hdi_per_observation(samples, prob)
        np.testing.assert_array_equal(
            _mass_counts(samples, lower, upper), _hdi_sample_count(n_samples, prob)
        )


class TestFullMass:
    @pytest.mark.parametrize(("n_samples", "prob"), [(10, 0.95), (100, 0.999), (4, 0.8)])
    def test_saturating_windows_return_the_full_range(self, n_samples, prob):
        assert _hdi_sample_count(n_samples, prob) == n_samples
        rng = np.random.default_rng(21)
        samples = rng.normal(size=(n_samples, 4))
        lower, upper = _hdi_per_observation(samples, prob)

        np.testing.assert_array_equal(lower, samples.min(axis=0))
        np.testing.assert_array_equal(upper, samples.max(axis=0))
        np.testing.assert_array_equal(_mass_counts(samples, lower, upper), n_samples)

    def test_full_mass_is_reached_exactly_at_the_boundary_count(self):
        # 20 draws at prob = 0.95 needs 19; at 0.951 it needs all 20.
        rng = np.random.default_rng(22)
        samples = rng.normal(size=(20, 2))
        lower, upper = _hdi_per_observation(samples, 0.95)
        np.testing.assert_array_equal(_mass_counts(samples, lower, upper), 19)

        lower, upper = _hdi_per_observation(samples, 0.951)
        np.testing.assert_array_equal(_mass_counts(samples, lower, upper), 20)


class TestVectorization:
    def test_columns_are_independent(self):
        rng = np.random.default_rng(23)
        samples = np.column_stack(
            [
                rng.normal(loc=0.0, scale=1.0, size=300),
                rng.normal(loc=50.0, scale=12.0, size=300),
                rng.lognormal(mean=1.0, sigma=0.5, size=300),
            ]
        )
        lower, upper = _hdi_per_observation(samples, 0.9)
        assert lower.shape == (3,)
        assert upper.shape == (3,)

        for obs in range(3):
            solo_low, solo_high = _hdi_per_observation(samples[:, [obs]], 0.9)
            assert float(lower[obs]) == float(solo_low[0])
            assert float(upper[obs]) == float(solo_high[0])

    def test_column_order_does_not_matter(self):
        rng = np.random.default_rng(24)
        samples = rng.normal(size=(120, 5))
        lower, upper = _hdi_per_observation(samples, 0.8)
        permuted_low, permuted_high = _hdi_per_observation(samples[:, ::-1], 0.8)
        np.testing.assert_array_equal(lower, permuted_low[::-1])
        np.testing.assert_array_equal(upper, permuted_high[::-1])


class TestComputeCoverageHDI:
    """The public entry point keeps its validation and gets the tighter interval."""

    def test_hdi_bounds_are_ordered_and_cover_calibrated_data(self):
        rng = np.random.default_rng(25)
        n_obs, n_draws = 3000, 400
        mu = rng.normal(70.0, 8.0, size=n_obs)
        y_samples = mu[None, :] + rng.normal(0.0, 5.0, size=(n_draws, n_obs))
        y_true = mu + rng.normal(0.0, 5.0, size=n_obs)

        result = compute_coverage(y_true, y_samples, prob=0.95, interval_type="hdi")
        assert np.all(result.lower_bound <= result.upper_bound)
        assert result.empirical == pytest.approx(0.95, abs=0.02)

    def test_hdi_is_no_wider_than_the_equal_tailed_interval_when_skewed(self):
        rng = np.random.default_rng(26)
        y_samples = rng.lognormal(mean=0.0, sigma=1.0, size=(2000, 200))
        y_true = np.median(y_samples, axis=0)

        hdi = compute_coverage(y_true, y_samples, prob=0.9, interval_type="hdi")
        equal_tailed = compute_coverage(y_true, y_samples, prob=0.9)
        assert hdi.interval_width < equal_tailed.interval_width

    @pytest.mark.parametrize("prob", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_probability_still_rejected(self, prob):
        y_samples = np.zeros((5, 3))
        with pytest.raises(ValueError, match="prob must satisfy"):
            compute_coverage(np.zeros(3), y_samples, prob=prob, interval_type="hdi")

    def test_shape_validation_still_rejected(self):
        with pytest.raises(ValueError, match="y_samples must be 2D"):
            compute_coverage(np.zeros(3), np.zeros(3), interval_type="hdi")
        with pytest.raises(ValueError, match="observations"):
            compute_coverage(np.zeros(3), np.zeros((5, 4)), interval_type="hdi")
        with pytest.raises(ValueError, match="at least one posterior sample"):
            compute_coverage(np.zeros(3), np.zeros((0, 3)), interval_type="hdi")

    def test_unknown_interval_type_still_rejected(self):
        with pytest.raises(ValueError, match="interval_type"):
            compute_coverage(np.zeros(3), np.zeros((5, 3)), interval_type="quantile")
