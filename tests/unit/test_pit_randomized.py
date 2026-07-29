"""Randomized-rank PIT (#369): validity, provenance, and reproducibility.

The old deterministic mid-P quantity ``(below + 0.5 * equal) / n_draws`` is not
uniform under calibration. These tests pin the replacement two ways: simulation
uniformity where a calibrated forecast must pass and mid-P must fail, and exact
reproducibility from the recorded seed with no global RNG involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelcast.evaluation import slices as slices_mod
from panelcast.evaluation.calibration import (
    PIT_DEFAULT_SEED,
    PIT_METHOD,
    compute_pit_per_row,
    compute_pit_values,
    summarize_pit,
)
from panelcast.evaluation.conformal import conformalize
from panelcast.evaluation.slices import calibration_by_slice, coverage_by_slice


def _mid_p(y_true: np.ndarray, y_samples: np.ndarray) -> np.ndarray:
    """The superseded convention, kept here so the tests can prove it fails."""
    below = (y_samples < y_true[None, :]).sum(axis=0)
    equal = (y_samples == y_true[None, :]).sum(axis=0)
    return (below + 0.5 * equal) / y_samples.shape[0]


def _ks_pvalue(pit: np.ndarray) -> float:
    return float(stats.kstest(pit, "uniform").pvalue)


def _calibrated_poisson(rng, n_obs=8000, n_draws=60, rate=3.0):
    """Observations and draws from one shared discrete predictive per row."""
    rates = rng.gamma(shape=2.0, scale=rate / 2.0, size=n_obs)
    y_true = rng.poisson(rates).astype(float)
    y_samples = rng.poisson(np.broadcast_to(rates, (n_draws, n_obs))).astype(float)
    return y_true, y_samples


def _calibrated_beta_binomial(rng, n_obs=8000, n_draws=60, trials=3):
    probs = rng.beta(2.0, 3.0, size=n_obs)
    y_true = rng.binomial(trials, probs).astype(float)
    y_samples = rng.binomial(trials, np.broadcast_to(probs, (n_draws, n_obs))).astype(float)
    return y_true, y_samples


def _calibrated_censored(rng, n_obs=8000, n_draws=60, bound=100.0):
    """Location-scale draws clipped at a bound — a real predictive atom."""
    mu = rng.normal(95.0, 4.0, size=n_obs)
    y_true = np.clip(mu + rng.normal(0.0, 6.0, size=n_obs), 0.0, bound)
    y_samples = np.clip(
        np.broadcast_to(mu, (n_draws, n_obs)) + rng.normal(0.0, 6.0, size=(n_draws, n_obs)),
        0.0,
        bound,
    )
    return y_true, y_samples


class TestSimulationUniformity:
    """Calibrated forecasts must produce uniform PIT; mid-P must not."""

    @pytest.mark.parametrize(
        "maker", [_calibrated_poisson, _calibrated_beta_binomial, _calibrated_censored]
    )
    def test_calibrated_discrete_forecast_is_uniform(self, maker):
        rng = np.random.default_rng(4242)
        y_true, y_samples = maker(rng)
        pit = compute_pit_per_row(y_true, y_samples, seed=17)
        assert _ks_pvalue(pit) > 1e-3

    @pytest.mark.parametrize(
        "maker", [_calibrated_poisson, _calibrated_beta_binomial, _calibrated_censored]
    )
    def test_mid_p_fails_the_same_calibrated_forecast(self, maker):
        rng = np.random.default_rng(4242)
        y_true, y_samples = maker(rng)
        assert _ks_pvalue(_mid_p(y_true, y_samples)) < 1e-6

    def test_uniform_across_many_randomization_seeds(self):
        rng = np.random.default_rng(99)
        y_true, y_samples = _calibrated_poisson(rng, n_obs=2000, n_draws=40)
        pvalues = [_ks_pvalue(compute_pit_per_row(y_true, y_samples, seed=s)) for s in range(12)]
        # A valid PIT gives roughly uniform p-values; a broken one would put
        # every seed in the floor.
        assert min(pvalues) > 1e-4
        assert max(pvalues) > 0.2

    def test_coarse_continuous_draws_are_uniform(self):
        # Ten draws per row: the empirical CDF is a coarse lattice, and only
        # the randomization makes the PIT continuous-uniform.
        rng = np.random.default_rng(5)
        n_obs, n_draws = 4000, 10
        mu = rng.normal(0.0, 1.0, size=n_obs)
        y_true = mu + rng.normal(0.0, 1.0, size=n_obs)
        y_samples = np.broadcast_to(mu, (n_draws, n_obs)) + rng.normal(
            0.0, 1.0, size=(n_draws, n_obs)
        )
        assert _ks_pvalue(compute_pit_per_row(y_true, y_samples, seed=3)) > 1e-3
        assert _ks_pvalue(_mid_p(y_true, y_samples)) < 1e-6

    def test_mean_and_spread_match_uniform(self):
        rng = np.random.default_rng(6)
        y_true, y_samples = _calibrated_beta_binomial(rng, n_obs=6000, n_draws=50)
        summary = compute_pit_values(y_true, y_samples, seed=8)
        assert summary["mean"] == pytest.approx(0.5, abs=0.02)
        assert summary["std"] == pytest.approx(np.sqrt(1 / 12), abs=0.02)
        assert summary["max_abs_dev_from_uniform"] < 0.02

    def test_miscalibration_is_still_detected(self):
        rng = np.random.default_rng(7)
        n_obs, n_draws = 3000, 100
        y_true = rng.normal(0.0, 2.0, size=n_obs)  # truth twice as wide
        y_samples = rng.normal(0.0, 1.0, size=(n_draws, n_obs))
        summary = compute_pit_values(y_true, y_samples, n_bins=10, seed=1)
        freq = np.asarray(summary["counts"], dtype=float) / summary["n_obs"]
        assert freq[0] > 0.15 and freq[-1] > 0.15
        assert freq[4] < 0.08 and freq[5] < 0.08
        assert _ks_pvalue(compute_pit_per_row(y_true, y_samples, seed=1)) < 1e-6


class TestTiedMass:
    """Ties are the case mid-P collapsed onto a single interior point."""

    def test_all_draws_tie_the_observation(self):
        y_true = np.full(2000, 5.0)
        y_samples = np.full((30, 2000), 5.0)
        pit = compute_pit_per_row(y_true, y_samples, seed=2)
        # Whole predictive mass ties: the PIT is the raw randomization.
        assert _ks_pvalue(pit) > 1e-3
        assert pit.min() < 0.05 and pit.max() > 0.95
        np.testing.assert_allclose(_mid_p(y_true, y_samples), 0.5)

    def test_tie_block_is_bracketed_by_its_neighbours(self):
        # 3 draws below, 4 tied, 3 above: the PIT must land inside the tie
        # block's rank interval [3/11, 8/11) for every randomization.
        y_true = np.array([10.0])
        column = np.array([1.0, 2.0, 3.0, 10.0, 10.0, 10.0, 10.0, 11.0, 12.0, 13.0])
        y_samples = column.reshape(-1, 1)
        for seed in range(50):
            pit = compute_pit_per_row(y_true, y_samples, seed=seed)[0]
            assert 3 / 11 <= pit < 8 / 11

    def test_observation_below_every_draw_stays_in_the_first_cell(self):
        y_true = np.array([-1.0])
        y_samples = np.arange(1.0, 11.0).reshape(-1, 1)
        for seed in range(20):
            assert 0.0 <= compute_pit_per_row(y_true, y_samples, seed=seed)[0] < 1 / 11

    def test_observation_above_every_draw_stays_in_the_last_cell(self):
        y_true = np.array([99.0])
        y_samples = np.arange(1.0, 11.0).reshape(-1, 1)
        for seed in range(20):
            assert 10 / 11 <= compute_pit_per_row(y_true, y_samples, seed=seed)[0] < 1.0

    def test_pit_lies_inside_its_rank_bracket(self):
        # The randomization may only move an observation within the rank cell
        # its own draws define — never across a neighbouring value.
        rng = np.random.default_rng(13)
        n_draws, n_obs = 30, 400
        y_samples = rng.integers(0, 6, size=(n_draws, n_obs)).astype(float)
        y_true = rng.integers(0, 6, size=n_obs).astype(float)
        pit = compute_pit_per_row(y_true, y_samples, seed=6)

        below = (y_samples < y_true[None, :]).sum(axis=0)
        equal = (y_samples == y_true[None, :]).sum(axis=0)
        assert np.all(pit >= below / (n_draws + 1))
        assert np.all(pit < (below + equal + 1) / (n_draws + 1))


class TestReproducibility:
    def test_same_seed_is_bitwise_identical(self):
        rng = np.random.default_rng(0)
        y_true, y_samples = _calibrated_poisson(rng, n_obs=500, n_draws=20)
        first = compute_pit_per_row(y_true, y_samples, seed=1234)
        second = compute_pit_per_row(y_true, y_samples, seed=1234)
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_differ(self):
        rng = np.random.default_rng(0)
        y_true, y_samples = _calibrated_poisson(rng, n_obs=500, n_draws=20)
        first = compute_pit_per_row(y_true, y_samples, seed=1234)
        second = compute_pit_per_row(y_true, y_samples, seed=1235)
        assert not np.array_equal(first, second)

    def test_no_global_randomness(self):
        rng = np.random.default_rng(0)
        y_true, y_samples = _calibrated_poisson(rng, n_obs=300, n_draws=20)

        np.random.seed(7)
        before = np.random.get_state()
        first = compute_pit_per_row(y_true, y_samples, seed=5)
        after = np.random.get_state()

        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]

        # Consuming global randomness in between cannot move the result.
        np.random.random(100)
        np.testing.assert_array_equal(first, compute_pit_per_row(y_true, y_samples, seed=5))

    def test_values_stay_in_the_unit_interval(self):
        rng = np.random.default_rng(3)
        y_true, y_samples = _calibrated_censored(rng, n_obs=1500, n_draws=25)
        for seed in range(6):
            pit = compute_pit_per_row(y_true, y_samples, seed=seed)
            assert pit.shape == y_true.shape
            assert pit.min() >= 0.0
            assert pit.max() < 1.0

    def test_seed_and_method_are_recorded(self):
        rng = np.random.default_rng(4)
        y_true, y_samples = _calibrated_poisson(rng, n_obs=200, n_draws=20)
        summary = compute_pit_values(y_true, y_samples, seed=77)
        assert summary["method"] == PIT_METHOD == "randomized_rank"
        assert summary["randomization_seed"] == 77

    def test_summary_matches_per_row_computation(self):
        rng = np.random.default_rng(5)
        y_true, y_samples = _calibrated_poisson(rng, n_obs=400, n_draws=30)
        pit = compute_pit_per_row(y_true, y_samples, seed=21)
        assert compute_pit_values(y_true, y_samples, seed=21) == summarize_pit(pit, seed=21)


class TestValidation:
    def test_rejects_incompatible_shapes(self):
        with pytest.raises(ValueError, match="incompatible"):
            compute_pit_per_row(np.zeros(5), np.zeros((10, 4)), seed=0)

    def test_rejects_non_1d_observations(self):
        with pytest.raises(ValueError, match="y_true must be 1D"):
            compute_pit_per_row(np.zeros((2, 3)), np.zeros((10, 3)), seed=0)

    def test_rejects_empty_draws(self):
        with pytest.raises(ValueError, match="at least one predictive draw"):
            compute_pit_per_row(np.zeros(3), np.zeros((0, 3)), seed=0)

    def test_omitted_seed_uses_a_recorded_compatibility_default(self):
        y_true = np.zeros(3)
        y_samples = np.zeros((5, 3))
        np.testing.assert_array_equal(
            compute_pit_per_row(y_true, y_samples),
            compute_pit_per_row(y_true, y_samples, seed=PIT_DEFAULT_SEED),
        )
        pit = compute_pit_per_row(y_true, y_samples)
        assert summarize_pit(pit)["randomization_seed"] == PIT_DEFAULT_SEED == 0
        summary = compute_pit_values(y_true, y_samples)
        assert summary["randomization_seed"] == PIT_DEFAULT_SEED == 0

    def test_pipeline_split_streams_are_disjoint(self):
        from panelcast.pipelines.evaluate import (
            _PIT_CONFORMAL_OFFSET,
            _PIT_PRIMARY_OFFSET,
            _PIT_SECONDARY_OFFSET,
        )

        y_true = np.zeros(20)
        y_samples = np.zeros((10, 20))
        streams = [
            compute_pit_per_row(y_true, y_samples, seed=42 + offset)
            for offset in (_PIT_CONFORMAL_OFFSET, _PIT_PRIMARY_OFFSET, _PIT_SECONDARY_OFFSET)
        ]

        assert not np.array_equal(streams[0], streams[1])
        assert not np.array_equal(streams[1], streams[2])

    def test_empty_summary_reports_nulls_not_nan(self):
        summary = summarize_pit(np.array([]), seed=3)
        assert summary["n_obs"] == 0
        assert summary["mean"] is None
        assert summary["max_abs_dev_from_uniform"] is None


class TestSliceConsistency:
    """One randomization per run: a slice's PIT is its rows' PIT, exactly."""

    def _fixture(self, n=300, n_draws=200):
        rng = np.random.default_rng(12)
        y_true = rng.normal(70.0, 5.0, size=n)
        y_samples = rng.normal(loc=y_true, scale=5.0, size=(n_draws, n))
        labels = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2), dtype=object)
        return y_true, y_samples, labels

    def test_slice_deviation_equals_masked_global_pit(self):
        y_true, y_samples, labels = self._fixture()
        out = coverage_by_slice(y_true, y_samples, labels, (0.8,), dimension="group", seed=31)
        pit = compute_pit_per_row(y_true, y_samples, seed=31)

        by_label = {s.label: s for s in out}
        for label in ("a", "b"):
            expected = summarize_pit(pit[labels == label], seed=31)["max_abs_dev_from_uniform"]
            assert by_label[label].pit_max_abs_dev == expected

    def test_slice_seed_is_recorded_and_changes_the_result(self):
        y_true, y_samples, labels = self._fixture()
        row_ids = pd.DataFrame(
            {
                "entity": [f"e{i % 30}" for i in range(len(y_true))],
                "group": labels,
                "n_reviews": np.arange(len(y_true)) % 400 + 1,
                "train_history": np.arange(len(y_true)) % 15,
            }
        )
        first = calibration_by_slice(y_true, y_samples, row_ids, (0.8,), seed=31)
        second = calibration_by_slice(y_true, y_samples, row_ids, (0.8,), seed=32)
        assert first["pit_randomization_seed"] == 31
        assert second["pit_randomization_seed"] == 32
        devs_first = [s["pit_max_abs_dev"] for s in first["slices"]]
        devs_second = [s["pit_max_abs_dev"] for s in second["slices"]]
        assert devs_first != devs_second

    def test_all_dimensions_share_one_pit_computation(self, monkeypatch):
        y_true, y_samples, labels = self._fixture()
        row_ids = pd.DataFrame(
            {
                "group": labels,
                "n_reviews": np.arange(len(y_true)) + 1,
                "train_history": np.arange(len(y_true)) % 12,
            }
        )
        calls = {"n": 0}
        real = slices_mod.compute_pit_per_row

        def counted(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(slices_mod, "compute_pit_per_row", counted)
        calibration_by_slice(y_true, y_samples, row_ids, (0.8,), seed=31)

        assert calls["n"] == 1

    def test_slice_result_is_reproducible(self):
        y_true, y_samples, labels = self._fixture()
        a = coverage_by_slice(y_true, y_samples, labels, (0.8,), dimension="group", seed=9)
        b = coverage_by_slice(y_true, y_samples, labels, (0.8,), dimension="group", seed=9)
        assert [s.pit_max_abs_dev for s in a] == [s.pit_max_abs_dev for s in b]


class TestConformalProvenance:
    def _cal(self, seed, n=400, n_draws=200):
        rng = np.random.default_rng(seed)
        y = rng.normal(70.0, 5.0, size=n)
        samples = rng.normal(loc=y, scale=5.0, size=(n_draws, n))
        return y, samples

    def test_seed_is_recorded_and_reproducible(self):
        y_cal, cal_samples = self._cal(1)
        y_test, test_samples = self._cal(2)
        first = conformalize(y_cal, cal_samples, y_test, test_samples, (0.9,), seed=13)
        second = conformalize(y_cal, cal_samples, y_test, test_samples, (0.9,), seed=13)

        assert first["pit_method"] == PIT_METHOD
        assert first["pit_randomization_seed"] == 13
        assert first == second

    def test_recalibration_map_moves_with_the_seed(self):
        y_cal, cal_samples = self._cal(1)
        y_test, test_samples = self._cal(2)
        first = conformalize(y_cal, cal_samples, y_test, test_samples, (0.9,), seed=13)
        other = conformalize(y_cal, cal_samples, y_test, test_samples, (0.9,), seed=14)
        assert first["pit_quantile_grid"]["values"] != other["pit_quantile_grid"]["values"]

    def test_recalibration_still_recovers_coverage(self):
        # Predictive too narrow in both cal and test: recalibration must still
        # pull coverage up toward nominal under the randomized PIT.
        rng = np.random.default_rng(21)
        n, n_draws = 800, 300
        y_cal = rng.normal(70.0, 5.0, size=n)
        cal_samples = rng.normal(loc=y_cal, scale=3.5, size=(n_draws, n))
        y_test = rng.normal(70.0, 5.0, size=n)
        test_samples = rng.normal(loc=y_test, scale=3.5, size=(n_draws, n))
        block = conformalize(y_cal, cal_samples, y_test, test_samples, (0.9,), seed=4)
        assert block["levels"]["0.90"]["recalibrated_coverage"] >= 0.85


class TestPipelinePayload:
    def test_payload_pit_matches_the_seeded_computation(self):
        from panelcast.pipelines.evaluate import _evaluate_predictions

        rng = np.random.default_rng(0)
        y_true = np.array([70.0, 80.0, 60.0], dtype=np.float32)
        y_samples = rng.normal(loc=y_true, scale=5.0, size=(200, 3))
        row_ids = pd.DataFrame(
            {
                "entity": ["A", "A", "B"],
                "event": ["a1", "a2", "b1"],
                "n_reviews": [10, 20, 30],
                "train_history": [2, 2, 0],
            }
        )

        metrics, payload, _ = _evaluate_predictions(
            y_true,
            y_samples,
            calibration_intervals=(0.8,),
            coverage_tolerance=0.05,
            prediction_interval=0.8,
            row_ids=row_ids,
            pit_seed=55,
        )

        expected = compute_pit_per_row(np.asarray(y_true, dtype=float), y_samples, seed=55)
        np.testing.assert_array_equal(np.asarray(payload["pit"]), expected)
        assert payload["pit_randomization_seed"] == 55
        # The histogram and the per-row column come from the same draw.
        assert metrics["calibration"]["pit"]["randomization_seed"] == 55
        assert metrics["calibration"]["pit"] == summarize_pit(expected, seed=55)
