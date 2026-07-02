"""Unit tests for the baseline predictors and benchmark harness."""

from __future__ import annotations

import numpy as np
import pytest

from panelcast.models.baselines import (
    ConformalGBMBaseline,
    EntityMeanBaseline,
    GlobalMeanBaseline,
    LastScoreBaseline,
    PanelData,
    RidgeBaseline,
    benchmark_baselines,
    build_default_baselines,
)
from panelcast.reporting.tables import create_baseline_benchmark_table


def _panels(n_train: int = 200, n_test: int = 60, p: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    bounds = (0.0, 100.0)

    def make(n, entities):
        X = rng.standard_normal((n, p))
        entity = rng.choice(entities, size=n)
        # Signal: linear in X plus an entity offset, clipped to bounds.
        y = 60 + 5 * X[:, 0] - 3 * X[:, 1] + rng.normal(0, 4, n)
        y = np.clip(y, *bounds)
        return PanelData(X=X, y=y, entity=entity, prev_score=None, bounds=bounds)

    train = make(n_train, [f"E{i}" for i in range(20)])
    test = make(n_test, [f"E{i}" for i in range(20)])
    return train, test


class TestPredictors:
    def test_shapes_and_bounds(self):
        train, test = _panels()
        rng = np.random.default_rng(1)
        for bl in build_default_baselines(train.bounds):
            bl.fit(train)
            pred = bl.predict(test, n_samples=50, rng=rng)
            assert pred.point.shape == (test.y.shape[0],)
            assert pred.samples.shape == (50, test.y.shape[0])
            assert np.all(pred.point >= train.bounds[0] - 1e-6)
            assert np.all(pred.point <= train.bounds[1] + 1e-6)
            assert np.all(pred.samples >= train.bounds[0] - 1e-6)
            assert np.all(pred.samples <= train.bounds[1] + 1e-6)

    def test_global_mean_is_constant(self):
        train, test = _panels()
        bl = GlobalMeanBaseline(train.bounds).fit(train)
        point = bl.predict(test, n_samples=10, rng=np.random.default_rng(0)).point
        assert np.allclose(point, point[0])
        assert point[0] == pytest.approx(float(np.mean(train.y)), abs=1e-6)

    def test_entity_mean_unseen_falls_back_to_global(self):
        train, _ = _panels()
        bl = EntityMeanBaseline(train.bounds).fit(train)
        unseen = PanelData(
            X=np.zeros((1, train.X.shape[1])),
            y=np.array([np.nan]),
            entity=np.array(["UNSEEN"]),
            bounds=train.bounds,
        )
        point = bl.predict(unseen, n_samples=5, rng=np.random.default_rng(0)).point
        assert point[0] == pytest.approx(bl._global, abs=1e-6)

    def test_ridge_beats_global_mean_on_signal(self):
        train, test = _panels()
        rng = np.random.default_rng(2)
        ridge = RidgeBaseline(train.bounds).fit(train)
        gmean = GlobalMeanBaseline(train.bounds).fit(train)
        ridge_mae = np.mean(np.abs(ridge.predict(test, 100, rng).point - test.y))
        gmean_mae = np.mean(np.abs(gmean.predict(test, 100, rng).point - test.y))
        assert ridge_mae < gmean_mae

    def test_last_score_uses_prev_when_present(self):
        train, _ = _panels()
        bl = LastScoreBaseline(train.bounds).fit(train)
        test = PanelData(
            X=np.zeros((3, train.X.shape[1])),
            y=np.array([70.0, 80.0, 90.0]),
            entity=np.array(["E0", "E1", "UNSEEN"]),
            prev_score=np.array([71.0, 79.0, np.nan]),
            bounds=train.bounds,
        )
        point = bl.predict(test, n_samples=5, rng=np.random.default_rng(0)).point
        assert point[0] == pytest.approx(71.0)
        assert point[1] == pytest.approx(79.0)
        # NaN prev for unseen entity falls back (global or entity mean), finite.
        assert np.isfinite(point[2])


class TestBenchmark:
    def test_benchmark_rows_and_table(self):
        train, test = _panels()
        scores = benchmark_baselines(train, test, split="within_entity_temporal", n_samples=200)
        assert len(scores) == 6
        levels = (0.80, 0.95)
        rows = [s.to_row(levels) for s in scores]
        for row in rows:
            assert np.isfinite(row["mae"]) and row["mae"] >= 0
            assert np.isfinite(row["crps"]) and row["crps"] >= 0
            assert 0.0 <= row["cov80"] <= 1.0
            assert 0.0 <= row["cov95"] <= 1.0

        table = create_baseline_benchmark_table(rows, levels=levels)
        assert len(table) == 6
        # No TBD / empty cells: every cell is a non-empty string.
        assert not (table.astype(str) == "").any().any()
        assert "—" not in table["MAE"].tolist()

    def test_masked_test_targets_dropped(self):
        train, test = _panels(n_test=40)
        # Mask half the test targets (held-out labels).
        test.y[::2] = np.nan
        scores = benchmark_baselines(train, test, split="entity_disjoint", n_samples=100)
        assert all(s.n_obs == 20 for s in scores)

    def test_empty_table_is_columned_not_raising(self):
        table = create_baseline_benchmark_table([], levels=(0.80, 0.95))
        assert table.empty
        assert "Model" in table.columns


class TestConformalGBM:
    def test_coverage_at_nominal_on_exchangeable_data(self):
        train, test = _panels(n_train=2000, n_test=800, seed=3)
        bl = ConformalGBMBaseline(train.bounds).fit(train)
        pred = bl.predict(test, n_samples=2000, rng=np.random.default_rng(4))
        lo, hi = np.percentile(pred.samples, [2.5, 97.5], axis=0)
        cov95 = np.mean((test.y >= lo) & (test.y <= hi))
        assert 0.93 <= cov95 <= 0.995
        lo80, hi80 = np.percentile(pred.samples, [10.0, 90.0], axis=0)
        cov80 = np.mean((test.y >= lo80) & (test.y <= hi80))
        assert 0.76 <= cov80 <= 0.92

    def test_deterministic_given_seeds(self):
        train, test = _panels(seed=5)
        a = ConformalGBMBaseline(train.bounds).fit(train)
        b = ConformalGBMBaseline(train.bounds).fit(train)
        sa = a.predict(test, n_samples=50, rng=np.random.default_rng(9)).samples
        sb = b.predict(test, n_samples=50, rng=np.random.default_rng(9)).samples
        np.testing.assert_array_equal(sa, sb)

    def test_intervals_come_from_calibration_residuals(self):
        # The predictive spread must track the calibration residual pool, not
        # a Gaussian wrapper: every sampled deviation (before clipping) is an
        # element of the pool.
        train, test = _panels(n_train=400, n_test=10, seed=7)
        bl = ConformalGBMBaseline(train.bounds).fit(train)
        pred = bl.predict(test, n_samples=200, rng=np.random.default_rng(2))
        interior = (pred.samples > train.bounds[0]) & (pred.samples < train.bounds[1])
        deviations = (pred.samples - pred.point[None, :])[interior]
        pool = np.sort(bl._residual_pool)
        idx = np.clip(np.searchsorted(pool, deviations), 0, pool.size - 1)
        nearest = np.minimum(
            np.abs(pool[idx] - deviations),
            np.abs(pool[np.maximum(idx - 1, 0)] - deviations),
        )
        assert np.max(nearest) < 1e-9

    def test_invalid_calibration_fraction_raises(self):
        with pytest.raises(ValueError, match="calibration_fraction"):
            ConformalGBMBaseline(calibration_fraction=1.5)

    def test_tiny_train_falls_back_without_error(self):
        rng = np.random.default_rng(0)
        train = PanelData(
            X=rng.standard_normal((2, 3)),
            y=np.array([50.0, 70.0]),
            entity=np.array(["a", "b"]),
        )
        test = PanelData(
            X=rng.standard_normal((2, 3)),
            y=np.array([55.0, 65.0]),
            entity=np.array(["a", "b"]),
        )
        pred = ConformalGBMBaseline().fit(train).predict(test, 20, np.random.default_rng(1))
        assert np.all(np.isfinite(pred.samples))
        assert pred.samples.shape == (20, 2)
