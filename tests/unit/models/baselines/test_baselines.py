"""Unit tests for the baseline predictors and benchmark harness."""

from __future__ import annotations

import numpy as np
import pytest

from panelcast.models.baselines import (
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
        ridge_mae = np.mean(
            np.abs(ridge.predict(test, 100, rng).point - test.y)
        )
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
        assert len(scores) == 5
        levels = (0.80, 0.95)
        rows = [s.to_row(levels) for s in scores]
        for row in rows:
            assert np.isfinite(row["mae"]) and row["mae"] >= 0
            assert np.isfinite(row["crps"]) and row["crps"] >= 0
            assert 0.0 <= row["cov80"] <= 1.0
            assert 0.0 <= row["cov95"] <= 1.0

        table = create_baseline_benchmark_table(rows, levels=levels)
        assert len(table) == 5
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
