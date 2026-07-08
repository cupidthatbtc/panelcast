"""Sliced calibration audit (#181): Wilson CIs, slice flags, history parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.evaluation.slices import (
    HISTORY_BINS,
    calibration_by_slice,
    coverage_by_slice,
    history_bin_labels,
    stratified_history_metrics,
    wilson_interval,
)


class TestWilsonInterval:
    def test_matches_known_value(self):
        # k=80, n=100 at 95%: classic textbook Wilson bounds.
        lo, hi = wilson_interval(80, 100)
        assert lo == pytest.approx(0.7112, abs=1e-3)
        assert hi == pytest.approx(0.8661, abs=1e-3)

    def test_degenerate_inputs(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)
        lo, hi = wilson_interval(0, 50)
        assert lo == 0.0 and hi < 0.15
        lo, hi = wilson_interval(50, 50)
        assert hi == 1.0 and lo > 0.85

    def test_narrows_with_n(self):
        lo_s, hi_s = wilson_interval(8, 10)
        lo_l, hi_l = wilson_interval(800, 1000)
        assert (hi_l - lo_l) < (hi_s - lo_s)


class TestHistoryBinLabels:
    def test_bins_match_shared_definition(self):
        labels = history_bin_labels(np.array([0, 1, 2, 3, 5, 6, 10, 11, 40]))
        assert labels.tolist() == [
            "0", "1-2", "1-2", "3-5", "3-5", "6-10", "6-10", "11+", "11+",
        ]
        assert HISTORY_BINS == ((1, 2), (3, 5), (6, 10), (11, None))


class TestCoverageBySlice:
    def _samples(self, rng, y_true, scale=5.0, n_draws=400):
        return rng.normal(loc=y_true, scale=scale, size=(n_draws, len(y_true)))

    def test_min_n_floor_drops_rare_slices(self):
        rng = np.random.default_rng(0)
        y = rng.normal(70, 5, size=60)
        samples = self._samples(rng, y)
        labels = np.array(["big"] * 55 + ["rare"] * 5, dtype=object)
        out = coverage_by_slice(y, samples, labels, (0.8,), dimension="group", min_n=20)
        assert [s.label for s in out] == ["big"]
        assert out[0].n == 55

    def test_gross_miscalibration_is_flagged(self):
        rng = np.random.default_rng(1)
        y = rng.normal(70, 5, size=200)
        # Predictive far too narrow around a fixed center: nominal 0.95
        # nowhere near empirical.
        samples = rng.normal(70.0, 0.5, size=(400, 200))
        out = coverage_by_slice(
            y, samples, np.array(["all"] * 200, dtype=object), (0.95,), dimension="group"
        )
        lv = out[0].levels["0.95"]
        assert lv["flagged"]
        assert lv["empirical"] < 0.5

    def test_calibrated_slice_not_flagged(self):
        rng = np.random.default_rng(2)
        y = rng.normal(70, 5, size=500)
        samples = rng.normal(70, 5, size=(500, 500))
        out = coverage_by_slice(
            y, samples, np.array(["all"] * 500, dtype=object), (0.8,), dimension="group"
        )
        assert not out[0].levels["0.80"]["flagged"]

    def test_label_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="labels length"):
            coverage_by_slice(
                np.zeros(3), np.zeros((10, 3)), np.array(["a"]), (0.8,), dimension="x"
            )


class TestCalibrationBySlice:
    def test_dimensions_and_false_flag_note(self):
        rng = np.random.default_rng(3)
        n = 240
        y = rng.normal(70, 5, size=n)
        samples = rng.normal(70, 5, size=(300, n))
        row_ids = pd.DataFrame(
            {
                "entity": [f"e{i % 40}" for i in range(n)],
                "group": np.where(np.arange(n) % 2 == 0, "rock", "pop"),
                "n_reviews": rng.integers(1, 500, size=n),
                "train_history": rng.integers(0, 20, size=n),
            }
        )
        out = calibration_by_slice(y, samples, row_ids, (0.8, 0.95))
        dims = {s["dimension"] for s in out["slices"]}
        assert {"group", "n_reviews_decile", "train_history", "target_tercile"} <= dims
        assert out["expected_false_flags"] == pytest.approx(0.05 * out["n_tests"], abs=0.01)
        assert "false" in out["note"].lower() or "chance" in out["note"]

    def test_without_row_ids_only_target_terciles(self):
        rng = np.random.default_rng(4)
        y = rng.normal(70, 5, size=90)
        samples = rng.normal(70, 5, size=(200, 90))
        out = calibration_by_slice(y, samples, None, (0.8,))
        assert {s["dimension"] for s in out["slices"]} == {"target_tercile"}


class TestStratifiedHistoryMetrics:
    def test_same_schema_and_binning_as_legacy(self):
        rng = np.random.default_rng(5)
        y = rng.normal(70, 5, size=100)
        samples = rng.normal(70, 5, size=(200, 100))
        history = np.array([1] * 30 + [4] * 30 + [8] * 20 + [15] * 15 + [0] * 5)
        rows = stratified_history_metrics(y, samples, history, interval=0.8)
        assert [r["train_albums_bin"] for r in rows] == ["1-2", "3-5", "6-10", "11+"]
        # History-0 rows fall outside every bin, exactly as before.
        assert sum(r["n"] for r in rows) == 95
        for r in rows:
            assert set(r) == {
                "train_albums_bin", "n", "rmse", "r2", "coverage",
                "mean_interval_width", "interval",
            }
