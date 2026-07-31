"""A non-finite prediction is a numerical failure, not a missing metric.

``compute_point_metrics`` rejected NaN but not infinity, and ``compute_crps``
validated only shapes. An infinite prediction therefore produced infinite
MAE/RMSE/bias, which ``_json_safe`` wrote as ``null`` under ``allow_nan=False``
while the evaluation stage still reported success -- predictive overflow read as
metrics that simply had not been computed.

Reaching this takes an identity-scale blow-up: the offset-logit inverse is
sigmoid-bounded, so the default path cannot produce an infinite prediction. The
point is that when it does happen the signal survives.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import structlog.testing

from panelcast.evaluation import NonFinitePredictionError, require_finite
from panelcast.evaluation.calibration import (
    compute_coverage,
    compute_interval_score,
    compute_multi_coverage,
    compute_pit_per_row,
    compute_reliability_data,
    compute_weighted_interval_score,
)
from panelcast.evaluation.metrics import compute_crps, compute_point_metrics
from panelcast.evaluation.ppc import compute_ppc_statistics
from panelcast.evaluation.ranking import compute_ranking_metrics
from panelcast.pipelines.evaluate import _evaluate_predictions, _json_safe, _write_json


@pytest.fixture
def finite_inputs():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(50.0, 90.0, 8)
    y_samples = y_true[None, :] + rng.normal(0.0, 5.0, (40, 8))
    return y_true, y_samples


def _with_bad_value(array: np.ndarray, value: float) -> np.ndarray:
    spoiled = np.array(array, dtype=float, copy=True)
    spoiled.reshape(-1)[0] = value
    return spoiled


BAD_VALUES = [np.inf, -np.inf, np.nan]


class TestRequireFinite:
    def test_finite_input_passes_through(self):
        values = np.array([1.0, 2.0])
        np.testing.assert_array_equal(require_finite(values, "y_true"), values)

    @pytest.mark.parametrize("value", BAD_VALUES)
    def test_rejects_every_non_finite_value(self, value):
        with pytest.raises(NonFinitePredictionError, match="y_true"):
            require_finite(np.array([1.0, value]), "y_true")

    def test_reports_the_offending_rows(self):
        with pytest.raises(NonFinitePredictionError, match=r"rows \[1, 3\]"):
            require_finite(np.array([1.0, np.inf, 2.0, np.nan]), "y_true")

    def test_reports_draw_and_row_counts_for_samples(self):
        samples = np.ones((5, 4))
        samples[2, 1] = np.inf
        with pytest.raises(NonFinitePredictionError, match="1 of 5 draws, 1 of 4 rows"):
            require_finite(samples, "y_samples")

    def test_is_a_value_error_so_existing_handlers_still_catch_it(self):
        assert issubclass(NonFinitePredictionError, ValueError)


class TestPointMetrics:
    @pytest.mark.parametrize("value", BAD_VALUES)
    def test_infinite_prediction_raises_instead_of_returning_infinite_metrics(
        self, value, finite_inputs
    ):
        """The reported defect: y_pred_mean=[inf, 2] gave infinite MAE/RMSE/bias."""
        y_true, _ = finite_inputs
        y_pred = _with_bad_value(y_true, value)
        with pytest.raises(NonFinitePredictionError, match="y_pred_mean"):
            compute_point_metrics(y_true, y_pred)

    @pytest.mark.parametrize("value", BAD_VALUES)
    def test_non_finite_truth_raises(self, value, finite_inputs):
        y_true, _ = finite_inputs
        with pytest.raises(NonFinitePredictionError, match="y_true"):
            compute_point_metrics(_with_bad_value(y_true, value), y_true)


class TestCrps:
    @pytest.mark.parametrize("value", BAD_VALUES)
    def test_non_finite_samples_raise(self, value, finite_inputs):
        """compute_crps validated shapes only."""
        y_true, y_samples = finite_inputs
        with pytest.raises(NonFinitePredictionError, match="y_samples"):
            compute_crps(y_true, _with_bad_value(y_samples, value))

    def test_non_finite_truth_raises(self, finite_inputs):
        y_true, y_samples = finite_inputs
        with pytest.raises(NonFinitePredictionError, match="y_true"):
            compute_crps(_with_bad_value(y_true, np.inf), y_samples)


class TestDownstreamCalculations:
    """Calibration, PPC and ranking see the same guard."""

    def test_coverage(self, finite_inputs):
        y_true, y_samples = finite_inputs
        with pytest.raises(NonFinitePredictionError):
            compute_coverage(y_true, _with_bad_value(y_samples, np.inf))

    def test_multi_coverage(self, finite_inputs):
        y_true, y_samples = finite_inputs
        with pytest.raises(NonFinitePredictionError):
            compute_multi_coverage(y_true, _with_bad_value(y_samples, np.inf))

    def test_interval_score(self, finite_inputs):
        y_true, y_samples = finite_inputs
        with pytest.raises(NonFinitePredictionError):
            compute_interval_score(y_true, _with_bad_value(y_samples, np.inf))

    def test_weighted_interval_score(self, finite_inputs):
        y_true, y_samples = finite_inputs
        with pytest.raises(NonFinitePredictionError):
            compute_weighted_interval_score(y_true, _with_bad_value(y_samples, np.inf))

    def test_reliability_data(self, finite_inputs):
        y_true, y_samples = finite_inputs
        with pytest.raises(NonFinitePredictionError):
            compute_reliability_data(y_true, _with_bad_value(y_samples, np.inf))

    def test_pit(self, finite_inputs):
        y_true, y_samples = finite_inputs
        with pytest.raises(NonFinitePredictionError):
            compute_pit_per_row(y_true, _with_bad_value(y_samples, np.inf), seed=0)

    def test_ppc(self, finite_inputs):
        y_true, y_samples = finite_inputs
        with pytest.raises(NonFinitePredictionError):
            compute_ppc_statistics(y_true, _with_bad_value(y_samples, np.inf))

    def test_ranking(self, finite_inputs):
        y_true, y_samples = finite_inputs
        with pytest.raises(NonFinitePredictionError):
            compute_ranking_metrics(y_true, _with_bad_value(y_samples, np.inf))

    def test_finite_inputs_are_unaffected(self, finite_inputs):
        """The guard must not move any number on the path that matters."""
        y_true, y_samples = finite_inputs
        point = compute_point_metrics(y_true, y_samples.mean(axis=0))
        assert np.isfinite(point.mae)
        assert np.isfinite(compute_crps(y_true, y_samples).mean_crps)
        assert np.isfinite(compute_coverage(y_true, y_samples, prob=0.8).empirical)


class TestSplitEvaluation:
    """The stage-level payload builder, which is where the nulls used to land."""

    def _evaluate(self, y_true, y_samples):
        return _evaluate_predictions(
            y_true,
            y_samples,
            calibration_intervals=(0.5, 0.95),
            coverage_tolerance=0.05,
            prediction_interval=0.95,
            pit_seed=0,
        )

    def test_a_finite_split_still_evaluates(self, finite_inputs):
        y_true, y_samples = finite_inputs
        metrics, _, _ = self._evaluate(y_true, y_samples)
        assert np.isfinite(metrics["point_metrics"]["mae"])

    def test_an_infinite_draw_fails_the_split_instead_of_nulling_it(self, finite_inputs):
        """Previously: infinite MAE/RMSE/bias, nulled by _json_safe, stage green."""
        y_true, y_samples = finite_inputs
        with pytest.raises(NonFinitePredictionError):
            self._evaluate(y_true, _with_bad_value(y_samples, np.inf))


class TestJsonSerialization:
    def test_non_finite_floats_are_still_nulled(self):
        assert _json_safe({"mae": float("inf")}) == {"mae": None}

    def test_the_nulled_fields_are_reported(self):
        nulled: list[str] = []
        _json_safe({"a": {"b": float("nan")}, "c": [1.0, float("inf")]}, nulled=nulled)
        assert nulled == ["a.b", "c[1]"]

    def test_a_clean_payload_reports_nothing(self):
        nulled: list[str] = []
        _json_safe({"a": 1.0, "b": [2.0, 3.0], "c": None}, nulled=nulled)
        assert nulled == []

    def _warnings(self, monkeypatch, path, payload) -> list[dict]:
        """Warnings _write_json emits, via a per-test logger.

        Not ``capture_logs``: with ``cache_logger_on_first_use`` the module's
        lazy proxy binds to whichever context first uses it, so capturing here
        would steal the binding from every later test in the session.
        """
        from panelcast.pipelines import evaluate as evaluate_module

        capturing = structlog.testing.CapturingLogger()
        monkeypatch.setattr(evaluate_module, "log", capturing)
        _write_json(path, payload)
        return [call.kwargs for call in capturing.calls if call.method_name == "warning"]

    def test_write_json_warns_which_fields_were_nulled(self, tmp_path, monkeypatch):
        """An absent key means "not computed"; a null means it overflowed."""
        path = tmp_path / "metrics.json"
        warnings = self._warnings(
            monkeypatch, path, {"point_metrics": {"mae": float("inf")}}
        )
        assert json.loads(path.read_text())["point_metrics"]["mae"] is None
        assert len(warnings) == 1
        assert warnings[0]["fields"] == ["point_metrics.mae"]
        assert warnings[0]["artifact"] == "metrics.json"

    def test_write_json_is_quiet_on_a_finite_payload(self, tmp_path, monkeypatch):
        path = tmp_path / "metrics.json"
        assert self._warnings(monkeypatch, path, {"point_metrics": {"mae": 5.0}}) == []
