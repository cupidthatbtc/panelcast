"""Runtime predictor: history-based rates with honest cold-start fallback."""

from __future__ import annotations

import json

import pytest

from panelcast.gpu_memory.runtime_predictor import predict_fit_seconds


def _record(seconds: float, num_samples: int = 1000, n_obs: int = 5000, transform="offset_logit"):
    return {
        "estimate_inputs": {
            "n_observations": n_obs,
            "n_features": 40,
            "n_artists": 900,
            "max_seq": 30,
            "num_chains": 4,
            "num_samples": num_samples,
            "num_warmup": num_samples,
        },
        "expected_gb": 8.0,
        "actual_peak_gb": 7.0,
        "wall_clock_seconds": seconds,
        "context": {"transform": transform},
    }


def _store(tmp_path, records):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"version": 1, "records": records}), encoding="utf-8")
    return path


class TestColdStart:
    def test_diagnostic_scale_matches_anchor(self, tmp_path):
        pred = predict_fit_seconds(4, 1000, 1000, 5000, store_path=tmp_path / "none.json")
        assert pred.seconds == pytest.approx(50 * 60.0)
        assert "cold-start" in pred.source

    def test_publication_scale_matches_anchor(self, tmp_path):
        pred = predict_fit_seconds(4, 5000, 5000, 5000, store_path=tmp_path / "none.json")
        assert pred.hours == pytest.approx(6.3)

    def test_scales_with_data_size(self, tmp_path):
        half = predict_fit_seconds(4, 1000, 1000, 2500, store_path=tmp_path / "none.json")
        assert half.seconds == pytest.approx(25 * 60.0)


class TestHistory:
    def test_median_rate_from_matching_transform(self, tmp_path):
        # Rate = seconds / (draws * n_obs); draws = 4 * 2000, n_obs = 5000.
        records = [_record(4000.0), _record(4000.0), _record(4000.0)]
        path = _store(tmp_path, records)
        pred = predict_fit_seconds(
            4, 1000, 1000, 5000, transform="offset_logit", store_path=path
        )
        assert pred.seconds == pytest.approx(4000.0)
        assert "local history" in pred.source

    def test_prediction_scales_linearly_from_history(self, tmp_path):
        path = _store(tmp_path, [_record(4000.0)] * 3)
        pred = predict_fit_seconds(
            8, 1000, 1000, 5000, transform="offset_logit", store_path=path
        )
        assert pred.seconds == pytest.approx(8000.0)

    def test_falls_back_to_all_records_when_transform_unmatched(self, tmp_path):
        path = _store(tmp_path, [_record(4000.0, transform="identity")] * 3)
        pred = predict_fit_seconds(
            4, 1000, 1000, 5000, transform="offset_logit", store_path=path
        )
        assert pred.seconds == pytest.approx(4000.0)
        assert "local history" in pred.source

    def test_thin_history_falls_back_to_cold(self, tmp_path):
        path = _store(tmp_path, [_record(4000.0)] * 2)
        pred = predict_fit_seconds(4, 1000, 1000, 5000, store_path=path)
        assert "cold-start" in pred.source
