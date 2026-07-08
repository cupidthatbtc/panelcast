"""Runtime predictor: model-aware affine estimate with honest cold-start fallback."""

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

    def test_unmatched_transform_uses_cold_not_other_models(self, tmp_path):
        # Fast identity fits must not drag an offset_logit estimate low: with no
        # offset_logit history we scale offset_logit's own cold anchor.
        path = _store(tmp_path, [_record(4000.0, transform="identity")] * 3)
        pred = predict_fit_seconds(
            4, 1000, 1000, 5000, transform="offset_logit", store_path=path
        )
        assert "cold-start" in pred.source
        assert "offset_logit" in pred.source
        assert pred.seconds == pytest.approx(50 * 60.0)

    def test_thin_history_falls_back_to_cold(self, tmp_path):
        path = _store(tmp_path, [_record(4000.0)] * 2)
        pred = predict_fit_seconds(4, 1000, 1000, 5000, store_path=path)
        assert "cold-start" in pred.source


class TestModelAwareAffine:
    """#112 follow-up: an affine, transform-keyed estimate that a pool of tiny
    probe fits can't poison, and that keeps offset_logit above identity."""

    def _seeded_store(self, tmp_path):
        # The shape of the real (poisoned) store: several leak-style probes
        # (tiny, no transform), one real offset_logit fit, several identity fits.
        records = (
            [_record(12.5, num_samples=5, n_obs=100, transform=None)] * 6
            + [_record(5788.0, num_samples=1000, n_obs=4235)]
            + [_record(560.0, num_samples=1000, n_obs=4235, transform="identity")] * 3
        )
        return _store(tmp_path, records)

    def test_offset_logit_publication_not_probe_poisoned(self, tmp_path):
        # The median-ratio model returned ~29h here (probe rate x real size);
        # the affine per-transform estimate stays near the true ~1-8h.
        path = self._seeded_store(tmp_path)
        pub = predict_fit_seconds(
            4, 5000, 5000, 4235, transform="offset_logit", store_path=path
        )
        assert pub.hours < 10

    def test_offset_logit_costs_far_more_than_identity(self, tmp_path):
        path = self._seeded_store(tmp_path)
        off = predict_fit_seconds(
            4, 1000, 1000, 4235, transform="offset_logit", store_path=path
        )
        ident = predict_fit_seconds(
            4, 1000, 1000, 4235, transform="identity", store_path=path
        )
        assert off.seconds > 5 * ident.seconds

    def test_thin_history_probe_does_not_skew_rate(self, tmp_path):
        # <5 records so FIXED collapses to 0; the local-history branch must still
        # ignore a leak-style probe (no transform) rather than let its ~40x rate
        # inflate the offset_logit estimate.
        records = [
            _record(12.5, num_samples=5, n_obs=100, transform=None),
            _record(5788.0, num_samples=1000, n_obs=4235),
        ]
        path = _store(tmp_path, records)
        pred = predict_fit_seconds(
            4, 1000, 1000, 4235, transform="offset_logit", store_path=path
        )
        assert "local history (offset_logit)" in pred.source
        assert pred.hours == pytest.approx(5788.0 / 3600.0, abs=0.05)


class TestChainMethodKeying:
    """Vectorized wall-clocks must not corrupt sequential rates (#176)."""

    def _tagged(self, seconds: float, chain_method: str | None):
        record = _record(seconds)
        if chain_method is not None:
            record["context"]["chain_method"] = chain_method
        return record

    def test_vectorized_records_excluded_from_sequential(self, tmp_path):
        path = _store(
            tmp_path,
            [self._tagged(1000.0, "vectorized")] * 3 + [self._tagged(4000.0, "sequential")] * 3,
        )
        pred = predict_fit_seconds(4, 1000, 1000, 5000, transform="offset_logit", store_path=path)
        assert pred.seconds == pytest.approx(4000.0)

    def test_untagged_records_count_as_sequential(self, tmp_path):
        path = _store(tmp_path, [self._tagged(4000.0, None)] * 3)
        pred = predict_fit_seconds(4, 1000, 1000, 5000, transform="offset_logit", store_path=path)
        assert "local history" in pred.source
        assert pred.seconds == pytest.approx(4000.0)

    def test_vectorized_prediction_uses_vectorized_history(self, tmp_path):
        path = _store(
            tmp_path,
            [self._tagged(1000.0, "vectorized")] * 3 + [self._tagged(4000.0, None)] * 3,
        )
        pred = predict_fit_seconds(
            4, 1000, 1000, 5000, transform="offset_logit", store_path=path,
            chain_method="vectorized",
        )
        assert pred.seconds == pytest.approx(1000.0)

    def test_vectorized_cold_start_falls_back_to_anchor(self, tmp_path):
        path = _store(tmp_path, [self._tagged(4000.0, "sequential")] * 3)
        pred = predict_fit_seconds(
            4, 1000, 1000, 5000, transform="offset_logit", store_path=path,
            chain_method="vectorized",
        )
        assert "cold-start" in pred.source
