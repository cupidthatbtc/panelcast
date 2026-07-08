"""Per-machine calibration store: accumulation, refit, never-under envelope."""

from __future__ import annotations

import json

import numpy as np
import pytest

from panelcast.gpu_memory.calibration_store import (
    _MIN_LOCAL_ENVELOPE,
    _linear_terms,
    append_record,
    estimate_with_calibration,
    load_records,
    refit_constants,
    resolve_calibration,
)
from panelcast.gpu_memory.estimate import (
    COLLECTION_OVERHEAD_FACTOR,
    FIXED_OVERHEAD_GB,
    estimate_memory_gb,
)


def _inputs(num_samples: int = 500, num_chains: int = 2, n_obs: int = 4000) -> dict:
    return {
        "n_observations": n_obs,
        "n_features": 40,
        "n_artists": 900,
        "max_seq": 30,
        "num_chains": num_chains,
        "num_samples": num_samples,
        "num_warmup": num_samples,
        "exclude_rw_raw_from_collection": False,
    }


def _synth_records(factor: float, fixed: float, n: int = 8) -> list[dict]:
    """Records whose actual peaks follow the estimator formula exactly."""
    records = []
    for i in range(n):
        inputs = _inputs(num_samples=100 + 150 * i)
        est = estimate_memory_gb(
            collection_overhead_factor=factor, fixed_overhead_gb=fixed, **inputs
        )
        records.append(
            {
                "estimate_inputs": inputs,
                "expected_gb": est.total_gb,
                "actual_peak_gb": est.total_gb,
                "wall_clock_seconds": 600.0 + 60 * i,
                "context": {"transform": "offset_logit"},
            }
        )
    return records


class TestStore:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "cal.json"
        append_record(_inputs(), 8.3, 7.4, 3000.0, {"transform": "offset_logit"}, path=path)
        records = load_records(path)
        assert len(records) == 1
        assert records[0]["actual_peak_gb"] == 7.4
        assert records[0]["context"]["transform"] == "offset_logit"

    def test_corrupt_store_tolerated(self, tmp_path):
        path = tmp_path / "cal.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_records(path) == []
        append_record(_inputs(), 8.3, 7.4, 3000.0, path=path)
        assert len(load_records(path)) == 1

    def test_non_object_json_store_treated_as_corrupt(self, tmp_path):
        """Valid JSON that isn't an object ([], null, string) must not raise
        out of append_record — telemetry never breaks a fit."""
        for content in ('[{"records": []}]', "null", '"records"'):
            path = tmp_path / "cal.json"
            path.write_text(content, encoding="utf-8")
            assert load_records(path) == []
            append_record(_inputs(), 8.3, 7.4, 3000.0, path=path)
            assert len(load_records(path)) == 1
            path.unlink()

    def test_cap_keeps_most_recent(self, tmp_path):
        path = tmp_path / "cal.json"
        for i in range(205):
            append_record(_inputs(num_samples=10 + i), 1.0, 1.0, 1.0, path=path)
        records = load_records(path)
        assert len(records) == 200
        assert records[-1]["estimate_inputs"]["num_samples"] == 10 + 204


class TestRefit:
    def test_recovers_planted_constants(self):
        planted_factor, planted_fixed = 2.2, 0.18
        cal = refit_constants(_synth_records(planted_factor, planted_fixed))
        assert cal is not None
        # Exact-formula records: the regression recovers the constants, then
        # the envelope inflation lifts them by ~_MIN_LOCAL_ENVELOPE.
        assert cal.collection_overhead_factor == pytest.approx(
            planted_factor * _MIN_LOCAL_ENVELOPE, rel=1e-2
        )
        assert cal.min_ratio >= _MIN_LOCAL_ENVELOPE

    def test_never_under_on_every_local_point(self):
        rng = np.random.default_rng(0)
        records = _synth_records(2.5, 0.2, n=10)
        for r in records:
            r["actual_peak_gb"] *= float(rng.uniform(0.85, 1.15))
        cal = refit_constants(records)
        assert cal is not None
        for r in records:
            est = estimate_memory_gb(
                collection_overhead_factor=cal.collection_overhead_factor,
                fixed_overhead_gb=cal.fixed_overhead_gb,
                **r["estimate_inputs"],
            )
            assert est.total_gb >= r["actual_peak_gb"] * _MIN_LOCAL_ENVELOPE * (1 - 1e-9)

    def test_too_few_points_returns_none(self):
        assert refit_constants(_synth_records(3.0, 0.25, n=3)) is None

    def test_degenerate_design_returns_none(self):
        records = _synth_records(3.0, 0.25, n=1) * 6
        assert refit_constants(records) is None

    def test_records_without_actual_ignored(self):
        records = _synth_records(3.0, 0.25, n=8)
        for r in records[:5]:
            r["actual_peak_gb"] = None
        assert refit_constants(records) is None


class TestResolve:
    def test_cold_start_uses_shipped_constants(self, tmp_path):
        factor, fixed, source = resolve_calibration(tmp_path / "missing.json")
        assert factor == COLLECTION_OVERHEAD_FACTOR
        assert fixed == FIXED_OVERHEAD_GB
        assert "shipped" in source

    def test_history_earns_per_machine_source(self, tmp_path):
        path = tmp_path / "cal.json"
        payload = {"version": 1, "records": _synth_records(2.0, 0.15)}
        path.write_text(json.dumps(payload), encoding="utf-8")
        factor, fixed, source = resolve_calibration(path)
        assert "per-machine" in source
        assert factor < COLLECTION_OVERHEAD_FACTOR  # tighter than shipped on this synthetic machine

    def test_estimate_with_calibration_reports_source(self, tmp_path):
        estimate, source = estimate_with_calibration(tmp_path / "missing.json", **_inputs())
        assert estimate.total_gb > 0
        assert "shipped" in source


class TestAppendRobustness:
    """A transient read failure must not clobber the accumulated store."""

    def _patch_read_text(self, monkeypatch, target, fail_times):
        from pathlib import Path

        real_read_text = Path.read_text
        calls = {"n": 0}

        def flaky_read_text(self, *args, **kwargs):
            if self == target:
                calls["n"] += 1
                if calls["n"] <= fail_times:
                    raise PermissionError("sharing violation")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky_read_text)
        return calls

    def test_missing_store_created_with_one_record(self, tmp_path):
        path = tmp_path / "cal.json"
        append_record(_inputs(), 8.3, 7.4, 3000.0, path=path)
        assert len(load_records(path)) == 1

    def test_unreadable_store_skips_append(self, tmp_path, monkeypatch):
        import panelcast.gpu_memory.calibration_store as store_mod

        path = tmp_path / "cal.json"
        append_record(_inputs(), 8.3, 7.4, 3000.0, path=path)
        before = path.read_bytes()

        monkeypatch.setattr(store_mod.time, "sleep", lambda _s: None)
        self._patch_read_text(monkeypatch, path, fail_times=99)
        append_record(_inputs(num_samples=999), 1.0, 1.0, 1.0, path=path)

        monkeypatch.undo()
        assert path.read_bytes() == before  # store intact, nothing rewritten
        assert len(load_records(path)) == 1

    def test_transient_failure_retries_then_appends(self, tmp_path, monkeypatch):
        import panelcast.gpu_memory.calibration_store as store_mod

        path = tmp_path / "cal.json"
        append_record(_inputs(), 8.3, 7.4, 3000.0, path=path)

        monkeypatch.setattr(store_mod.time, "sleep", lambda _s: None)
        self._patch_read_text(monkeypatch, path, fail_times=2)
        append_record(_inputs(num_samples=999), 1.0, 1.0, 1.0, path=path)

        monkeypatch.undo()
        records = load_records(path)
        assert len(records) == 2
        assert records[-1]["estimate_inputs"]["num_samples"] == 999


class TestLinearTermsGateFlags:
    """_linear_terms folds structural gate flags into the fit terms."""

    def test_eiv_flag_grows_collection_unit(self):
        base_inputs = _inputs()
        eiv_inputs = {**base_inputs, "errors_in_variables": True}
        base_terms = _linear_terms(base_inputs)
        eiv_terms = _linear_terms(eiv_inputs)
        assert eiv_terms[1] > base_terms[1]  # n_obs latent collected per draw
        assert eiv_terms[0] > base_terms[0]  # and a parameter either way

    def test_eiv_with_exclusion_leaves_unit_unchanged(self):
        excl = {**_inputs(), "exclude_rw_raw_from_collection": True}
        both = {**excl, "errors_in_variables": True}
        assert _linear_terms(both)[1] == pytest.approx(_linear_terms(excl)[1])
        assert _linear_terms(both)[0] > _linear_terms(excl)[0]

    def test_refit_recovers_constants_from_gated_records(self):
        """Records fit under EIV refit cleanly: the gate is in the terms, so
        the recovered constants match the planted ones (a formula that
        ignored the flag would bias the factor by the missing n_obs term)."""
        planted_factor, planted_fixed = 2.2, 0.18
        records = []
        for i in range(8):
            inputs = {**_inputs(num_samples=100 + 150 * i), "errors_in_variables": True}
            est = estimate_memory_gb(
                collection_overhead_factor=planted_factor,
                fixed_overhead_gb=planted_fixed,
                **inputs,
            )
            records.append(
                {
                    "estimate_inputs": inputs,
                    "expected_gb": est.total_gb,
                    "actual_peak_gb": est.total_gb,
                    "wall_clock_seconds": 600.0,
                    "context": {},
                }
            )
        cal = refit_constants(records)
        assert cal is not None
        assert cal.collection_overhead_factor == pytest.approx(
            planted_factor * _MIN_LOCAL_ENVELOPE, rel=1e-2
        )
