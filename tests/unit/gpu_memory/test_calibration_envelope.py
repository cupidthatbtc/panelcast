"""Adversarial cover for the local GPU-calibration envelope (#370).

`refit_constants` promises that a per-machine calibration never under-covers
any point it was fit on. These tests attack that promise: poisoned telemetry,
coefficients that cannot be fit, designs the old fixed-iteration inflation
could not reach, and leverage points that dominate the binding constraint.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from panelcast.gpu_memory.calibration_store import (
    _MIN_LOCAL_ENVELOPE,
    PerMachineCalibration,
    _envelope_scale,
    _partition_points,
    refit_constants,
    resolve_calibration,
)
from panelcast.gpu_memory.estimate import (
    COLLECTION_OVERHEAD_FACTOR,
    FIXED_OVERHEAD_GB,
    estimate_memory_gb,
)

ENVELOPE_SLACK = 1 - 1e-9


def _inputs(**overrides) -> dict:
    base = {
        "n_observations": 4000,
        "n_features": 40,
        "n_artists": 900,
        "max_seq": 30,
        "num_chains": 2,
        "num_samples": 500,
        "num_warmup": 500,
        "exclude_rw_raw_from_collection": False,
    }
    base.update(overrides)
    return base


def _provenance(actual: float, *, attribution: str = "fit_interval_new_process_peak") -> dict:
    peak = round(actual * 1024**3)

    def snapshot(value: int) -> dict:
        return {
            "recorded_at": "2026-07-30T00:00:00+00:00",
            "process_id": 1234,
            "thread_id": 5678,
            "device_id": 0,
            "process_index": 0,
            "platform": "gpu",
            "device_kind": "test GPU",
            "bytes_in_use": value,
            "peak_bytes_in_use": value,
            "bytes_limit": max(peak * 2, 1),
            "bytes_reserved": value,
            "num_allocs": 1,
        }

    trusted = attribution == "fit_interval_new_process_peak"
    before_peak = max(peak - 1, 0) if trusted else peak
    return {
        "version": 1,
        "source": "jax_allocator",
        "scope": "process",
        "trusted_for_calibration": trusted,
        "attribution": attribution,
        "before": snapshot(before_peak),
        "after": snapshot(peak),
    }


def _record(inputs: dict, actual: float | None) -> dict:
    return {
        "estimate_inputs": inputs,
        "expected_gb": 1.0,
        "actual_peak_gb": actual,
        "peak_provenance": _provenance(actual)
        if isinstance(actual, float) and actual > 0
        else None,
        "wall_clock_seconds": 600.0,
        "context": {},
    }


def _exact_records(factor: float, fixed: float, n: int = 8, **input_overrides) -> list[dict]:
    """Records whose peaks follow the estimator formula exactly."""
    records = []
    for i in range(n):
        inputs = _inputs(num_samples=100 + 150 * i, **input_overrides)
        est = estimate_memory_gb(
            collection_overhead_factor=factor, fixed_overhead_gb=fixed, **inputs
        )
        records.append(_record(inputs, est.total_gb))
    return records


def _assert_envelope_holds(cal: PerMachineCalibration, records: list[dict]) -> None:
    """Every retained point must be over-covered by the returned constants."""
    for record in records:
        actual = record["actual_peak_gb"]
        estimate = estimate_memory_gb(
            collection_overhead_factor=cal.collection_overhead_factor,
            fixed_overhead_gb=cal.fixed_overhead_gb,
            **record["estimate_inputs"],
        )
        assert estimate.total_gb >= actual * _MIN_LOCAL_ENVELOPE * ENVELOPE_SLACK, (
            f"under-covered {actual:.6g} GB with {estimate.total_gb:.6g} GB"
        )


class TestPoisonedRecordsAreRejected:
    @pytest.mark.parametrize(
        "actual",
        [float("nan"), float("inf"), float("-inf"), 10**400, 0.0, -1.0, None, True, "7.4", []],
    )
    def test_unusable_peaks_are_not_fit(self, actual):
        records = _exact_records(2.5, 0.2, n=8)
        for record in records:
            record["actual_peak_gb"] = actual
        assert refit_constants(records) is None

    @pytest.mark.parametrize("actual", [float("nan"), float("inf"), -3.0])
    def test_a_single_poisoned_peak_cannot_taint_the_fit(self, actual):
        records = _exact_records(2.5, 0.2, n=9)
        records[0]["actual_peak_gb"] = actual
        cal = refit_constants(records)
        assert cal is not None
        assert cal.n_points == 8
        assert math.isfinite(cal.collection_overhead_factor)
        assert math.isfinite(cal.fixed_overhead_gb)
        _assert_envelope_holds(cal, records[1:])

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf"), 10**400, "many", None, [1]]
    )
    def test_unusable_estimate_inputs_are_not_fit(self, value):
        records = _exact_records(2.5, 0.2, n=8)
        for record in records:
            record["estimate_inputs"]["n_observations"] = value
        assert refit_constants(records) is None

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("n_observations", 0),
            ("n_artists", -1),
            ("max_seq", 0),
            ("num_chains", 0),
            ("num_samples", 0),
            ("n_features", -1),
            ("n_observations", 1.5),
            ("num_chains", "2"),
        ],
    )
    def test_non_positive_or_coerced_dimensions_are_not_fit(self, name, value):
        records = _exact_records(2.5, 0.2, n=8)
        for record in records:
            record["estimate_inputs"][name] = value
        assert refit_constants(records) is None

    def test_non_dict_records_are_skipped(self):
        records = [*_exact_records(2.5, 0.2, n=8), "junk", None, 42]
        cal = refit_constants(records)
        assert cal is not None
        assert cal.n_points == 8

    def test_non_dict_estimate_inputs_are_skipped(self):
        records = _exact_records(2.5, 0.2, n=8)
        records[0]["estimate_inputs"] = "not a mapping"
        cal = refit_constants(records)
        assert cal is not None
        assert cal.n_points == 7

    @pytest.mark.parametrize("jbp", [float("nan"), float("inf"), 10**400, -0.5, -1.0])
    def test_unusable_jit_buffer_is_refused(self, jbp):
        assert refit_constants(_exact_records(2.5, 0.2), jit_buffer_percent=jbp) is None


class TestPeakOwnershipAndInfluence:
    def test_stale_process_peak_is_quarantined(self):
        records = _exact_records(2.5, 0.2, n=9)
        records[0]["actual_peak_gb"] *= 60.0
        records[0]["peak_provenance"] = _provenance(
            records[0]["actual_peak_gb"], attribution="stale_process_peak"
        )

        cal = refit_constants(records)

        assert cal is not None
        assert cal.n_points == 8
        assert cal.quarantine_reasons == (("stale_process_peak", 1),)
        _assert_envelope_holds(cal, records[1:])

    def test_missing_provenance_forces_shipped_fallback(self, tmp_path):
        records = _exact_records(2.5, 0.2, n=8)
        for record in records:
            record.pop("peak_provenance")
        path = tmp_path / "cal.json"
        path.write_text(json.dumps({"version": 1, "records": records}), encoding="utf-8")

        factor, fixed, source = resolve_calibration(path)

        assert factor == COLLECTION_OVERHEAD_FACTOR
        assert fixed == FIXED_OVERHEAD_GB
        assert "8 quarantined" in source
        assert "missing provenance=8" in source

    def test_recent_clean_history_quarantines_a_relative_jump(self):
        records = []
        for i in range(9):
            inputs = _inputs(num_samples=100 + 150 * i)
            shipped = estimate_memory_gb(**inputs).total_gb
            actual = shipped * (0.4 if i < 8 else 1.9)
            records.append(_record(inputs, actual))

        cal = refit_constants(records)

        assert cal is not None
        assert cal.n_points == 8
        assert cal.quarantine_reasons == (("peak influence", 1),)

    def test_legitimate_large_workloads_remain_eligible(self):
        records = _exact_records(
            COLLECTION_OVERHEAD_FACTOR,
            FIXED_OVERHEAD_GB,
            n=8,
            n_observations=20_000_000,
            n_features=200,
            n_artists=100_000,
            max_seq=100,
        )

        cal = refit_constants(records)

        assert cal is not None
        assert cal.n_points == 8
        assert cal.quarantined_points == 0
        assert max(record["actual_peak_gb"] for record in records) > 10.0
        _assert_envelope_holds(cal, records)


class TestNonFiniteCoefficients:
    def test_nan_fit_is_refused(self, monkeypatch):
        monkeypatch.setattr(np, "polyfit", lambda *a, **k: (float("nan"), float("nan")))
        assert refit_constants(_exact_records(2.5, 0.2)) is None

    def test_inf_fit_is_refused(self, monkeypatch):
        monkeypatch.setattr(np, "polyfit", lambda *a, **k: (float("inf"), 0.0))
        assert refit_constants(_exact_records(2.5, 0.2)) is None

    def test_singular_fit_is_refused(self, monkeypatch):
        def boom(*args, **kwargs):
            raise np.linalg.LinAlgError("singular")

        monkeypatch.setattr(np, "polyfit", boom)
        assert refit_constants(_exact_records(2.5, 0.2)) is None


class TestDegenerateDesigns:
    def test_identical_points_cannot_identify_two_constants(self):
        assert refit_constants(_exact_records(3.0, 0.25, n=1) * 6) is None

    def test_zero_collection_spread_is_refused(self):
        records = [_record(_inputs(num_samples=500), 4.0 + 0.01 * i) for i in range(8)]
        assert refit_constants(records) is None

    def test_all_zero_collection_terms_are_refused(self):
        records = [_record(_inputs(num_samples=0, num_warmup=0), 4.0 + 0.01 * i) for i in range(8)]
        assert refit_constants(records) is None

    def test_too_few_usable_points(self):
        assert refit_constants(_exact_records(3.0, 0.25, n=4)) is None


class TestEnvelopeIsVerifiedNotAssumed:
    def test_binding_point_with_no_scalable_term_is_refused(self):
        scale = _envelope_scale(
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            factor=0.0,
            fixed=0.0,
            jit_buffer_percent=0.1,
        )
        assert scale is None

    def test_high_base_leverage_meets_the_envelope(self):
        """A dominant base term needs a large inflation, not five nudges.

        The scalable part of the estimate is tiny next to the raw model term,
        so a bounded ladder of shrinking corrections stops far short of the
        envelope while still reporting a calibration.
        """
        records = _exact_records(
            2.5,
            0.5,
            n=8,
            n_observations=8_000_000,
            n_features=400,
            n_artists=50,
            max_seq=3,
            num_chains=1,
        )
        cal = refit_constants(records)
        assert cal is not None
        assert cal.min_ratio >= _MIN_LOCAL_ENVELOPE * ENVELOPE_SLACK
        _assert_envelope_holds(cal, records)

    def test_zero_draw_record_is_rejected_without_poisoning_valid_history(self):
        records = _exact_records(2.5, 0.2, n=8)
        records.append(_record(_inputs(num_samples=0, num_warmup=0), 1000.0))
        cal = refit_constants(records)
        assert cal is not None
        assert cal.n_points == 8
        _assert_envelope_holds(cal, records[:-1])

    def test_reported_min_ratio_matches_the_returned_constants(self):
        records = _exact_records(2.2, 0.18, n=10)
        cal = refit_constants(records)
        assert cal is not None
        observed = min(
            estimate_memory_gb(
                collection_overhead_factor=cal.collection_overhead_factor,
                fixed_overhead_gb=cal.fixed_overhead_gb,
                **r["estimate_inputs"],
            ).total_gb
            / r["actual_peak_gb"]
            for r in records
        )
        assert observed == pytest.approx(cal.min_ratio, rel=1e-9)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"chain_method": "sequential", "num_chains": 4},
            {"chain_method": "parallel", "num_chains": 4},
            {"chain_method": "vectorized", "num_chains": 4},
        ],
    )
    def test_terms_match_the_estimator_for_supported_shapes(self, overrides):
        records = _exact_records(2.5, 0.2, n=8, **overrides)
        cal = refit_constants(records)
        assert cal is not None
        _assert_envelope_holds(cal, records)

    def test_single_extreme_leverage_point_is_quarantined(self):
        records = _exact_records(2.5, 0.2, n=9)
        records[0]["actual_peak_gb"] *= 60.0
        records[0]["peak_provenance"] = _provenance(records[0]["actual_peak_gb"])
        cal = refit_constants(records)
        assert cal is not None
        assert cal.n_points == 8
        assert cal.quarantined_points == 1
        assert cal.quarantine_reasons == (("peak influence", 1),)
        _assert_envelope_holds(cal, records[1:])

    def test_already_covered_constants_are_not_inflated_further(self):
        """The 0.1 factor floor already over-covers here, so no scale applies."""
        records = _exact_records(0.001, 0.0, n=8)
        cal = refit_constants(records)
        assert cal is not None
        assert cal.collection_overhead_factor == 0.1
        # The scale is what a factor still on its floor rules out. The intercept
        # is a least-squares residual around zero, and whether it lands on -0.0
        # (clamped) or a sub-ulp positive is BLAS- and platform-dependent.
        assert cal.fixed_overhead_gb == pytest.approx(0.0, abs=1e-9)
        assert cal.min_ratio > _MIN_LOCAL_ENVELOPE
        _assert_envelope_holds(cal, records)


class TestEnvelopeProperty:
    """Whatever the data, the answer is either None or a covered calibration."""

    @pytest.mark.parametrize("seed", range(12))
    def test_broad_scales(self, seed):
        rng = np.random.default_rng(seed)
        records = []
        for _ in range(12):
            inputs = _inputs(
                n_observations=int(10 ** rng.uniform(2, 7)),
                n_features=int(rng.integers(1, 500)),
                n_artists=int(10 ** rng.uniform(1, 5)),
                max_seq=int(rng.integers(2, 200)),
                num_chains=int(rng.integers(1, 9)),
                num_samples=int(10 ** rng.uniform(1.7, 4.3)),
            )
            inputs["num_warmup"] = inputs["num_samples"]
            truth = estimate_memory_gb(
                collection_overhead_factor=float(10 ** rng.uniform(-3, 3)),
                fixed_overhead_gb=float(rng.uniform(0.0, 100.0)),
                **inputs,
            ).total_gb
            records.append(_record(inputs, truth * float(rng.uniform(0.2, 5.0))))
        cal = refit_constants(records)
        if cal is None:
            return
        assert math.isfinite(cal.collection_overhead_factor)
        assert math.isfinite(cal.fixed_overhead_gb)
        assert cal.collection_overhead_factor > 0.0
        assert cal.fixed_overhead_gb >= 0.0
        points, _ = _partition_points(records, 0.10)
        assert cal.n_points == len(points)
        for point in points:
            estimate = 1.10 * (
                point.base + cal.fixed_overhead_gb + cal.collection_overhead_factor * point.unit
            )
            assert estimate >= point.actual * _MIN_LOCAL_ENVELOPE * ENVELOPE_SLACK

    @pytest.mark.parametrize("seed", range(8))
    def test_with_leverage_and_poisoned_points(self, seed):
        rng = np.random.default_rng(1000 + seed)
        records = _exact_records(float(rng.uniform(0.5, 5.0)), float(rng.uniform(0.0, 2.0)), n=14)
        for record in records:
            record["actual_peak_gb"] *= float(rng.uniform(0.5, 2.0))
            record["peak_provenance"] = _provenance(record["actual_peak_gb"])
        records[0]["actual_peak_gb"] *= float(10 ** rng.uniform(1, 3))
        records[0]["peak_provenance"] = _provenance(records[0]["actual_peak_gb"])
        poisoned = rng.choice(len(records), size=3, replace=False)
        for index in poisoned:
            records[int(index)]["actual_peak_gb"] = rng.choice(
                [float("nan"), float("inf"), 0.0, -1.0]
            )
        cal = refit_constants(records)
        if cal is None:
            return
        points, _ = _partition_points(records, 0.10)
        assert cal.n_points == len(points)
        for point in points:
            estimate = 1.10 * (
                point.base + cal.fixed_overhead_gb + cal.collection_overhead_factor * point.unit
            )
            assert estimate >= point.actual * _MIN_LOCAL_ENVELOPE * ENVELOPE_SLACK

    @pytest.mark.parametrize("noise", [1e-6, 1e-3, 0.5, 2.0, 10.0])
    def test_noise_scales(self, noise):
        rng = np.random.default_rng(7)
        records = _exact_records(2.5, 0.3, n=10)
        for record in records:
            record["actual_peak_gb"] *= float(rng.uniform(1.0, 1.0 + noise))
            record["peak_provenance"] = _provenance(record["actual_peak_gb"])
        cal = refit_constants(records)
        if cal is None:
            return
        points, _ = _partition_points(records, 0.10)
        assert cal.n_points == len(points)
        for point in points:
            estimate = 1.10 * (
                point.base + cal.fixed_overhead_gb + cal.collection_overhead_factor * point.unit
            )
            assert estimate >= point.actual * _MIN_LOCAL_ENVELOPE * ENVELOPE_SLACK


class TestFallbackToShippedConstants:
    def test_resolve_falls_back_when_refit_refuses(self, tmp_path, monkeypatch):
        import panelcast.gpu_memory.calibration_store as store

        diagnostics = store.CalibrationDiagnostics(0, 0, 0, 0, (), ())
        monkeypatch.setattr(
            store,
            "_refit_with_diagnostics",
            lambda *a, **k: (None, diagnostics),
        )
        factor, fixed, source = resolve_calibration(tmp_path / "cal.json")
        assert factor == COLLECTION_OVERHEAD_FACTOR
        assert fixed == FIXED_OVERHEAD_GB
        assert "shipped constants" in source

    def test_poisoned_store_falls_back(self, tmp_path):
        import json

        path = tmp_path / "cal.json"
        records = _exact_records(2.5, 0.2, n=8)
        for record in records:
            record["actual_peak_gb"] = float("nan")
        path.write_text(
            json.dumps({"version": 1, "records": records}, allow_nan=True), encoding="utf-8"
        )
        factor, fixed, source = resolve_calibration(path)
        assert factor == COLLECTION_OVERHEAD_FACTOR
        assert fixed == FIXED_OVERHEAD_GB
        assert "shipped constants" in source

    def test_huge_integer_store_falls_back(self, tmp_path):
        import json

        path = tmp_path / "cal.json"
        records = _exact_records(2.5, 0.2, n=8)
        for record in records[:4]:
            record["actual_peak_gb"] = 10**400
        for record in records[4:]:
            record["estimate_inputs"]["num_samples"] = 10**400
        path.write_text(json.dumps({"version": 1, "records": records}), encoding="utf-8")

        factor, fixed, source = resolve_calibration(path)

        assert factor == COLLECTION_OVERHEAD_FACTOR
        assert fixed == FIXED_OVERHEAD_GB
        assert "shipped constants" in source
