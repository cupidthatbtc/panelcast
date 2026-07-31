"""A calibration is keyed on the exclusion it measured, not the one requested.

`run_and_measure` reconciles the `exclude_collection` it is handed against the
`max_seq` in its own args JSON, dropping random-walk sites the loaded model
never samples (#410 / #431). That keeps the mini-run from crashing, but callers
folded the *requested* tuple into the calibration cache signature, so a
reconciled run wrote an entry keyed on a structure that was never measured: the
key said `("user_rw_raw",)` while the measurement was taken with the dominant
collection term still present. A later run whose panel genuinely samples the
walk would hit that entry and reuse a peak-memory number that under-reports —
the unsafe direction (#433).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from panelcast.preflight.calibrate import CalibrationError, run_calibration

MODEL_ARGS = {
    "artist_idx": [0, 1, 0, 1],
    "album_seq": [1, 1, 2, 2],
    "prev_score": [70.0, 71.0, 72.0, 73.0],
    "X": [[0.1], [0.2], [0.3], [0.4]],
    "y": [70.0, 71.0, 72.0, 73.0],
    "n_artists": 2,
    "max_seq": 2,
}


def _measurement(peak_bytes: int, effective: list[str] | None) -> dict:
    result = {"success": True, "peak_memory_bytes": peak_bytes, "runtime_seconds": 1.0}
    if effective is not None:
        result["effective_exclude_collection"] = effective
    return result


def _calibrate(requested: tuple[str, ...], effective: list[str] | None):
    peaks = iter([1 << 30, 2 << 30])
    with patch(
        "panelcast.preflight.full_check._run_mini_mcmc_subprocess",
        side_effect=lambda *a, **kw: _measurement(next(peaks), effective),
    ):
        return run_calibration(MODEL_ARGS, exclude_collection=requested)


class TestCacheKey:
    def test_an_applied_exclusion_keys_on_itself(self):
        applied = _calibrate(("user_rw_raw",), ["user_rw_raw"])
        again = _calibrate(("user_rw_raw",), ["user_rw_raw"])
        assert applied.config_hash == again.config_hash

    def test_a_reconciled_exclusion_cannot_be_served_to_a_run_that_applied_one(self):
        """The measurement was taken with the walk still collected, so its
        entry must not answer a lookup for the excluded structure."""
        reconciled = _calibrate(("user_rw_raw",), [])
        applied = _calibrate(("user_rw_raw",), ["user_rw_raw"])
        assert reconciled.config_hash != applied.config_hash

    def test_a_reconciled_exclusion_keys_like_the_structure_it_measured(self):
        reconciled = _calibrate(("user_rw_raw",), [])
        never_requested = _calibrate((), None)
        assert reconciled.config_hash == never_requested.config_hash

    def test_a_measurement_without_the_field_is_treated_as_no_exclusion(self):
        """A mini-run from an older build reports no effective tuple; reading
        that as "nothing was excluded" is the safe direction."""
        legacy = _calibrate(("user_rw_raw",), None)
        applied = _calibrate(("user_rw_raw",), ["user_rw_raw"])
        assert legacy.config_hash != applied.config_hash


class TestPointConsistency:
    def test_points_measuring_different_exclusions_are_refused(self):
        """Two peaks taken under different collection structures cannot be
        fitted to one line."""
        peaks = iter([1 << 30, 2 << 30])
        effective = iter([["user_rw_raw"], []])
        with patch(
            "panelcast.preflight.full_check._run_mini_mcmc_subprocess",
            side_effect=lambda *a, **kw: _measurement(next(peaks), next(effective)),
        ):
            with pytest.raises(CalibrationError, match="not comparable"):
                run_calibration(MODEL_ARGS, exclude_collection=("user_rw_raw",))
