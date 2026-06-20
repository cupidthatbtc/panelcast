"""Measured-ladder regression guard for memory extrapolation (slow, GPU).

Runs two real mini-MCMC rungs on synthetic data sized like a small model,
fits the production two-point calibration on the lower rungs, and asserts
the projection error at the upper rung stays within tolerance. CI keeps the
cheap unit tests of the calibration math (test_calibrate*.py); this test
needs a GPU and runs locally/nightly.

The full-scale validation against the real dataset lives in
scripts/experiment_preflight_validation.py (results in
outputs/experiments/preflight_validation.json).
"""

from __future__ import annotations

import numpy as np
import pytest

from panelcast.preflight.calibrate import calculate_calibration
from panelcast.preflight.full_check import (
    _run_mini_mcmc_subprocess,
    serialize_model_args,
)


def _has_gpu() -> bool:
    try:
        import jax

        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def _synthetic_model_args(n_artists: int = 200, albums_per_artist: int = 8) -> dict:
    rng = np.random.default_rng(7)
    n_obs = n_artists * albums_per_artist
    return {
        "artist_idx": np.repeat(np.arange(n_artists), albums_per_artist).astype(np.int32),
        "album_seq": np.tile(np.arange(1, albums_per_artist + 1), n_artists).astype(np.int32),
        "prev_score": (70 + 5 * rng.normal(size=n_obs)).astype(np.float32),
        "X": rng.normal(size=(n_obs, 10)).astype(np.float32),
        "y": (70 + 5 * rng.normal(size=n_obs)).astype(np.float32),
        "n_artists": n_artists,
        "max_seq": albums_per_artist,
    }


@pytest.mark.slow
@pytest.mark.skipif(not _has_gpu(), reason="requires a GPU for memory measurement")
class TestExtrapolationLadderRegression:
    # Projection-error tolerance at the held-out rung (plan acceptance: <15%,
    # padded for measurement noise on tiny models).
    TOLERANCE_PERCENT = 25.0

    def test_two_point_fit_projects_held_out_rung(self):
        args_path = serialize_model_args(_synthetic_model_args())
        try:
            peaks: dict[int, float] = {}
            for num_samples in (50, 150, 400):
                result = _run_mini_mcmc_subprocess(
                    args_path,
                    timeout_seconds=600,
                    num_warmup=10,
                    num_samples=num_samples,
                )
                assert result.get("success", False), result.get("error")
                peaks[num_samples] = result["peak_memory_bytes"] / (1024**3)
        finally:
            args_path.unlink(missing_ok=True)

        fixed, per_sample = calculate_calibration((50, peaks[50]), (150, peaks[150]))
        projected = fixed + per_sample * 400
        error_percent = 100.0 * abs(projected - peaks[400]) / peaks[400]
        assert error_percent < self.TOLERANCE_PERCENT, (
            f"Extrapolation error {error_percent:.1f}% exceeds "
            f"{self.TOLERANCE_PERCENT}% (projected {projected:.2f} GiB, "
            f"measured {peaks[400]:.2f} GiB; points {peaks})"
        )
