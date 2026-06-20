"""Regression test: chunked Predictive must handle a ragged final batch.

Predictive freezes its batch shape at construction; before the fix,
_run_known_artist_predictive built one instance from the first batch and
reassigned posterior_samples across chunks, so any total draw count that was
not a multiple of batch_size crashed in numpyro's soft_vmap shape assertion
(all stock run sizes happened to be multiples of 500, which is why this never
fired on AOTY runs).
"""

import numpy as np
import pytest

from panelcast.pipelines.evaluate import _run_known_artist_predictive


def _minimal_model_args(n_obs: int, n_artists: int, n_features: int) -> dict:
    rng = np.random.default_rng(0)
    artist_idx = np.arange(n_obs) % n_artists
    album_seq = (np.arange(n_obs) // n_artists) + 1
    return {
        "artist_idx": artist_idx.astype(np.int32),
        "album_seq": album_seq.astype(np.int32),
        "prev_score": np.full(n_obs, 70.0, dtype=np.float32),
        "X": rng.normal(size=(n_obs, n_features)).astype(np.float32),
        "y": None,
        "n_artists": n_artists,
        "max_seq": int(album_seq.max()),
    }


@pytest.mark.parametrize("n_draws,batch_size", [(7, 5), (10, 5), (3, 5)])
def test_ragged_and_exact_chunks(n_draws: int, batch_size: int) -> None:
    n_obs, n_artists, n_features = 12, 3, 2
    rng = np.random.default_rng(1)
    posterior_samples = {
        "user_beta": rng.normal(size=(n_draws, n_features)).astype(np.float32),
        "user_sigma_obs": np.full((n_draws,), 5.0, dtype=np.float32),
    }
    y = _run_known_artist_predictive(
        posterior_samples,
        _minimal_model_args(n_obs, n_artists, n_features),
        prefix="user",
        batch_size=batch_size,
    )
    assert y.shape == (n_draws, n_obs)
    assert np.all(np.isfinite(y))
