"""Warmup transfer (#178): adaptation export/import with exact-signature scoping."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from panelcast.models.bayes.fit import MCMCConfig, fit_model
from panelcast.models.bayes.model import make_score_model


def _tiny_model_args(n_features: int = 3) -> dict:
    rng = np.random.default_rng(0)
    n_artists = 5
    per_artist = 4
    n_obs = n_artists * per_artist
    return {
        "artist_idx": np.repeat(np.arange(n_artists), per_artist).astype(np.int32),
        "album_seq": np.tile(np.arange(1, per_artist + 1), n_artists).astype(np.int32),
        "prev_score": (70 + 5 * rng.normal(size=n_obs)).astype(np.float32),
        "X": rng.normal(size=(n_obs, n_features)).astype(np.float32),
        "y": (70 + 5 * rng.normal(size=n_obs)).astype(np.float32),
        "n_artists": n_artists,
        "max_seq": per_artist,
    }


def _fit(warmup=30, export=None, imp=None, n_features=3):
    config = MCMCConfig(num_warmup=warmup, num_samples=30, num_chains=1, seed=11)
    return fit_model(
        model=make_score_model("user"),
        model_args=_tiny_model_args(n_features),
        config=config,
        progress_bar=False,
        warmup_export_path=export,
        warmup_import_path=imp,
    )


@pytest.mark.slow
class TestWarmStart:
    def test_export_then_import_warm_starts(self, tmp_path):
        export = tmp_path / "warmup.pkl"
        reference = _fit(export=export)
        assert export.exists()
        assert not reference.warm_started

        warm = _fit(warmup=10, imp=export)
        assert warm.warm_started
        assert set(warm.idata.posterior.data_vars) == set(reference.idata.posterior.data_vars)
        for site in warm.idata.posterior.data_vars:
            assert np.isfinite(np.asarray(warm.idata.posterior[site])).all()

    def test_signature_mismatch_misses_cleanly_to_cold(self, tmp_path):
        export = tmp_path / "warmup.pkl"
        _fit(export=export)
        # One more feature column reshapes beta: the transfer must miss.
        cold = _fit(warmup=10, imp=export, n_features=4)
        assert not cold.warm_started
        assert cold.divergences >= 0  # fit still completed, just cold

    def test_version_guard_misses_cleanly(self, tmp_path):
        export = tmp_path / "warmup.pkl"
        _fit(export=export)
        payload = pickle.loads(export.read_bytes())
        payload["numpyro_version"] = "0.0.0"
        export.write_bytes(pickle.dumps(payload))
        cold = _fit(warmup=10, imp=export)
        assert not cold.warm_started

    def test_chain_count_guard_misses_cleanly(self, tmp_path):
        export = tmp_path / "warmup.pkl"
        _fit(export=export)
        config = MCMCConfig(num_warmup=10, num_samples=10, num_chains=2, seed=11)
        cold = fit_model(
            model=make_score_model("user"),
            model_args=_tiny_model_args(),
            config=config,
            progress_bar=False,
            warmup_import_path=export,
        )
        assert not cold.warm_started

    def test_unreadable_export_misses_cleanly(self, tmp_path):
        bogus = tmp_path / "warmup.pkl"
        bogus.write_bytes(b"not a pickle")
        cold = _fit(warmup=10, imp=bogus)
        assert not cold.warm_started

    def test_multi_chain_export_import(self, tmp_path):
        export = tmp_path / "warmup.pkl"
        config = MCMCConfig(num_warmup=20, num_samples=20, num_chains=2, seed=11)
        fit_model(
            model=make_score_model("user"),
            model_args=_tiny_model_args(),
            config=config,
            progress_bar=False,
            warmup_export_path=export,
        )
        warm = fit_model(
            model=make_score_model("user"),
            model_args=_tiny_model_args(),
            config=MCMCConfig(num_warmup=10, num_samples=20, num_chains=2, seed=12),
            progress_bar=False,
            warmup_import_path=export,
        )
        assert warm.warm_started
        for site in warm.idata.posterior.data_vars:
            assert np.isfinite(np.asarray(warm.idata.posterior[site])).all()
