"""Vectorized chain execution (#176): batched chains sample the same model.

Vectorized chains draw a different rng fan-out than sequential — draws are NOT
comparable bit-for-bit, so these tests pin structure and sane diagnostics on
both latent-process branches (the AR/RW path runs through lax.scan, where
NumPyro's vectorized mode has historically had edge cases).
"""

from __future__ import annotations

import numpy as np
import pytest

from panelcast.models.bayes.fit import MCMCConfig, fit_model
from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import PriorConfig


def _model_args(latent_process: str) -> dict:
    rng = np.random.default_rng(0)
    n_artists = 5
    per_artist = 4
    n_obs = n_artists * per_artist
    return {
        "artist_idx": np.repeat(np.arange(n_artists), per_artist).astype(np.int32),
        "album_seq": np.tile(np.arange(1, per_artist + 1), n_artists).astype(np.int32),
        "prev_score": (70 + 5 * rng.normal(size=n_obs)).astype(np.float32),
        "X": rng.normal(size=(n_obs, 3)).astype(np.float32),
        "y": (70 + 5 * rng.normal(size=n_obs)).astype(np.float32),
        "n_artists": n_artists,
        "max_seq": per_artist,
        "priors": PriorConfig(latent_process=latent_process),
    }


@pytest.mark.slow
@pytest.mark.parametrize("latent_process", ["rw", "ar1"])
class TestVectorizedChains:
    def _fit(self, chain_method: str, latent_process: str):
        config = MCMCConfig(
            num_warmup=30, num_samples=30, num_chains=2, seed=7, chain_method=chain_method
        )
        return fit_model(
            model=make_score_model("user"),
            model_args=_model_args(latent_process),
            config=config,
            progress_bar=False,
        )

    def test_vectorized_matches_sequential_structure(self, latent_process):
        seq = self._fit("sequential", latent_process)
        vec = self._fit("vectorized", latent_process)
        seq_sites = set(seq.idata.posterior.data_vars)
        assert set(vec.idata.posterior.data_vars) == seq_sites
        for site in sorted(seq_sites):
            a = np.asarray(seq.idata.posterior[site])
            b = np.asarray(vec.idata.posterior[site])
            assert a.shape == b.shape, f"shape mismatch for {site}"
            assert np.isfinite(b).all(), f"non-finite vectorized draws for {site}"
        assert vec.divergences >= 0
        assert "diverging" in vec.idata.sample_stats
