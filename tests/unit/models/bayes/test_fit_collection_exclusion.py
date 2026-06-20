"""Guard tests for in-sampler site exclusion (exclude_from_collection).

The ``~z.<site>`` exclusion changes only what the sampler STORES, never the
chain itself: a same-seed fit with and without the exclusion must produce
bit-identical draws for every other site. This is the parity guard required
before the publication run relies on the exclusion for its memory budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from panelcast.models.bayes.fit import MCMCConfig, fit_model
from panelcast.models.bayes.model import make_score_model


def _tiny_model_args(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_artists = 6
    albums_per_artist = 4
    n_obs = n_artists * albums_per_artist
    artist_idx = np.repeat(np.arange(n_artists), albums_per_artist)
    album_seq = np.tile(np.arange(1, albums_per_artist + 1), n_artists)
    X = rng.normal(size=(n_obs, 3)).astype(np.float32)
    y = (70 + 5 * rng.normal(size=n_obs)).astype(np.float32)
    prev_score = (70 + 5 * rng.normal(size=n_obs)).astype(np.float32)
    return {
        "artist_idx": artist_idx.astype(np.int32),
        "album_seq": album_seq.astype(np.int32),
        "prev_score": prev_score,
        "X": X,
        "y": y,
        "n_artists": n_artists,
        "max_seq": albums_per_artist,
    }


@pytest.mark.slow
class TestCollectionExclusionParity:
    def _fit(self, exclude_from_collection):
        config = MCMCConfig(num_warmup=30, num_samples=30, num_chains=1, seed=123)
        return fit_model(
            model=make_score_model("user"),
            model_args=_tiny_model_args(),
            config=config,
            progress_bar=False,
            exclude_from_collection=exclude_from_collection,
        )

    def test_excluded_site_absent_others_bit_identical(self):
        baseline = self._fit(None)
        excluded = self._fit(("user_rw_raw",))

        baseline_sites = set(baseline.idata.posterior.data_vars)
        excluded_sites = set(excluded.idata.posterior.data_vars)

        assert "user_rw_raw" in baseline_sites
        assert "user_rw_raw" not in excluded_sites
        assert excluded_sites == baseline_sites - {"user_rw_raw"}

        for site in sorted(excluded_sites):
            a = np.asarray(baseline.idata.posterior[site])
            b = np.asarray(excluded.idata.posterior[site])
            np.testing.assert_array_equal(a, b, err_msg=f"posterior draws differ for site {site}")

        assert baseline.divergences == excluded.divergences

    def test_post_hoc_idata_filter_remains_a_fallback(self):
        # Belt and braces: combining both exclusion mechanisms behaves like
        # the in-sampler exclusion alone.
        config = MCMCConfig(num_warmup=30, num_samples=30, num_chains=1, seed=123)
        result = fit_model(
            model=make_score_model("user"),
            model_args=_tiny_model_args(),
            config=config,
            progress_bar=False,
            exclude_from_idata=("user_rw_raw",),
            exclude_from_collection=("user_rw_raw",),
        )
        assert "user_rw_raw" not in set(result.idata.posterior.data_vars)
