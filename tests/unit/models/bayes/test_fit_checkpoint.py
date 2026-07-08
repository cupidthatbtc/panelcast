"""Checkpointed sampling (#177): blocked draws must equal single-shot draws.

Continuing a chain through ``post_warmup_state`` is the same Markov chain —
these tests pin that parity at diagnostic scale, plus the resume/refuse
semantics of the cursor.
"""

from __future__ import annotations

import numpy as np
import pytest

import panelcast.models.bayes.fit as fit_mod
from panelcast.models.bayes.fit import MCMCConfig, _block_sizes, fit_model
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


def _fit(checkpoint_every=None, checkpoint_dir=None, seed=123):
    config = MCMCConfig(
        num_warmup=30,
        num_samples=30,
        num_chains=1,
        seed=seed,
        checkpoint_every_draws=checkpoint_every,
    )
    return fit_model(
        model=make_score_model("user"),
        model_args=_tiny_model_args(),
        config=config,
        progress_bar=False,
        checkpoint_dir=checkpoint_dir,
    )


def _assert_same_posterior(a, b):
    a_sites = set(a.idata.posterior.data_vars)
    assert a_sites == set(b.idata.posterior.data_vars)
    for site in sorted(a_sites):
        np.testing.assert_array_equal(
            np.asarray(a.idata.posterior[site]),
            np.asarray(b.idata.posterior[site]),
            err_msg=f"posterior draws differ for site {site}",
        )
    assert a.divergences == b.divergences


class TestBlockSizes:
    def test_even_split(self):
        assert _block_sizes(30, 10) == [10, 10, 10]

    def test_ragged_tail(self):
        assert _block_sizes(25, 10) == [10, 10, 5]

    def test_block_larger_than_total_is_single_shot_shaped(self):
        assert _block_sizes(30, 100) == [30]


@pytest.mark.slow
class TestCheckpointParity:
    def test_blocked_equals_single_shot(self, tmp_path):
        single = _fit()
        blocked = _fit(checkpoint_every=10, checkpoint_dir=tmp_path / "ckpt")
        _assert_same_posterior(single, blocked)
        assert not blocked.resumed_from_checkpoint

    def test_crash_resume_matches_single_shot(self, tmp_path, monkeypatch):
        class Boom(RuntimeError):
            pass

        real_mcmc = fit_mod.MCMC
        calls = {"n": 0}

        class FlakyMCMC(real_mcmc):
            def run(self, *args, **kwargs):
                calls["n"] += 1
                if calls["n"] == 3:
                    raise Boom("simulated crash mid-fit")
                return super().run(*args, **kwargs)

        ckpt = tmp_path / "ckpt"
        monkeypatch.setattr(fit_mod, "MCMC", FlakyMCMC)
        with pytest.raises(Boom):
            _fit(checkpoint_every=10, checkpoint_dir=ckpt)
        monkeypatch.setattr(fit_mod, "MCMC", real_mcmc)

        resumed = _fit(checkpoint_every=10, checkpoint_dir=ckpt)
        assert resumed.resumed_from_checkpoint
        _assert_same_posterior(_fit(), resumed)

    def test_fully_checkpointed_fit_rebuilds_without_sampling(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "ckpt"
        first = _fit(checkpoint_every=10, checkpoint_dir=ckpt)

        def no_sampling(*args, **kwargs):
            raise AssertionError("resume with all blocks done must not sample")

        monkeypatch.setattr(fit_mod.MCMC, "run", no_sampling)
        rebuilt = _fit(checkpoint_every=10, checkpoint_dir=ckpt)
        assert rebuilt.resumed_from_checkpoint
        assert rebuilt.mcmc is None
        _assert_same_posterior(first, rebuilt)

    def test_mismatched_checkpoint_refuses(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        _fit(checkpoint_every=10, checkpoint_dir=ckpt)
        with pytest.raises(ValueError, match="different fit"):
            _fit(checkpoint_every=10, checkpoint_dir=ckpt, seed=124)


class TestCheckpointGuards:
    def test_checkpoint_requires_dir(self):
        with pytest.raises(ValueError, match="checkpoint_dir"):
            _fit(checkpoint_every=10, checkpoint_dir=None)
