"""Checkpointed sampling (#177): blocked draws must equal single-shot draws.

Continuing a chain through ``post_warmup_state`` is the same Markov chain —
these tests pin that parity at diagnostic scale, plus the resume/refuse
semantics of the cursor.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import panelcast.models.bayes.fit as fit_mod
from panelcast.models.bayes.fit import MCMCConfig, _block_sizes, _checkpoint_identity, fit_model
from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import priors_for_transform


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

    def test_crash_between_state_and_cursor_resumes_to_single_shot(self, tmp_path, monkeypatch):
        """The window the audit found: draws and state on disk, cursor stale.

        The old layout overwrote one shared state.pkl before the cursor, so a
        kill here left a state a block ahead of the cursor and the resume
        replayed block 1 from the wrong point. Now the resume must land on the
        block-0 state and reproduce the single-shot chain exactly.
        """
        real_replace = os.replace
        seen = {"n": 0}

        def guarded(src, dst, *args, **kwargs):
            if Path(dst).name == "cursor.json":
                seen["n"] += 1
                if seen["n"] == 2:  # block 1's commit
                    raise RuntimeError("simulated kill before the cursor commit")
            return real_replace(src, dst, *args, **kwargs)

        ckpt = tmp_path / "ckpt"
        monkeypatch.setattr(os, "replace", guarded)
        with pytest.raises(RuntimeError, match="simulated kill"):
            _fit(checkpoint_every=10, checkpoint_dir=ckpt)
        monkeypatch.setattr(os, "replace", real_replace)

        cursor = json.loads((ckpt / "cursor.json").read_text(encoding="utf-8"))
        assert cursor["blocks_done"] == 1
        # Block 1's artifacts landed but were never committed.
        assert (ckpt / "block_0001.npz").exists()
        assert (ckpt / "state_0001.pkl").exists()

        resumed = _fit(checkpoint_every=10, checkpoint_dir=ckpt)
        assert resumed.resumed_from_checkpoint
        _assert_same_posterior(_fit(), resumed)


class TestCheckpointGuards:
    def test_checkpoint_requires_dir(self):
        with pytest.raises(ValueError, match="checkpoint_dir"):
            _fit(checkpoint_every=10, checkpoint_dir=None)


class TestCheckpointIdentity:
    """The identity must cover every model input, not just y/X: a resume that
    ignores priors, likelihood knobs, or the other arrays would concatenate
    draws from two different models."""

    def _run_args(self, **overrides) -> dict:
        args = {
            **_tiny_model_args(),
            "priors": priors_for_transform(),
            "likelihood_df": 4.0,
            "n_exponent": 0.0,
            "learn_n_exponent": False,
            "n_ref": None,
            "target_bounds": (0.0, 100.0),
        }
        args.update(overrides)
        return args

    def test_same_inputs_same_identity(self):
        config = MCMCConfig(num_samples=20, checkpoint_every_draws=10)
        assert _checkpoint_identity(config, self._run_args()) == _checkpoint_identity(
            config, self._run_args()
        )

    def test_non_yx_array_change_changes_identity(self):
        config = MCMCConfig(num_samples=20, checkpoint_every_draws=10)
        base = _checkpoint_identity(config, self._run_args())
        shifted = self._run_args()
        shifted["prev_score"] = shifted["prev_score"] + 20.0
        assert _checkpoint_identity(config, shifted) != base

    def test_scalar_arg_change_changes_identity(self):
        config = MCMCConfig(num_samples=20, checkpoint_every_draws=10)
        base = _checkpoint_identity(config, self._run_args())
        assert _checkpoint_identity(config, self._run_args(likelihood_df=30.0)) != base

    def test_priors_change_changes_identity(self):
        config = MCMCConfig(num_samples=20, checkpoint_every_draws=10)
        base = _checkpoint_identity(config, self._run_args())
        loose = self._run_args(priors=priors_for_transform(sigma_obs_scale=99.0))
        assert _checkpoint_identity(config, loose) != base

    def test_identity_survives_json_round_trip(self):
        # The cursor compares identities after a json round trip; a non-JSON-
        # stable identity would refuse every legitimate resume.
        config = MCMCConfig(num_samples=20, checkpoint_every_draws=10)
        identity = _checkpoint_identity(
            config,
            self._run_args(),
            model=make_score_model("user"),
            extra_fields=("diverging", "num_steps", "~z.rw_raw"),
            exclude_from_idata=("rw_raw",),
            warm_start={
                "step_size": 0.25,
                "adapt_mass_matrix": False,
                "inverse_mass_matrix": np.ones(4),
            },
        )
        assert json.loads(json.dumps(identity)) == identity


class TestCheckpointIdentityFitArguments:
    """Every output-affecting fit argument, not just the model inputs."""

    def _identity(self, **overrides):
        config = overrides.pop("config", MCMCConfig(num_samples=20, checkpoint_every_draws=10))
        kwargs = {
            "model": make_score_model("user"),
            "extra_fields": ("diverging", "num_steps"),
            "exclude_from_idata": None,
            "warm_start": None,
        }
        kwargs.update(overrides)
        return _checkpoint_identity(config, _tiny_model_args(), **kwargs)

    def test_model_change_changes_identity(self):
        # Different site prefixes are a different posterior; the run_args are
        # byte-identical, so only the model fingerprint can catch it.
        assert self._identity(model=make_score_model("critic")) != self._identity()

    def test_collected_fields_change_identity(self):
        # exclude_from_collection reaches the sampler as "~z.<site>" entries,
        # so a change to it changes which draws the blocks even contain.
        excluded = self._identity(extra_fields=("diverging", "num_steps", "~z.user_rw_raw"))
        assert excluded != self._identity()

    def test_idata_filter_changes_identity(self):
        assert self._identity(exclude_from_idata=("user_rw_raw",)) != self._identity()

    def test_idata_filter_order_does_not_change_identity(self):
        assert self._identity(exclude_from_idata=("a", "b")) == self._identity(
            exclude_from_idata=("b", "a")
        )

    def test_warm_start_changes_identity(self):
        warm = {
            "step_size": 0.25,
            "adapt_mass_matrix": False,
            "inverse_mass_matrix": np.ones(4),
        }
        assert self._identity(warm_start=warm) != self._identity()

    def test_warm_start_contents_change_identity(self):
        base = {
            "step_size": 0.25,
            "adapt_mass_matrix": False,
            "inverse_mass_matrix": np.ones(4),
        }
        shifted = {**base, "inverse_mass_matrix": np.full(4, 2.0)}
        assert self._identity(warm_start=shifted) != self._identity(warm_start=base)

    def test_warm_start_step_size_changes_identity(self):
        base = {
            "step_size": 0.25,
            "adapt_mass_matrix": False,
            "inverse_mass_matrix": np.ones(4),
        }
        assert self._identity(warm_start={**base, "step_size": 0.5}) != self._identity(
            warm_start=base
        )

    def test_environment_axes_are_recorded(self):
        identity = self._identity()
        assert identity["model"]["source_sha256"]
        assert identity["jax_backend"]
        assert identity["jax_devices"]
        assert isinstance(identity["jax_x64"], bool)
        assert identity["numpyro_version"]
        assert identity["numpy_version"]
        assert identity["jax_version"]
        assert identity["jaxlib_version"]
        assert identity["platform"]
        assert identity["machine"]
