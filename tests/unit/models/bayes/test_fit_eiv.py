"""Parity-lock + behavior tests for the errors-in-variables gate (model-v2).

The load-bearing invariant: with the gate OFF (the default PriorConfig) the model
executes the exact legacy code path, so every fitted number is bit-reproducible.
The single new site ``{prefix}prev_latent_raw`` is created only on the gate-ON
branch and only AFTER every existing site; because NumPyro's ``seed`` handler
splits PRNG keys in site-execution order, appending it cannot perturb any earlier
site's draw. These tests prove that at the forward-trace level (cheap) and then
show the mechanism: de-noising the lagged regressor de-attenuates rho.

Naming is via the ``{prefix}`` contract (here ``user_``), matching the sibling
``test_fit_entity_obs.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, trace

from panelcast.models.bayes.fit import MCMCConfig, fit_model
from panelcast.models.bayes.model import make_score_model, user_score_model
from panelcast.models.bayes.priors import PriorConfig

PREFIX = "user_"
NEW_SITE = f"{PREFIX}prev_latent_raw"


def _model_args(priors: PriorConfig, observe_y: bool = True) -> dict:
    n_obs, n_artists, n_features = 24, 6, 3
    rng = np.random.default_rng(0)
    args = {
        "artist_idx": jnp.array([i % n_artists for i in range(n_obs)], dtype=jnp.int32),
        "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)], dtype=jnp.int32),
        "prev_score": jnp.full(n_obs, 70.0, dtype=jnp.float32),
        "prev_meas_sigma": jnp.full(n_obs, 2.0, dtype=jnp.float32),
        "X": jnp.asarray(rng.normal(size=(n_obs, n_features)), dtype=jnp.float32),
        "n_artists": n_artists,
        "max_seq": int(n_obs // n_artists),
        "priors": priors,
    }
    if observe_y:
        args["y"] = jnp.asarray(rng.normal(70.0, 5.0, n_obs), dtype=jnp.float32)
    return args


def _sample_sites(tr: dict) -> set[str]:
    return {
        name
        for name, site in tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }


def _deterministic_sites(tr: dict) -> set[str]:
    return {name for name, site in tr.items() if site["type"] == "deterministic"}


def _get_trace(priors: PriorConfig, rng_seed: int = 0) -> dict:
    return trace(seed(user_score_model, rng_seed=rng_seed)).get_trace(**_model_args(priors))


class TestEivParityLock:
    def test_gate_off_adds_no_site(self):
        assert NEW_SITE not in _get_trace(PriorConfig())

    def test_gate_on_adds_exactly_one_sample_site(self):
        off = _sample_sites(_get_trace(PriorConfig()))
        on = _sample_sites(_get_trace(PriorConfig(errors_in_variables=True)))
        assert on - off == {NEW_SITE}
        assert off - on == set()

    def test_shared_sites_bit_identical_off_vs_on(self):
        """Appending prev_latent_raw at the end leaves every earlier site's
        forward draw bit-identical (same seed, observed y)."""
        off = _get_trace(PriorConfig())
        on = _get_trace(PriorConfig(errors_in_variables=True))
        shared = (_sample_sites(off) | _deterministic_sites(off)) & set(on)
        assert shared == (_sample_sites(off) | _deterministic_sites(off))
        for site in sorted(shared):
            a = np.asarray(off[site]["value"])
            b = np.asarray(on[site]["value"])
            np.testing.assert_array_equal(
                a, b, err_msg=f"forward draw differs for shared site {site!r}"
            )

    def test_debut_pinning_keeps_prev_latent_at_prev_score(self):
        """prev_meas_sigma == 0 (debuts) => prev_latent == prev_score, so the
        deterministic mu is identical to the gate-off mu regardless of the
        prev_latent_raw draw."""
        priors = PriorConfig(errors_in_variables=True)
        args_off = _model_args(PriorConfig())
        args_on = _model_args(priors)
        args_on["prev_meas_sigma"] = jnp.zeros_like(args_on["prev_meas_sigma"])
        # Same conditioned y and seed; only the gate differs. With meas_sigma 0
        # the EIV branch recomputes mu_raw to the identical value.
        off = trace(seed(user_score_model, rng_seed=1)).get_trace(**args_off)
        on = trace(seed(user_score_model, rng_seed=1)).get_trace(**args_on)
        np.testing.assert_allclose(
            np.asarray(off[f"{PREFIX}y"]["fn"].loc),
            np.asarray(on[f"{PREFIX}y"]["fn"].loc),
            rtol=1e-6,
        )

    def test_missing_prev_meas_sigma_raises(self):
        args = _model_args(PriorConfig(errors_in_variables=True))
        args.pop("prev_meas_sigma")
        with pytest.raises(ValueError, match="prev_meas_sigma"):
            trace(seed(user_score_model, rng_seed=0)).get_trace(**args)


@pytest.mark.slow
class TestEivCollectionExclusionParity:
    """prev_latent_raw goes through the same ~z. exclusion machinery as rw_raw:
    excluding it changes only what is STORED, never the chain."""

    def _fit(self, exclude_from_collection):
        rng = np.random.default_rng(0)
        n_artists, per = 6, 4
        n_obs = n_artists * per
        model_args = {
            "artist_idx": np.repeat(np.arange(n_artists), per).astype(np.int32),
            "album_seq": np.tile(np.arange(1, per + 1), n_artists).astype(np.int32),
            "prev_score": (70 + 5 * rng.normal(size=n_obs)).astype(np.float32),
            "prev_meas_sigma": np.full(n_obs, 2.0, dtype=np.float32),
            "X": rng.normal(size=(n_obs, 3)).astype(np.float32),
            "y": (70 + 5 * rng.normal(size=n_obs)).astype(np.float32),
            "n_artists": n_artists,
            "max_seq": per,
            "priors": PriorConfig(errors_in_variables=True),
        }
        config = MCMCConfig(num_warmup=30, num_samples=30, num_chains=1, seed=123)
        return fit_model(
            model=make_score_model("user"),
            model_args=model_args,
            config=config,
            progress_bar=False,
            exclude_from_idata=("user_rw_raw",),
            exclude_from_collection=exclude_from_collection,
        )

    def test_prev_latent_raw_excludable_others_bit_identical(self):
        baseline = self._fit(("user_rw_raw",))
        excluded = self._fit(("user_rw_raw", "user_prev_latent_raw"))
        base_sites = set(baseline.idata.posterior.data_vars)
        excl_sites = set(excluded.idata.posterior.data_vars)
        assert "user_prev_latent_raw" in base_sites
        assert "user_prev_latent_raw" not in excl_sites
        assert excl_sites == base_sites - {"user_prev_latent_raw"}
        for site in sorted(excl_sites):
            a = np.asarray(baseline.idata.posterior[site])
            b = np.asarray(excluded.idata.posterior[site])
            np.testing.assert_array_equal(a, b, err_msg=f"draws differ for {site}")
        assert baseline.divergences == excluded.divergences


def _make_eiv_synthetic(rho_true: float, n_artists: int, T: int, seed_val: int) -> dict:
    """Panel with a known AR(1) on the true level and review-count measurement
    noise on the observed score. Regressing the observed score on its observed
    lag attenuates rho; EIV de-noises the lag and recovers part of it.

    Design choices that isolate the measurement-error mechanism from the model's
    other persistence channels:
      * Homogeneous true levels (no between-artist variation) and a static artist
        effect (album_seq == 1, max_seq == 1, so the random-walk trajectory never
        sized) -> the only channel for the album-to-album lag is rho, not the RW
        (the sigma_rw <-> rho competition the model-v2 design calls out), and the
        per-artist effect cannot soak up the AR dynamics.
      * Few artists with long series -> negligible dynamic-panel (Nickell) bias.
      * Heavy small-n measurement noise (meas_const ~ score std) -> strong, well
        calibrated attenuation so the EIV lift clears the MCMC noise floor.
    """
    rng = np.random.default_rng(seed_val)
    proc_sd = 10.0
    meas_const = 18.0
    artist_idx, prev_score, prev_n, y = [], [], [], []
    for a in range(n_artists):
        true_prev = 70.0
        obs_prev = None
        prev_count = None
        for _t in range(1, T + 1):
            true_t = 70.0 + rho_true * (true_prev - 70.0) + rng.normal(0, proc_sd)
            n_rev = int(rng.integers(2, 4))  # small => large measurement error
            obs_t = true_t + rng.normal(0, meas_const / np.sqrt(n_rev))
            artist_idx.append(a)
            prev_score.append(obs_prev)  # None at debut -> filled below
            prev_n.append(prev_count)  # None at debut
            y.append(obs_t)
            true_prev = true_t
            obs_prev = obs_t
            prev_count = n_rev
    y_arr = np.asarray(y, dtype=np.float32)
    center = float(np.mean(y_arr))
    global_std = float(np.std(y_arr))
    prev_score_arr = np.asarray(
        [center if p is None else p for p in prev_score], dtype=np.float32
    )
    prev_meas_sigma = np.asarray(
        [0.0 if pn is None else global_std / np.sqrt(pn) for pn in prev_n],
        dtype=np.float32,
    )
    n_obs = len(y_arr)
    return {
        "artist_idx": np.asarray(artist_idx, dtype=np.int32),
        "album_seq": np.ones(n_obs, dtype=np.int32),
        "prev_score": prev_score_arr,
        "prev_meas_sigma": prev_meas_sigma,
        "X": np.zeros((n_obs, 1), dtype=np.float32),
        "y": y_arr,
        "n_reviews": np.full(n_obs, 5, dtype=np.int32),
        "n_artists": n_artists,
        "max_seq": 1,
        "ar_center": np.float32(center),
    }


@pytest.mark.slow
class TestEivDeAttenuatesRho:
    """Core model-v2 claim: with review-count noise on the lagged regressor,
    v1 (EIV off) underestimates rho; v2 (EIV on) recovers part of it. The
    hierarchical artist effect / sigma_obs share the noise absorption, so the
    full-model lift is partial -- the directional claim is what matters."""

    def _rho_mean(self, model_args: dict, priors: PriorConfig) -> float:
        args = dict(model_args, priors=priors)
        config = MCMCConfig(
            num_warmup=800, num_samples=800, num_chains=2, seed=7, target_accept_prob=0.95
        )
        result = fit_model(
            model=make_score_model("user"),
            model_args=args,
            config=config,
            progress_bar=False,
        )
        return float(np.mean(np.asarray(result.idata.posterior["user_rho"])))

    def test_v2_rho_exceeds_v1_and_recovers(self):
        rho_true = 0.7
        data = _make_eiv_synthetic(rho_true, n_artists=8, T=120, seed_val=0)
        rho_v1 = self._rho_mean(data, PriorConfig(errors_in_variables=False))
        rho_v2 = self._rho_mean(data, PriorConfig(errors_in_variables=True))
        # v1 is attenuated by the measurement error on the regressor.
        assert rho_v1 < rho_true
        # v2 de-attenuates: strictly larger, by a margin clear of MCMC noise,
        # and closer to the truth.
        assert rho_v2 > rho_v1 + 0.02
        assert abs(rho_v2 - rho_true) < abs(rho_v1 - rho_true)
