"""Parity-lock + behavior tests for the entity-overdispersion gate (C1) and
the lognormal sigma_obs prior (C2).

The load-bearing invariant for both upgrades: with the gates OFF (the default
PriorConfig), the model executes the *exact* legacy code path, so every
existing fitted number is bit-reproducible. The C1 sites are created only on
the gate-ON branch and only AFTER every existing site; because NumPyro's
``seed`` handler splits PRNG keys in site-execution order, appending sites at
the end cannot perturb any earlier site's draw. These tests prove that
property directly at the forward-trace level (cheap, no MCMC), which is why
they run before the experiment family that the model upgrade feeds.

Naming is via the ``{prefix}`` contract (here ``user_``), so the suite stays
domain-agnostic alongside ``test_no_domain_literals.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace

from panelcast.models.bayes.fit import MCMCConfig, fit_model
from panelcast.models.bayes.model import make_score_model, user_score_model
from panelcast.models.bayes.priors import PriorConfig

PREFIX = "user_"
NEW_SAMPLE_SITES = {f"{PREFIX}tau_entity", f"{PREFIX}entity_obs_raw"}
NEW_DETERMINISTIC_SITE = f"{PREFIX}entity_log_scale"


def _model_args(priors: PriorConfig, observe_y: bool = True) -> dict:
    """Small homoscedastic design with an observed target (so y is not drawn)."""
    n_obs, n_artists, n_features = 24, 6, 3
    rng = np.random.default_rng(0)
    args = {
        "artist_idx": jnp.array([i % n_artists for i in range(n_obs)], dtype=jnp.int32),
        "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)], dtype=jnp.int32),
        "prev_score": jnp.full(n_obs, 70.0, dtype=jnp.float32),
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


class TestEntityObsParityLock:
    """C1 gate-off bit-identity and gate-on append-only contract."""

    def test_gate_off_adds_no_sites(self):
        """Default PriorConfig: none of the C1 sites exist anywhere in the trace."""
        tr = _get_trace(PriorConfig())
        assert NEW_SAMPLE_SITES.isdisjoint(set(tr))
        assert NEW_DETERMINISTIC_SITE not in tr

    def test_gate_on_adds_exactly_the_two_sample_sites(self):
        off = _sample_sites(_get_trace(PriorConfig()))
        on = _sample_sites(_get_trace(PriorConfig(heteroscedastic_entity_obs=True)))
        assert on - off == NEW_SAMPLE_SITES
        # And nothing the off path had disappeared.
        assert off - on == set()

    def test_gate_on_exports_interpretable_deterministic(self):
        tr = _get_trace(PriorConfig(heteroscedastic_entity_obs=True))
        assert NEW_DETERMINISTIC_SITE in _deterministic_sites(tr)
        # entity_log_scale == tau_entity * entity_obs_raw, shape (n_artists,)
        s = np.asarray(tr[NEW_DETERMINISTIC_SITE]["value"])
        tau = float(np.asarray(tr[f"{PREFIX}tau_entity"]["value"]))
        z = np.asarray(tr[f"{PREFIX}entity_obs_raw"]["value"])
        assert s.shape == (6,)
        np.testing.assert_allclose(s, tau * z, rtol=1e-6)

    def test_shared_sites_bit_identical_off_vs_on(self):
        """RNG-order proof: appending the C1 sites at the end leaves every
        earlier site's forward draw bit-identical (same seed, observed y)."""
        off = _get_trace(PriorConfig())
        on = _get_trace(PriorConfig(heteroscedastic_entity_obs=True))
        shared = (_sample_sites(off) | _deterministic_sites(off)) & set(on)
        # The shared set must be the entire off-path latent/deterministic set.
        assert shared == (_sample_sites(off) | _deterministic_sites(off))
        for site in sorted(shared):
            a = np.asarray(off[site]["value"])
            b = np.asarray(on[site]["value"])
            np.testing.assert_array_equal(
                a, b, err_msg=f"forward draw differs for shared site {site!r}"
            )

    def test_tau_entity_scale_is_configurable(self):
        """tau_entity is HalfNormal(tau_entity_scale); a larger scale shifts mass up."""
        small = []
        large = []
        for s in range(40):
            tr_s = _get_trace(
                PriorConfig(heteroscedastic_entity_obs=True, tau_entity_scale=0.1), rng_seed=s
            )
            tr_l = _get_trace(
                PriorConfig(heteroscedastic_entity_obs=True, tau_entity_scale=1.0), rng_seed=s
            )
            small.append(float(np.asarray(tr_s[f"{PREFIX}tau_entity"]["value"])))
            large.append(float(np.asarray(tr_l[f"{PREFIX}tau_entity"]["value"])))
        assert np.mean(large) > np.mean(small)


class TestSigmaObsPriorType:
    """C2: lognormal sigma_obs prior, default-halfnormal parity."""

    def test_default_is_halfnormal_and_unchanged(self):
        """Default PriorConfig keeps sampling {prefix}sigma_obs (not a rename)."""
        tr = _get_trace(PriorConfig())
        assert f"{PREFIX}sigma_obs" in _sample_sites(tr)

    def test_lognormal_changes_only_sigma_obs(self):
        """Switching the sigma_obs prior family keeps site ORDER, so every
        other site's forward draw is bit-identical and only sigma_obs moves."""
        hn = _get_trace(PriorConfig(sigma_obs_prior_type="halfnormal"))
        ln = _get_trace(PriorConfig(sigma_obs_prior_type="lognormal"))
        assert _sample_sites(hn) == _sample_sites(ln)
        for site in sorted(_sample_sites(hn) | _deterministic_sites(hn)):
            a = np.asarray(hn[site]["value"])
            b = np.asarray(ln[site]["value"])
            if site == f"{PREFIX}sigma_obs":
                assert not np.allclose(a, b), "lognormal should move sigma_obs"
            else:
                np.testing.assert_array_equal(
                    a, b, err_msg=f"non-sigma_obs site {site!r} changed under C2"
                )

    def test_invalid_sigma_obs_prior_raises(self):
        with pytest.raises(ValueError, match="Invalid sigma_obs_prior_type"):
            _get_trace(PriorConfig(sigma_obs_prior_type="bogus"))


def _y_scale(tr: dict) -> np.ndarray:
    """The per-observation likelihood scale actually fed to the y site."""
    return np.asarray(tr[f"{PREFIX}y"]["fn"].scale)


@pytest.mark.slow
class TestEntityObsCollectionExclusionParity:
    """The high-cardinality entity_obs_raw goes through the same ~z. exclusion
    machinery as rw_raw: excluding it changes only what is STORED, never the
    chain. The interpretable deterministic entity_log_scale is kept."""

    def _fit(self, exclude_from_collection):
        rng = np.random.default_rng(0)
        n_artists, per = 6, 4
        n_obs = n_artists * per
        model_args = {
            "artist_idx": np.repeat(np.arange(n_artists), per).astype(np.int32),
            "album_seq": np.tile(np.arange(1, per + 1), n_artists).astype(np.int32),
            "prev_score": (70 + 5 * rng.normal(size=n_obs)).astype(np.float32),
            "X": rng.normal(size=(n_obs, 3)).astype(np.float32),
            "y": (70 + 5 * rng.normal(size=n_obs)).astype(np.float32),
            "n_artists": n_artists,
            "max_seq": per,
            "priors": PriorConfig(heteroscedastic_entity_obs=True),
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

    def test_entity_obs_raw_excludable_others_bit_identical(self):
        baseline = self._fit(("user_rw_raw",))
        excluded = self._fit(("user_rw_raw", "user_entity_obs_raw"))
        base_sites = set(baseline.idata.posterior.data_vars)
        excl_sites = set(excluded.idata.posterior.data_vars)
        assert "user_entity_obs_raw" in base_sites
        assert "user_entity_obs_raw" not in excl_sites
        # The interpretable per-entity factor survives the raw-site exclusion.
        assert "user_entity_log_scale" in excl_sites
        assert excl_sites == base_sites - {"user_entity_obs_raw"}
        for site in sorted(excl_sites):
            a = np.asarray(baseline.idata.posterior[site])
            b = np.asarray(excluded.idata.posterior[site])
            np.testing.assert_array_equal(a, b, err_msg=f"draws differ for {site}")
        assert baseline.divergences == excluded.divergences


class TestEntityObsWidensIntervals:
    """C1 strictly widens the observation-noise channel.

    Prior-predictive y variance is dominated by the artist-effect priors and
    the soft-clip, so the entity-noise term is invisible at that level. We
    isolate the channel by reading the scale handed to the likelihood and
    integrating over the entity-noise prior with the base model pinned:
        E[sigma_scaled^2] = sigma_obs^2 * E[exp(2 * tau * z)]
                          = sigma_obs^2 * exp(2 * tau^2)  >  sigma_obs^2.
    """

    def test_on_widens_observation_scale(self):
        off = _get_trace(PriorConfig())
        base_scale = _y_scale(off)  # scalar (homoscedastic gate-off)
        sigma_obs = np.asarray(off[f"{PREFIX}sigma_obs"]["value"])
        tau = np.float32(0.5)
        fixed = {f"{PREFIX}sigma_obs": sigma_obs, f"{PREFIX}tau_entity": tau}
        priors_on = PriorConfig(heteroscedastic_entity_obs=True)
        sq = []
        for s in range(200):
            tr = trace(seed(substitute(user_score_model, data=fixed), rng_seed=s)).get_trace(
                **_model_args(priors_on)
            )
            sq.append(_y_scale(tr) ** 2)
        mean_var_on = np.mean(sq, axis=0)  # (n_obs,)
        # Strictly wider everywhere, with the expected ~exp(2*0.25)=1.65x lift.
        assert np.all(mean_var_on > base_scale**2)
        assert mean_var_on.mean() > 1.3 * float((base_scale**2).mean())
