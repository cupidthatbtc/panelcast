"""Parity-lock + behavior tests for the beta_prior_type gate (#155).

The load-bearing invariant: with the gate OFF (the default PriorConfig), the
model executes the exact legacy ``{prefix}beta ~ Normal`` sample in the exact
legacy position, so every existing fitted number is bit-reproducible. Gate-on
replaces that sample with the regularized-horseshoe block mid-sequence
(Piironen & Vehtari 2017) — sites BEFORE beta keep identical draws (NumPyro's
``seed`` handler splits keys in site-execution order), sites after
legitimately reshuffle (the entity_group_pooling precedent) — and re-exports
``{prefix}beta`` as a deterministic so downstream readers are untouched.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, trace

from panelcast.models.bayes.model import user_score_model
from panelcast.models.bayes.priors import PriorConfig

PREFIX = "user_"


def _model_args(priors: PriorConfig) -> dict:
    """Small observed design (mirrors the entity-obs parity fixture)."""
    n_obs, n_artists, n_features = 24, 6, 3
    rng = np.random.default_rng(0)
    return {
        "artist_idx": jnp.array([i % n_artists for i in range(n_obs)], dtype=jnp.int32),
        "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)], dtype=jnp.int32),
        "prev_score": jnp.full(n_obs, 70.0, dtype=jnp.float32),
        "X": jnp.asarray(rng.normal(size=(n_obs, n_features)), dtype=jnp.float32),
        "n_artists": n_artists,
        "max_seq": int(n_obs // n_artists),
        "priors": priors,
        "y": jnp.asarray(rng.normal(70.0, 5.0, n_obs), dtype=jnp.float32),
    }

BETA = f"{PREFIX}beta"
HS_SITES = {
    f"{PREFIX}beta_z",
    f"{PREFIX}beta_lambda",
    f"{PREFIX}beta_tau",
    f"{PREFIX}beta_c2",
}
# Sites sampled before the beta block in the model's execution order.
UPSTREAM_SITES = (
    f"{PREFIX}mu_artist",
    f"{PREFIX}sigma_artist",
    f"{PREFIX}sigma_rw",
    f"{PREFIX}rho",
)


def _get_trace(priors: PriorConfig, rng_seed: int = 0) -> dict:
    return trace(seed(user_score_model, rng_seed=rng_seed)).get_trace(**_model_args(priors))


def _sample_sites(tr: dict) -> set[str]:
    return {
        name
        for name, site in tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }


class TestBetaPriorParityLock:
    def test_gate_off_adds_no_sites_and_beta_stays_a_sample(self):
        tr = _get_trace(PriorConfig())
        assert HS_SITES.isdisjoint(set(tr))
        assert tr[BETA]["type"] == "sample"

    def test_gate_on_swaps_exactly_the_beta_block(self):
        off = _sample_sites(_get_trace(PriorConfig()))
        on = _sample_sites(_get_trace(PriorConfig(beta_prior_type="horseshoe")))
        assert on - off == HS_SITES
        assert off - on == {BETA}

    def test_gate_on_upstream_draws_bit_identical(self):
        """Sites sampled before the beta block keep their exact gate-off draws."""
        off = _get_trace(PriorConfig())
        on = _get_trace(PriorConfig(beta_prior_type="horseshoe"))
        for site in UPSTREAM_SITES:
            np.testing.assert_array_equal(
                np.asarray(off[site]["value"]), np.asarray(on[site]["value"]), err_msg=site
            )

    def test_gate_on_exports_beta_deterministic(self):
        tr = _get_trace(PriorConfig(beta_prior_type="horseshoe"))
        assert tr[BETA]["type"] == "deterministic"
        beta = np.asarray(tr[BETA]["value"])
        assert beta.shape == (3,)  # n_features in the shared design
        # beta == z * lambda_tilde * tau with the regularized-horseshoe lambda.
        z = np.asarray(tr[f"{PREFIX}beta_z"]["value"])
        lam = np.asarray(tr[f"{PREFIX}beta_lambda"]["value"])
        tau = float(np.asarray(tr[f"{PREFIX}beta_tau"]["value"]))
        c2 = float(np.asarray(tr[f"{PREFIX}beta_c2"]["value"]))
        lam_tilde = np.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
        np.testing.assert_allclose(beta, z * lam_tilde * tau, rtol=1e-5)

    def test_slab_bounds_the_scale(self):
        """lambda_tilde -> sqrt(c2) as lambda -> inf: no coefficient escapes the slab."""
        c2, tau = 4.0, 0.1
        lam = np.array([1e12])
        lam_tilde = np.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
        assert lam_tilde[0] * tau == pytest.approx(np.sqrt(c2), rel=1e-3)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid beta_prior_type"):
            _get_trace(PriorConfig(beta_prior_type="bogus"))
