"""Parity guard for the likelihood-registry refactor (gated model code).

The registry refactor (`models/bayes/likelihoods.py`) moved the four original
families out of `model._sample_likelihood`'s if/elif and the `predict` dispatch
into self-contained specs. These tests pin that the move is *behavior-preserving*
against artifacts captured from the pre-refactor code (ef9ddca, the commit
before the registry landed).

Bit-exact MCMC draws only reproduce on the machine that generated them — XLA
emits different code per CPU and NUTS amplifies one-ULP differences into fully
diverged chains — so the guard has two tiers:

- default (CI-safe): the model's joint log-density, evaluated at every golden
  posterior draw, must match the pre-refactor values in
  ``fixtures/likelihood_parity_logdensity.npz``. A pure forward pass is
  hardware-stable, and any change to a likelihood's math moves it.
- ``PANELCAST_PARITY_EXACT=1`` (the reference machine): the tiny NUTS fit must
  also reproduce the golden draws in ``fixtures/likelihood_parity.npz``
  bit-for-bit.

Regenerate both fixtures by running this module's helpers against the
pre-refactor checkout (see the project notes). Only the exact tier runs real
MCMC and is marked ``slow``.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density

from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import PriorConfig

FAMILIES = ("studentt", "normal", "beta", "skew_studentt")
_N_OBS, _N_FEAT, _N_ART = 40, 3, 5
_FIXTURES = Path(__file__).parent / "fixtures"
_GOLDEN_DRAWS = _FIXTURES / "likelihood_parity.npz"
_GOLDEN_LOGDENSITY = _FIXTURES / "likelihood_parity_logdensity.npz"

_EXACT = os.environ.get("PANELCAST_PARITY_EXACT") == "1"


def _model_kwargs(family: str) -> dict:
    rng = np.random.default_rng(0)
    artist_idx = jnp.array([i % _N_ART for i in range(_N_OBS)], dtype=jnp.int32)
    album_seq = jnp.array([(i // _N_ART) + 1 for i in range(_N_OBS)], dtype=jnp.int32)
    X = jnp.asarray(rng.standard_normal((_N_OBS, _N_FEAT)), dtype=jnp.float32)
    y = jnp.asarray(rng.normal(70.0, 8.0, _N_OBS), dtype=jnp.float32)
    return dict(
        artist_idx=artist_idx,
        album_seq=album_seq,
        prev_score=jnp.full(_N_OBS, 70.0),
        X=X,
        y=y,
        n_artists=_N_ART,
        max_seq=int(album_seq.max()),
        priors=PriorConfig(likelihood_family=family),
        target_bounds=(0.0, 100.0),
        likelihood_df=4.0,
        ar_center=70.0,
    )


def run_family(family: str) -> dict[str, np.ndarray]:
    """Tiny deterministic NUTS fit; returns all posterior draws as numpy arrays."""
    mcmc = MCMC(
        NUTS(make_score_model("user")),
        num_warmup=50,
        num_samples=50,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0), **_model_kwargs(family))
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


def _golden_sites(golden, family: str) -> dict[str, np.ndarray]:
    sites = {
        k[len(family) + 2 :]: golden[k]
        for k in golden.files
        if k.startswith(f"{family}__")
    }
    assert sites, f"no golden draws stored for family '{family}'"
    return sites


def test_golden_fixtures_present():
    """Fast tier: the golden files must exist — a silent skip would void the guard."""
    for path in (_GOLDEN_DRAWS, _GOLDEN_LOGDENSITY):
        assert path.exists(), f"golden fixture missing: {path}"
    assert _GOLDEN_DRAWS.stat().st_size > 10_000, "golden fixture is implausibly small"
    draws = np.load(_GOLDEN_DRAWS)
    logdensity = np.load(_GOLDEN_LOGDENSITY)
    for family in FAMILIES:
        assert any(k.startswith(f"{family}__") for k in draws.files), (
            f"golden fixture has no draws for family '{family}'"
        )
        assert family in logdensity.files, (
            f"golden fixture has no log-densities for family '{family}'"
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_registry_refactor_preserves_log_density(family: str):
    golden_ld = np.load(_GOLDEN_LOGDENSITY)[family]
    sites = _golden_sites(np.load(_GOLDEN_DRAWS), family)
    kwargs = _model_kwargs(family)
    model = make_score_model("user")

    shapes = {
        name: np.shape(site["value"])
        for name, site in trace(seed(model, 0)).get_trace(**kwargs).items()
        if site["type"] in ("sample", "deterministic")
        and not site.get("is_observed", False)
    }
    for site, draws in sites.items():
        assert site in shapes, f"{family}: refactor dropped posterior site '{site}'"
        assert draws.shape[1:] == shapes[site], f"{family}/{site}: site shape changed"

    lds = np.asarray(
        [
            float(
                log_density(
                    seed(model, 0),
                    (),
                    kwargs,
                    {s: jnp.asarray(v[i]) for s, v in sites.items()},
                )[0]
            )
            for i in range(golden_ld.shape[0])
        ]
    )
    np.testing.assert_allclose(
        lds,
        golden_ld,
        rtol=1e-4,
        err_msg=f"{family}: refactor changed the joint log-density",
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not _EXACT,
    reason="bit-exact draws only reproduce on the golden-fixture reference "
    "machine; set PANELCAST_PARITY_EXACT=1 there",
)
@pytest.mark.parametrize("family", FAMILIES)
def test_registry_refactor_draws_bit_identical(family: str):
    golden = np.load(_GOLDEN_DRAWS)
    draws = run_family(family)
    for site, gold in _golden_sites(golden, family).items():
        assert site in draws, f"{family}: refactor dropped posterior site '{site}'"
        np.testing.assert_array_equal(
            draws[site],
            gold,
            err_msg=f"{family}/{site}: refactor changed posterior draws",
        )
