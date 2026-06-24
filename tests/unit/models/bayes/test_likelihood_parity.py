"""Parity guard for the likelihood-registry refactor (gated model code).

The registry refactor (`models/bayes/likelihoods.py`) moved the four original
families out of `model._sample_likelihood`'s if/elif and the `predict` dispatch
into self-contained specs. This test pins that the move is *behavior-preserving*:
a tiny NUTS fit for each of `studentt / normal / beta / skew_studentt` must
reproduce posterior draws that are bit-identical to draws captured from the
pre-refactor code (commit before the registry landed).

The golden draws live in ``fixtures/likelihood_parity.npz``; regenerate them by
running this module's ``run_family`` against the pre-refactor checkout (see the
project notes). Marked ``slow`` — it runs real MCMC.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro.infer import MCMC, NUTS

from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import PriorConfig

FAMILIES = ("studentt", "normal", "beta", "skew_studentt")
_N_OBS, _N_FEAT, _N_ART = 40, 3, 5
_GOLDEN = Path(__file__).parent / "fixtures" / "likelihood_parity.npz"


def _make_data():
    rng = np.random.default_rng(0)
    artist_idx = jnp.array([i % _N_ART for i in range(_N_OBS)], dtype=jnp.int32)
    album_seq = jnp.array([(i // _N_ART) + 1 for i in range(_N_OBS)], dtype=jnp.int32)
    X = jnp.asarray(rng.standard_normal((_N_OBS, _N_FEAT)), dtype=jnp.float32)
    y = jnp.asarray(rng.normal(70.0, 8.0, _N_OBS), dtype=jnp.float32)
    return artist_idx, album_seq, X, y


def run_family(family: str) -> dict[str, np.ndarray]:
    """Tiny deterministic NUTS fit; returns all posterior draws as numpy arrays."""
    artist_idx, album_seq, X, y = _make_data()
    mcmc = MCMC(
        NUTS(make_score_model("user")),
        num_warmup=50,
        num_samples=50,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(
        random.PRNGKey(0),
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
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_registry_refactor_is_bit_identical(family: str):
    if not _GOLDEN.exists():
        pytest.skip(f"golden fixture missing: {_GOLDEN}")
    golden = np.load(_GOLDEN)
    draws = run_family(family)
    keys = [k for k in golden.files if k.startswith(f"{family}__")]
    assert keys, f"no golden draws stored for family '{family}'"
    for key in keys:
        site = key[len(family) + 2 :]
        assert site in draws, f"{family}: refactor dropped posterior site '{site}'"
        np.testing.assert_array_equal(
            draws[site],
            golden[key],
            err_msg=f"{family}/{site}: refactor changed posterior draws",
        )
