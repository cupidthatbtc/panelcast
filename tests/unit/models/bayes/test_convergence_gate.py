"""check_convergence gate scoping for construction-only latents."""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest

from panelcast.models.bayes.diagnostics import check_convergence


@pytest.fixture()
def idata_with_ridge_site():
    rng = np.random.default_rng(0)
    n_chains, n_draws = 2, 400
    good = rng.normal(size=(n_chains, n_draws))
    # Chains at different levels: catastrophic split-R-hat by construction.
    ridge = np.stack([rng.normal(0.0, 1.0, n_draws), rng.normal(10.0, 1.0, n_draws)])
    posterior = {
        "user_sigma_obs": good,
        "user_entity_skew_abs": ridge[..., None].repeat(3, axis=-1),
    }
    sample_stats = {"diverging": np.zeros((n_chains, n_draws), dtype=bool)}
    return az.from_dict(posterior=posterior, sample_stats=sample_stats)


def test_ridge_site_fails_ungated(idata_with_ridge_site):
    diags = check_convergence(idata_with_ridge_site, ess_threshold=100)
    assert not diags.passed
    assert any("entity_skew_abs" in p for p in diags.failing_params)


def test_gate_exclude_scopes_out_construction_latents(idata_with_ridge_site):
    diags = check_convergence(
        idata_with_ridge_site,
        ess_threshold=100,
        gate_exclude=("user_entity_skew_abs", "user_entity_skew_sym"),
    )
    assert diags.passed
    assert not any("entity_skew" in p for p in diags.failing_params)
