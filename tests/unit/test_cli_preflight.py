"""`panelcast preflight` (pre-fit prior/data + collinearity checks).

Pure check functions are exercised directly with synthetic arrays; a thin
CliRunner test covers wiring, --json, and the --strict exit code.
"""

from __future__ import annotations

import json
import math

import numpy as np
from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.model_preflight import (
    check_collinearity,
    check_prior_data_scale,
    cross_entity_mean_sd,
    within_entity_step_sd,
)
from panelcast.models.bayes.priors import PriorConfig

runner = CliRunner()


def _panel(
    seed: int = 0, n_entities: int = 40, per: int = 6, step: float = 0.05, spread: float = 0.5
):
    rng = np.random.default_rng(seed)
    artist_idx = np.repeat(np.arange(n_entities), per)
    entity_mean = rng.normal(0.0, spread, n_entities)
    y = entity_mean[artist_idx] + rng.normal(0.0, step, n_entities * per)
    return y, artist_idx


class TestMoments:
    def test_step_sd_recovers_innovation_scale(self):
        y, artist_idx = _panel(step=0.05, spread=0.5)
        assert abs(within_entity_step_sd(y, artist_idx) - 0.05) < 0.02

    def test_cross_entity_sd_recovers_spread(self):
        y, artist_idx = _panel(step=0.05, spread=0.5)
        assert abs(cross_entity_mean_sd(y, artist_idx) - 0.5) < 0.15


class TestCheckPriorDataScale:
    def test_well_matched_passes(self):
        y, artist_idx = _panel()
        rw = within_entity_step_sd(y, artist_idx)
        art = cross_entity_mean_sd(y, artist_idx)
        priors = PriorConfig(
            sigma_rw_prior_type="lognormal",
            sigma_rw_lognormal_loc=math.log(rw),
            sigma_artist_prior_type="halfnormal",
            sigma_artist_scale=art,
        )
        results = check_prior_data_scale(y=y, artist_idx=artist_idx, priors=priors)
        assert [r.name for r in results] == ["sigma_rw scale", "sigma_artist scale"]
        assert all(r.status == "PASS" for r in results)

    def test_prior_far_above_moment_fails_with_yaml(self):
        y, artist_idx = _panel()
        rw = within_entity_step_sd(y, artist_idx)
        art = cross_entity_mean_sd(y, artist_idx)
        priors = PriorConfig(
            sigma_rw_prior_type="lognormal",
            sigma_rw_lognormal_loc=math.log(rw) + 6.0,  # ~2.6 orders above
            sigma_artist_prior_type="halfnormal",
            sigma_artist_scale=art * 100.0,  # 2 orders above
        )
        results = check_prior_data_scale(y=y, artist_idx=artist_idx, priors=priors)
        by_name = {r.name: r for r in results}
        assert by_name["sigma_rw scale"].status == "FAIL"
        assert "sigma_rw_lognormal_loc" in by_name["sigma_rw scale"].suggestion
        assert by_name["sigma_artist scale"].status == "FAIL"

    def test_sigma_rw_below_moment_is_not_flagged_until_extreme(self):
        # The moment upper-bounds latent sigma_rw, so a modestly-lower prior is fine.
        y, artist_idx = _panel()
        rw = within_entity_step_sd(y, artist_idx)
        priors = PriorConfig(
            sigma_rw_prior_type="lognormal",
            sigma_rw_lognormal_loc=math.log(rw) - 1.0,  # ~0.43 orders below
            sigma_artist_prior_type="halfnormal",
            sigma_artist_scale=cross_entity_mean_sd(y, artist_idx),
        )
        by_name = {
            r.name: r for r in check_prior_data_scale(y=y, artist_idx=artist_idx, priors=priors)
        }
        assert by_name["sigma_rw scale"].status == "PASS"


class TestCheckCollinearity:
    def test_near_collinear_fails_and_names_features(self):
        # A within-entity-varying covariate the others nearly reconstruct — the
        # age-period-cohort shape. Near-exact (not bit-exact) so it survives the
        # structural-null strip and drives the residual condition number.
        rng = np.random.default_rng(1)
        n_entities, per = 20, 3
        artist_idx = np.repeat(np.arange(n_entities), per)
        n = n_entities * per
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x3 = x1 + 2.0 * x2 + 1e-6 * rng.normal(size=n)
        X = np.column_stack([x1, x2, x3])
        result = check_collinearity(X=X, artist_idx=artist_idx, feature_names=["x1", "x2", "x3"])
        assert result.status == "FAIL"
        assert "x3" in result.detail and "x1" in result.detail

    def test_exact_structural_redundancy_is_absorbed_not_failed(self):
        # A bit-exact duplicate (one-hot / sequence-count style) is ridge-soaked,
        # so it is stripped as a structural null and does not fail on its own.
        rng = np.random.default_rng(9)
        n_entities, per = 30, 4
        artist_idx = np.repeat(np.arange(n_entities), per)
        n = n_entities * per
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        dup = x1 + 3.0  # exact affine duplicate of a single column
        X = np.column_stack([x1, x2, dup])
        result = check_collinearity(X=X, artist_idx=artist_idx, feature_names=["x1", "x2", "dup"])
        assert result.status == "PASS"
        assert "exact-redundant" in result.detail

    def test_well_conditioned_passes(self):
        rng = np.random.default_rng(2)
        n_entities, per = 30, 4
        artist_idx = np.repeat(np.arange(n_entities), per)
        X = rng.normal(size=(n_entities * per, 3))
        result = check_collinearity(X=X, artist_idx=artist_idx, feature_names=["a", "b", "c"])
        assert result.status == "PASS"

    def test_within_entity_constant_column_is_absorbed(self):
        rng = np.random.default_rng(3)
        n_entities, per = 25, 3
        artist_idx = np.repeat(np.arange(n_entities), per)
        n = n_entities * per
        good1 = rng.normal(size=n)
        good2 = rng.normal(size=n)
        per_entity = rng.normal(size=n_entities)
        const_within = per_entity[artist_idx]  # absorbed by the entity intercept
        X = np.column_stack([good1, good2, const_within])
        result = check_collinearity(
            X=X, artist_idx=artist_idx, feature_names=["good1", "good2", "const_within"]
        )
        assert result.status == "PASS"
        assert "absorbed" in result.detail

    def test_cohort_dummies_appended_and_absorbed(self):
        rng = np.random.default_rng(4)
        n_entities, per = 24, 3
        artist_idx = np.repeat(np.arange(n_entities), per)
        X = rng.normal(size=(n_entities * per, 2))
        group_idx_by_artist = np.arange(n_entities) % 3  # 3 cohorts
        result = check_collinearity(
            X=X,
            artist_idx=artist_idx,
            feature_names=["a", "b"],
            group_idx_by_artist=group_idx_by_artist,
        )
        # cohort dummies are constant within entity -> absorbed, not fatal
        assert result.status == "PASS"
        assert "absorbed" in result.detail


class TestPreflightCli:
    def test_missing_data_reports_fail_row_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no data/ here
        result = runner.invoke(app, ["preflight", "--json"])
        start = result.output.index("[")
        payload = json.loads(result.output[start:])
        assert payload[0]["status"] == "FAIL"
        # warn-only by default: exit 0 even on a FAIL row
        assert result.exit_code == 0

    def test_strict_exits_nonzero_on_fail(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["preflight", "--strict"])
        assert result.exit_code == 1
