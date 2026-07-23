"""`panelcast preflight` (pre-fit prior/data + collinearity checks).

Pure check functions are exercised directly with synthetic arrays; a thin
CliRunner test covers wiring, --json, and the --strict exit code.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.model_preflight import (
    check_beta_binomial_trial_scale,
    check_collinearity,
    check_prior_data_scale,
    cross_entity_mean_sd,
    run_model_preflight,
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


class TestBetaBinomialTrialScale:
    def test_unit_span_passes(self):
        result = check_beta_binomial_trial_scale(
            likelihood_family="beta_binomial",
            n_obs_is_aggregation_count=True,
            target_bounds=(0.0, 1.0),
        )
        assert result.status == "PASS"

    def test_nonunit_span_fails_with_rescaling_guidance(self):
        result = check_beta_binomial_trial_scale(
            likelihood_family="beta_binomial",
            n_obs_is_aggregation_count=True,
            target_bounds=(0.0, 100.0),
        )
        assert result.status == "FAIL"
        assert "100" in result.detail
        assert "[0, 1]" in result.suggestion

    @pytest.mark.parametrize("span", [0.6, 1.2, 1.4])
    def test_fractional_nonunit_span_fails(self, span):
        result = check_beta_binomial_trial_scale(
            likelihood_family="beta_binomial",
            n_obs_is_aggregation_count=True,
            target_bounds=(0.0, span),
        )
        assert result.status == "FAIL"
        assert f"{span:g}" in result.detail

    def test_other_likelihood_is_inactive(self):
        result = check_beta_binomial_trial_scale(
            likelihood_family="studentt",
            n_obs_is_aggregation_count=True,
            target_bounds=(0.0, 100.0),
        )
        assert result.status == "PASS"


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
        start = result.output.index("[\n")  # the indent=2 array start, past any log noise
        payload = json.loads(result.output[start:])
        assert payload[0]["status"] == "FAIL"
        # warn-only by default: exit 0 even on a FAIL row
        assert result.exit_code == 0

    def test_strict_setup_error_exits_2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # features not built -> assemble fails
        result = runner.invoke(app, ["preflight", "--strict"])
        assert result.exit_code == 2  # setup error, distinct from a statistical FAIL

    def test_strict_statistical_fail_exits_1(self, monkeypatch):
        y, artist_idx = _panel()
        rng = np.random.default_rng(0)
        n = len(y)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x3 = x1 + 2.0 * x2 + 1e-6 * rng.normal(size=n)  # near-collinear -> FAIL
        inputs = _StubInputs(
            X=np.column_stack([x1, x2, x3]),
            artist_idx=artist_idx,
            y=y,
            feature_names=["x1", "x2", "x3"],
            group_idx_by_artist=None,
            priors=PriorConfig(
                sigma_rw_prior_type="lognormal",
                sigma_rw_lognormal_loc=math.log(within_entity_step_sd(y, artist_idx)),
                sigma_artist_prior_type="halfnormal",
                sigma_artist_scale=cross_entity_mean_sd(y, artist_idx),
            ),
        )
        monkeypatch.setattr(
            "panelcast.model_preflight_data.assemble_preflight_inputs",
            lambda *a, **k: inputs,
        )
        result = runner.invoke(app, ["preflight", "--strict"])
        assert result.exit_code == 1

    def test_human_output_prints_suggestion(self, tmp_path, monkeypatch):
        y, artist_idx = _panel()
        rw = within_entity_step_sd(y, artist_idx)
        inputs = _StubInputs(
            X=np.random.default_rng(0).normal(size=(len(y), 3)),
            artist_idx=artist_idx,
            y=y,
            feature_names=["a", "b", "c"],
            group_idx_by_artist=None,
            priors=PriorConfig(
                sigma_rw_prior_type="lognormal",
                sigma_rw_lognormal_loc=math.log(rw) + 6.0,
                sigma_artist_prior_type="halfnormal",
                sigma_artist_scale=cross_entity_mean_sd(y, artist_idx) * 100.0,
            ),
        )
        monkeypatch.setattr(
            "panelcast.model_preflight_data.assemble_preflight_inputs",
            lambda *a, **k: inputs,
        )
        result = runner.invoke(app, ["preflight"])
        assert "sigma_artist_lognormal_loc" in result.output  # suggestion block printed
        assert "WARN" in result.output or "FAIL" in result.output


class _StubInputs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestEdgeBranches:
    def test_zero_step_sd_warns(self):
        # y constant within each entity -> no within-entity variation
        artist_idx = np.repeat(np.arange(6), 3)
        y = np.repeat(np.arange(6).astype(float), 3)
        by = {
            r.name: r
            for r in check_prior_data_scale(y=y, artist_idx=artist_idx, priors=PriorConfig())
        }
        assert by["sigma_rw scale"].status == "WARN"

    def test_single_entity_cross_sd_warns(self):
        artist_idx = np.zeros(5, dtype=int)
        y = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        by = {
            r.name: r
            for r in check_prior_data_scale(y=y, artist_idx=artist_idx, priors=PriorConfig())
        }
        assert by["sigma_artist scale"].status == "WARN"

    def test_halfnormal_and_lognormal_prior_medians(self):
        y, artist_idx = _panel()
        rw = within_entity_step_sd(y, artist_idx)
        art = cross_entity_mean_sd(y, artist_idx)
        priors = PriorConfig(
            sigma_rw_prior_type="halfnormal",
            sigma_rw_scale=rw / 0.6744897501960817,  # HalfNormal median == rw
            sigma_artist_prior_type="lognormal",
            sigma_artist_lognormal_loc=math.log(art),
        )
        by = {r.name: r for r in check_prior_data_scale(y=y, artist_idx=artist_idx, priors=priors)}
        assert by["sigma_rw scale"].status == "PASS"
        assert by["sigma_artist scale"].status == "PASS"

    def test_empty_covariates_warn(self):
        result = check_collinearity(X=np.empty((5, 0)), artist_idx=np.arange(5), feature_names=[])
        assert result.status == "WARN"

    def test_single_varying_column_passes(self):
        artist_idx = np.repeat(np.arange(10), 3)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 1))
        result = check_collinearity(X=X, artist_idx=artist_idx, feature_names=["only"])
        assert result.status == "PASS"

    def test_run_model_preflight_reports_assembly_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no prepared data -> FileNotFoundError
        results = run_model_preflight(None, None)
        assert len(results) == 1 and results[0].status == "FAIL"

    def test_moment_guards_return_zero(self):
        assert within_entity_step_sd(np.array([1.0]), np.array([0])) == 0.0
        # every entity a singleton -> no within-entity diffs
        assert within_entity_step_sd(np.arange(4.0), np.arange(4)) == 0.0
        assert cross_entity_mean_sd(np.array([]), np.array([], dtype=int)) == 0.0

    def test_moderate_collinearity_warns(self):
        rng = np.random.default_rng(0)
        artist_idx = np.repeat(np.arange(40), 4)
        n = len(artist_idx)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x3 = x1 + 2.0 * x2 + 0.01 * rng.normal(size=n)  # residual cond ~460
        result = check_collinearity(
            X=np.column_stack([x1, x2, x3]),
            artist_idx=artist_idx,
            feature_names=["x1", "x2", "x3"],
        )
        assert result.status == "WARN"


def _write_prepared(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(7)
    n_art, per = 8, 4
    rows = n_art * per
    artist = np.repeat([f"A{i}" for i in range(n_art)], per)
    rid = np.arange(rows)
    base = rng.uniform(60.0, 85.0, n_art)
    user_score = np.clip(base[np.repeat(np.arange(n_art), per)] + rng.normal(0, 5, rows), 1, 99)
    ratings = rng.integers(20, 200, rows)
    split_df = pd.DataFrame(
        {
            "Artist": artist,
            "User_Score": user_score,
            "User_Ratings": ratings,
            "original_row_id": rid,
        }
    )
    features_df = pd.DataFrame(
        {
            "feat_a": rng.normal(size=rows),
            "feat_b": rng.normal(size=rows),
            "n_reviews": ratings,
            "original_row_id": rid,
        }
    )
    fdir = tmp_path / "data" / "features"
    fdir.mkdir(parents=True)
    features_df.to_parquet(fdir / "train_features.parquet")
    sdir = tmp_path / "data" / "splits" / "within_entity_temporal"
    sdir.mkdir(parents=True)
    split_df.to_parquet(sdir / "train.parquet")


class TestAssembleOnPreparedData:
    def test_assemble_and_run(self, tmp_path, monkeypatch):
        _write_prepared(tmp_path)
        monkeypatch.chdir(tmp_path)
        results = run_model_preflight(None, None)
        assert [r.name for r in results] == [
            "beta_binomial trial scale",
            "sigma_rw scale",
            "sigma_artist scale",
            "collinearity",
        ]
        assert all(r.status in ("PASS", "WARN", "FAIL") for r in results)

    def test_config_file_layer(self, tmp_path, monkeypatch):
        _write_prepared(tmp_path)
        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / "fit.yaml"
        cfg.write_text("target_transform: identity\n", encoding="utf-8")
        results = run_model_preflight(None, [str(cfg)])
        assert len(results) == 4
