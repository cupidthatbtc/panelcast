"""claims.yaml spec, extractors, and grading (#272)."""

from __future__ import annotations

import numpy as np
import pytest

from panelcast.replicate.evaluate import evaluate_claims, exit_code_for, grade_claim
from panelcast.replicate.extractors import ArtifactBundle, extract
from panelcast.replicate.spec import ClaimSpec, load_claims

_N_DRAWS = 400


def _bundle(rng_seed: int = 0) -> ArtifactBundle:
    """Synthetic posterior with known structure.

    Entities A..D with effects centered at [2, 1, 0, -1]; groups g0 < g1 < g2
    with offsets centered at [-1, 0, 1]; a quadratic age pair whose raw-scale
    vertex sits at 35 and an interaction pair that shifts the vertex to 30.
    """
    rng = np.random.default_rng(rng_seed)
    noise = lambda scale=0.05: rng.normal(0.0, scale, _N_DRAWS)  # noqa: E731
    # Standardization: age_c scaled by s_l=10, age_sq by s_q=700.
    s_l, s_q, s_dl, s_dq = 10.0, 700.0, 8.0, 500.0
    # Raw-scale quadratic -0.002*(x-35)^2: beta_q_raw=-0.002, beta_l_raw=0.14.
    beta_q = -0.002 * s_q + noise(0.01)
    beta_l = 0.14 * s_l + noise(0.01)
    beta_dl = -0.02 * s_dl + noise(0.005)
    beta_dq = noise(0.005)
    posterior = {
        "perf_init_artist_effect": np.stack(
            [2 + noise(), 1 + noise(), 0 + noise(), -1 + noise()], axis=1
        ),
        "perf_group_offset": np.stack(
            [-1 + noise(), 0 + noise(), 1 + noise()], axis=1
        ),
        "perf_beta": np.stack([beta_l, beta_q, beta_dl, beta_dq], axis=1),
    }
    summary = {
        "dataset": {"model_prefix": "perf"},
        "artist_to_idx": {"A": 0, "B": 1, "C": 2, "D": 3},
        "group_to_idx": {"__rest__": 0, "g0": 0, "g1": 1, "g2": 2},
        "feature_cols": ["age_c", "age_sq", "age_delta", "age_sq_delta"],
        "feature_scaler": {
            "feature_cols": ["age_c", "age_sq", "age_delta", "age_sq_delta"],
            "mean": [35.0, 1300.0, 0.0, 0.0],
            "std": [10.0, 700.0, 8.0, 500.0],
        },
    }
    return ArtifactBundle(posterior=posterior, summary=summary)


def _claim(**kwargs) -> ClaimSpec:
    return ClaimSpec(**kwargs)


class TestSpec:
    def test_load_and_parse(self, tmp_path):
        path = tmp_path / "claims.yaml"
        path.write_text(
            "claims:\n"
            "  - name: cohort_improvement\n"
            "    quantity: group_mean_trend\n"
            "    expect: {direction: increasing}\n"
            "  - name: peak_age\n"
            "    quantity: covariate_vertex(age_c, age_sq)\n"
            "    expect: {in: [30, 40]}\n"
            "    grade: qualitative\n",
            encoding="utf-8",
        )
        claims = load_claims(path)
        assert [c.name for c in claims.claims] == ["cohort_improvement", "peak_age"]
        assert claims.claims[1].extractor_name == "covariate_vertex"
        assert claims.claims[1].extractor_args == ["age_c", "age_sq"]

    def test_unknown_keys_rejected(self, tmp_path):
        path = tmp_path / "claims.yaml"
        path.write_text(
            "claims:\n"
            "  - name: x\n"
            "    quantity: group_mean_trend\n"
            "    expect: {direction: increasing}\n"
            "    surprise: 1\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_claims(path)

    def test_expect_needs_exactly_one_assertion(self):
        with pytest.raises(ValueError, match="exactly one"):
            _claim(name="x", quantity="group_mean_trend", expect={})
        with pytest.raises(ValueError, match="exactly one"):
            _claim(
                name="x",
                quantity="group_mean_trend",
                expect={"direction": "increasing", "greater_than": 0},
            )

    def test_duplicate_names_rejected(self, tmp_path):
        path = tmp_path / "claims.yaml"
        path.write_text(
            "claims:\n"
            "  - {name: x, quantity: group_mean_trend, expect: {direction: increasing}}\n"
            "  - {name: x, quantity: group_mean_trend, expect: {direction: decreasing}}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_claims(path)


class TestExtractors:
    def test_group_mean_trend_increasing(self):
        claim = _claim(
            name="t", quantity="group_mean_trend", expect={"direction": "increasing"}
        )
        q = extract(_bundle(), claim)
        assert q.shape_ok
        assert np.mean(q.draws > 0) > 0.99  # offsets rise g0 -> g2

    def test_group_mean_trend_from_anchor(self):
        claim = _claim(
            name="t",
            quantity="group_mean_trend",
            expect={"direction": "increasing", "from": "g1"},
        )
        q = extract(_bundle(), claim)
        assert q.shape_ok
        assert "2 ordered groups" in q.detail

    def test_covariate_vertex_lands_on_raw_scale(self):
        claim = _claim(
            name="v",
            quantity="covariate_vertex(age_c, age_sq)",
            expect={"in": [30, 40]},
        )
        q = extract(_bundle(), claim)
        assert q.shape_ok
        assert 34 < np.median(q.draws) < 36

    def test_covariate_coefficient_lands_on_raw_scale(self):
        claim = _claim(
            name="b",
            quantity="covariate_coefficient(age_c)",
            expect={"greater_than": 0},
        )
        q = extract(_bundle(), claim)
        assert q.shape_ok
        assert 0.13 < np.median(q.draws) < 0.15

    def test_covariate_vertex_difference_sign_and_scale(self):
        claim = _claim(
            name="v",
            quantity=(
                "covariate_vertex_difference("
                "age_c, age_sq, age_delta, age_sq_delta)"
            ),
            expect={"greater_than": 0},
        )
        q = extract(_bundle(), claim)
        assert q.shape_ok
        assert 4 < np.median(q.draws) < 6
        assert "base - interacted" in q.detail

    def test_covariate_vertex_difference_requires_concavity(self):
        bundle = _bundle()
        bundle.posterior["perf_beta"][:, 1] *= -1
        claim = _claim(
            name="v",
            quantity=(
                "covariate_vertex_difference("
                "age_c, age_sq, age_delta, age_sq_delta)"
            ),
            expect={"greater_than": 0},
        )
        assert not extract(bundle, claim).shape_ok

    def test_entity_contrast_vs_rest(self):
        claim = _claim(
            name="c",
            quantity="entity_contrast",
            expect={"greater_than": 0},
            entities={"group_a": ["A"]},
        )
        q = extract(_bundle(), claim)
        assert np.mean(q.draws > 0) > 0.99  # A sits 2 above the rest mean of 0

    def test_entity_contrast_unknown_entity_raises(self):
        claim = _claim(
            name="c",
            quantity="entity_contrast",
            expect={"greater_than": 0},
            entities={"group_a": ["Nobody"]},
        )
        with pytest.raises(ValueError, match="Nobody"):
            extract(_bundle(), claim)

    def test_entity_ranking_top_k(self):
        claim = _claim(
            name="r",
            quantity="entity_ranking(2)",
            expect={"greater_than": 0.9},
            entities={"group_a": ["A", "B"]},
        )
        q = extract(_bundle(), claim)
        assert np.median(q.draws) == 1.0  # A and B are the top two

    def test_decline_between_ages(self):
        claim = _claim(
            name="d",
            quantity="decline_between_ages(age_c, age_sq, 35, 45)",
            expect={"less_than": 0},
        )
        q = extract(_bundle(), claim)
        # -0.002*(45-35)^2 = -0.2 on the raw curve.
        assert -0.25 < np.median(q.draws) < -0.15

    def test_unknown_extractor_named_in_error(self):
        claim = _claim(name="u", quantity="nope", expect={"greater_than": 0})
        with pytest.raises(ValueError, match="unknown quantity"):
            extract(_bundle(), claim)


class TestExtractorErrors:
    def test_group_a_covering_everyone_raises(self):
        claim = _claim(
            name="c",
            quantity="entity_contrast",
            expect={"greater_than": 0},
            entities={"group_a": ["A", "B", "C", "D"]},
        )
        with pytest.raises(ValueError, match="group_b is empty"):
            extract(_bundle(), claim)

    def test_missing_entities_block_raises(self):
        claim = _claim(name="c", quantity="entity_contrast", expect={"greater_than": 0})
        with pytest.raises(ValueError, match="entities"):
            extract(_bundle(), claim)

    def test_wrong_arg_counts_raise(self):
        for quantity in (
            "covariate_vertex(age_c)",
            "covariate_coefficient(age_c, age_sq)",
            "covariate_vertex_difference(age_c, age_sq, age_delta)",
            "decline_between_ages(age_c, age_sq, 35)",
        ):
            claim = _claim(name="c", quantity=quantity, expect={"greater_than": 0})
            with pytest.raises(ValueError):
                extract(_bundle(), claim)

    def test_ranking_top_k_bounds(self):
        claim = _claim(
            name="r",
            quantity="entity_ranking(9)",
            expect={"greater_than": 0},
            entities={"group_a": ["A"]},
        )
        with pytest.raises(ValueError, match="top_k"):
            extract(_bundle(), claim)

    def test_unknown_feature_lists_roster(self):
        claim = _claim(
            name="v", quantity="covariate_vertex(nope, age_sq)", expect={"in": [0, 1]}
        )
        with pytest.raises(ValueError, match="Trained features"):
            extract(_bundle(), claim)

    def test_scaler_column_mismatch_raises(self):
        bundle = _bundle()
        bundle.summary["feature_scaler"]["feature_cols"] = ["age_sq", "age_c"]
        claim = _claim(
            name="v", quantity="covariate_vertex(age_c, age_sq)", expect={"in": [30, 40]}
        )
        with pytest.raises(ValueError, match="inconsistent"):
            extract(bundle, claim)

    def test_missing_site_names_available(self):
        bundle = _bundle()
        claim = _claim(
            name="t", quantity="group_mean_trend", expect={"direction": "increasing"}
        )
        del bundle.posterior["perf_group_offset"]
        with pytest.raises(ValueError, match="no site"):
            extract(bundle, claim)


class TestGrading:
    def test_grade_ladder_pass_divergence_fail(self):
        bundle = _bundle()
        match_claim = _claim(
            name="m",
            quantity="covariate_vertex(age_c, age_sq)",
            expect={"in": [30, 40]},
        )
        v = grade_claim(extract(bundle, match_claim), match_claim)
        assert (v.achieved, v.verdict) == ("match", "PASS")

        # A wrong tight interval whose widened band still brackets the
        # vertex: right ballpark, so a divergence.
        divergent = _claim(
            name="d",
            quantity="covariate_vertex(age_c, age_sq)",
            expect={"in": [35.5, 36.5]},
        )
        v = grade_claim(extract(bundle, divergent), divergent)
        assert (v.achieved, v.verdict) == ("qualitative", "DIVERGENCE")

        # Entirely elsewhere: only the shape survives.
        wrong = _claim(
            name="w",
            quantity="covariate_vertex(age_c, age_sq)",
            expect={"in": [80.0, 81.0]},
        )
        v = grade_claim(extract(bundle, wrong), wrong)
        assert (v.achieved, v.verdict) == ("shape_only", "DIVERGENCE")

    def test_shape_only_target_passes_on_shape(self):
        claim = _claim(
            name="s",
            quantity="covariate_vertex(age_c, age_sq)",
            expect={"in": [80.0, 81.0]},
            grade="shape_only",
        )
        v = grade_claim(extract(_bundle(), claim), claim)
        assert v.verdict == "PASS"

    def test_extractor_error_is_a_fail_verdict(self, tmp_path):
        path = tmp_path / "claims.yaml"
        path.write_text(
            "claims:\n"
            "  - name: broken\n"
            "    quantity: covariate_vertex(age_c)\n"
            "    expect: {in: [30, 40]}\n",
            encoding="utf-8",
        )
        verdicts = evaluate_claims(_bundle(), load_claims(path))
        assert verdicts[0].verdict == "FAIL"
        assert exit_code_for(verdicts) == 2

    def test_exit_codes(self, tmp_path):
        path = tmp_path / "claims.yaml"
        path.write_text(
            "claims:\n"
            "  - name: ok\n"
            "    quantity: covariate_vertex(age_c, age_sq)\n"
            "    expect: {in: [30, 40]}\n",
            encoding="utf-8",
        )
        verdicts = evaluate_claims(_bundle(), load_claims(path))
        assert [v.verdict for v in verdicts] == ["PASS"]
        assert exit_code_for(verdicts) == 0

    def test_threshold_and_direction_descriptions(self):
        bundle = _bundle()
        for expect, expected_text, verdict in (
            ({"greater_than": 0.0, "prob": 0.95}, "> 0", "PASS"),
            # The trend is increasing, so a less_than claim degrades to shape.
            ({"less_than": 0.0}, "< 0", "DIVERGENCE"),
            ({"direction": "increasing", "from": "g0"}, "increasing from g0", "PASS"),
        ):
            claim = _claim(name="t", quantity="group_mean_trend", expect=expect)
            v = grade_claim(extract(bundle, claim), claim)
            assert expected_text in v.expected
            assert v.verdict == verdict

    def test_non_finite_draws_described_readably(self):
        from panelcast.replicate.evaluate import _describe_draws

        assert _describe_draws(np.array([np.nan, 1.0])) == "non-finite draws"
        assert _describe_draws(np.array([])) == "no draws"


def _write_models_dir(tmp_path):
    """A real on-disk bundle: training_summary.json + a tiny arviz .nc."""
    import json as json_mod

    import arviz as az

    bundle = _bundle()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    summary = dict(bundle.summary)
    summary["model_type"] = "perf_score"
    (models_dir / "training_summary.json").write_text(
        json_mod.dumps(summary), encoding="utf-8"
    )
    posterior = {
        name: values.reshape(1, *values.shape)  # one chain
        for name, values in bundle.posterior.items()
    }
    az.from_dict(posterior=posterior).to_netcdf(str(models_dir / "perf_score_1.nc"))
    return models_dir


class TestBundleLoading:
    def test_load_bundle_round_trips(self, tmp_path):
        from panelcast.replicate.extractors import load_bundle

        bundle = load_bundle(_write_models_dir(tmp_path))
        assert bundle.prefix == "perf"
        assert bundle.site("group_offset").shape == (_N_DRAWS, 3)

    def test_missing_summary_raises(self, tmp_path):
        from panelcast.replicate.extractors import load_bundle

        with pytest.raises(FileNotFoundError, match="training_summary"):
            load_bundle(tmp_path)

    def test_missing_posterior_raises(self, tmp_path):
        import json as json_mod

        from panelcast.replicate.extractors import load_bundle

        (tmp_path / "training_summary.json").write_text(
            json_mod.dumps(_bundle().summary), encoding="utf-8"
        )
        with pytest.raises(FileNotFoundError, match=".nc"):
            load_bundle(tmp_path)


class TestCli:
    def _claims_file(self, tmp_path):
        path = tmp_path / "claims.yaml"
        path.write_text(
            "claims:\n"
            "  - name: peak_age\n"
            "    quantity: covariate_vertex(age_c, age_sq)\n"
            "    expect: {in: [30, 40]}\n",
            encoding="utf-8",
        )
        return path

    def test_grades_existing_fit_and_writes_json(self, tmp_path):
        import json as json_mod

        from typer.testing import CliRunner

        from panelcast.cli import app

        models_dir = _write_models_dir(tmp_path)
        out = tmp_path / "verdicts.json"
        result = CliRunner().invoke(
            app,
            [
                "replicate",
                "--claims", str(self._claims_file(tmp_path)),
                "--models", str(models_dir),
                "--json", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json_mod.loads(out.read_text(encoding="utf-8"))
        assert payload[0]["name"] == "peak_age"
        assert payload[0]["verdict"] == "PASS"

    def test_requires_exactly_one_source(self, tmp_path):
        from typer.testing import CliRunner

        from panelcast.cli import app

        result = CliRunner().invoke(
            app, ["replicate", "--claims", str(self._claims_file(tmp_path))]
        )
        assert result.exit_code == 2
        assert "exactly one" in result.output

    def test_divergence_exit_code(self, tmp_path):
        from typer.testing import CliRunner

        from panelcast.cli import app

        claims = tmp_path / "claims.yaml"
        claims.write_text(
            "claims:\n"
            "  - name: peak_age\n"
            "    quantity: covariate_vertex(age_c, age_sq)\n"
            "    expect: {in: [35.5, 36.5]}\n",
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            app,
            ["replicate", "--claims", str(claims), "--models", str(_write_models_dir(tmp_path))],
        )
        assert result.exit_code == 1
        assert "DIVERGENCE" in result.output
