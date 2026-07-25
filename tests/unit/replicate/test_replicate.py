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
    vertex sits at 35 and declines from 35 to 45.
    """
    rng = np.random.default_rng(rng_seed)
    noise = lambda scale=0.05: rng.normal(0.0, scale, _N_DRAWS)  # noqa: E731
    # Standardization: age_c scaled by s_l=10, age_sq by s_q=700.
    s_l, s_q = 10.0, 700.0
    # Raw-scale quadratic -0.002*(x-35)^2: beta_q_raw=-0.002, beta_l_raw=0.14.
    beta_q = -0.002 * s_q + noise(0.01)
    beta_l = 0.14 * s_l + noise(0.01)
    posterior = {
        "perf_init_artist_effect": np.stack(
            [2 + noise(), 1 + noise(), 0 + noise(), -1 + noise()], axis=1
        ),
        "perf_group_offset": np.stack(
            [-1 + noise(), 0 + noise(), 1 + noise()], axis=1
        ),
        "perf_beta": np.stack([beta_l, beta_q], axis=1),
    }
    summary = {
        "dataset": {"model_prefix": "perf"},
        "artist_to_idx": {"A": 0, "B": 1, "C": 2, "D": 3},
        "group_to_idx": {"__rest__": 0, "g0": 0, "g1": 1, "g2": 2},
        "feature_cols": ["age_c", "age_sq"],
        "feature_scaler": {
            "feature_cols": ["age_c", "age_sq"],
            "mean": [35.0, 1300.0],
            "std": [10.0, 700.0],
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
