"""Publication-claim drift guard (#298).

One canonical release-result manifest (.audit/release_results.json) backs every
headline claim in README.md and MODEL_CARD.md. These tests fail CI when the
manifest drifts from the committed metrics snapshot, or when a publication
surface drifts from the manifest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = json.loads((REPO_ROOT / ".audit" / "release_results.json").read_text("utf-8"))
SNAPSHOT = json.loads((REPO_ROOT / ".audit" / "baseline_metrics.json").read_text("utf-8"))
README = (REPO_ROOT / "README.md").read_text("utf-8")
MODEL_CARD = (REPO_ROOT / "MODEL_CARD.md").read_text("utf-8")

WITHIN = SNAPSHOT["splits"]["within_entity_temporal"]
COLD = SNAPSHOT["splits"]["entity_disjoint"]


class TestManifestMatchesMetricsSnapshot:
    def test_within_entity_point_metrics(self):
        m = MANIFEST["metrics"]["within_entity_temporal"]
        p = WITHIN["point_metrics"]
        assert m["mae"] == round(p["mae"], 2)
        assert m["rmse"] == round(p["rmse"], 2)
        assert m["r2"] == round(p["r2"], 3)
        assert m["crps"] == round(WITHIN["crps"]["mean_crps"], 2)

    def test_within_entity_elpd(self):
        m = MANIFEST["metrics"]["within_entity_temporal"]
        e = WITHIN["info_criteria"]["heldout_elpd"]
        assert m["heldout_elpd"] == round(e["elpd"], 1)
        assert m["heldout_elpd_se"] == round(e["se"], 1)
        assert m["heldout_elpd_per_obs"] == round(e["elpd_per_obs"], 2)

    def test_within_entity_coverage(self):
        m = MANIFEST["metrics"]["within_entity_temporal"]
        cov = WITHIN["calibration"]["coverages"]
        assert m["coverage_80"] == round(cov["0.80"]["empirical"], 3)
        assert m["coverage_95"] == round(cov["0.95"]["empirical"], 3)
        assert m["interval_width_80"] == round(cov["0.80"]["interval_width"], 1)
        assert m["interval_width_95"] == round(cov["0.95"]["interval_width"], 1)

    def test_entity_disjoint_metrics(self):
        m = MANIFEST["metrics"]["entity_disjoint"]
        p = COLD["point_metrics"]
        assert m["mae"] == round(p["mae"], 2)
        assert m["r2"] == round(p["r2"], 3)
        assert m["coverage_95"] == round(
            COLD["calibration"]["coverages"]["0.95"]["empirical"], 3
        )

    def test_evaluated_test_sizes(self):
        flow = MANIFEST["dataset_flow"]["evaluated_test"]
        assert flow["within_entity_temporal_n"] == WITHIN["point_metrics"]["n_observations"]
        assert flow["entity_disjoint_n"] == COLD["point_metrics"]["n_observations"]

    def test_ppc_pinned_set_matches_snapshot(self):
        assert sorted(MANIFEST["ppc"]["pinned"]) == sorted(
            WITHIN["ppc"]["extreme_statistics"]
        )
        recorded = set(MANIFEST["ppc"]["pinned"]) | set(MANIFEST["ppc"]["interior"])
        assert recorded == set(WITHIN["ppc"]["summary"])

    def test_ppc_p_values_match_snapshot(self):
        for stat, p in MANIFEST["ppc"]["p_values"].items():
            assert p == round(WITHIN["ppc"]["summary"][stat]["p_value"], 3), stat


class TestReadmeClaims:
    def test_headline_comparison_row(self):
        m = MANIFEST["metrics"]["within_entity_temporal"]
        row = (
            f"| **panelcast** | **{m['mae']:.2f}** | **{m['r2']:.3f}** "
            f"| {m['coverage_80']:.3f} | {m['coverage_95']:.3f} |"
        )
        assert row in README, row

    def test_convergence_claim(self):
        g = MANIFEST["convergence_gate"]
        claim = f"R-hat {g['rhat_max']:.2f}, bulk ESS {g['ess_bulk_min']:,}, {g['divergences']} divergences"
        assert claim in README, claim

    def test_cold_start_claim(self):
        m = MANIFEST["metrics"]["entity_disjoint"]
        for fragment in (f"MAE {m['mae']:.2f}", f"R² {m['r2']:.3f}", f"{m['coverage_95']:.3f}"):
            assert fragment in README, fragment


class TestModelCardClaims:
    def test_version_matches_pyproject(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text("utf-8")
        version = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
        assert f"**Version:** {version}" in MODEL_CARD

    def test_convergence_numbers(self):
        g = MANIFEST["convergence_gate"]
        assert f"R-hat (max): {g['rhat_max']:.2f}" in MODEL_CARD
        assert f"{g['ess_bulk_min']:,}" in MODEL_CARD
        assert f"Divergent transitions: {g['divergences']}" in MODEL_CARD

    def test_headline_metrics(self):
        m = MANIFEST["metrics"]["within_entity_temporal"]
        assert f"MAE: {m['mae']:.2f}" in MODEL_CARD
        assert f"R-squared: {m['r2']:.3f}" in MODEL_CARD

    def test_ppc_p_values(self):
        for stat, p in MANIFEST["ppc"]["p_values"].items():
            assert f"- {stat}: " in MODEL_CARD, stat
            assert f"p={p:.3f}" in MODEL_CARD, (stat, p)

    def test_scale_tiers_distinguished(self):
        for tier in ("Raw corpus", "Eligible corpus", "Validated subset"):
            assert tier in MODEL_CARD, tier


class TestNoSurfaceClaimsAClearedPin:
    @pytest.mark.parametrize("surface_name,text", [("README", README), ("MODEL_CARD", MODEL_CARD)])
    def test_pinned_language_only_names_pinned_stats(self, surface_name, text):
        interior = set(MANIFEST["ppc"]["interior"])
        for line in text.splitlines():
            match = re.search(r"stays? pinned", line)
            if not match:
                continue
            # Only the clause ending in "stay pinned" names the pinned
            # statistics; neighboring clauses may legitimately discuss cleared
            # ones ("cleared q10 and q90; only skewness and max stay pinned").
            subject = re.split(r"[;.—]", line[: match.start()])[-1]
            offenders = [
                stat for stat in interior if re.search(rf"\b{re.escape(stat)}\b", subject)
            ]
            assert not offenders, (surface_name, offenders, line)

    def test_historical_combined_pin_phrase_is_gone(self):
        assert "skewness/max/q90" not in README
        assert "skewness/max/q90" not in MODEL_CARD
