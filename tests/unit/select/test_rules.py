"""Pre-registered decision rules: threshold loading and per-arm verdicts."""

from __future__ import annotations

import textwrap

import pytest

from panelcast.select.rules import (
    DecisionRules,
    evaluate_candidate,
    promotable,
    screenable,
)
from panelcast.select.scoring import ArmScore


def _passing(**overrides) -> ArmScore:
    base = dict(
        arm="a1",
        elpd_z=3.0,
        cov80_delta=0.01,
        cov95_delta=-0.01,
        converged=True,
        rhat_max=1.005,
        ess_bulk_min=800.0,
        divergences=0,
    )
    base.update(overrides)
    return ArmScore(**base)


class TestLoad:
    def test_defaults_when_file_missing(self, tmp_path):
        rules = DecisionRules.load(tmp_path / "nope.yaml")
        assert rules.promote_z == 2.0
        assert rules.coverage_tolerance == 0.03
        assert rules.require_convergence is True
        assert rules.confirmation_seeds == (42, 43, 44)

    def test_reads_yaml_block(self, tmp_path):
        path = tmp_path / "select.yaml"
        path.write_text(
            textwrap.dedent(
                """
                rules:
                  promote_z: 4.0
                  coverage_tolerance: 0.05
                  require_convergence: false
                  confirmation_seeds: [1, 2]
                """
            ),
            encoding="utf-8",
        )
        rules = DecisionRules.load(path)
        assert rules.promote_z == 4.0
        assert rules.coverage_tolerance == 0.05
        assert rules.require_convergence is False
        assert rules.confirmation_seeds == (1, 2)

    def test_partial_block_keeps_defaults(self, tmp_path):
        path = tmp_path / "select.yaml"
        path.write_text("rules:\n  promote_z: 5.0\n", encoding="utf-8")
        rules = DecisionRules.load(path)
        assert rules.promote_z == 5.0
        assert rules.coverage_tolerance == 0.03

    def test_missing_rules_block_is_defaults(self, tmp_path):
        path = tmp_path / "select.yaml"
        path.write_text("grid: {}\n", encoding="utf-8")
        assert DecisionRules.load(path).promote_z == 2.0


class TestEvaluate:
    def test_clean_arm_promotes(self):
        verdict = evaluate_candidate(_passing(), DecisionRules())
        assert verdict.promote
        assert verdict.reasons == []

    def test_low_z_blocks(self):
        verdict = evaluate_candidate(_passing(elpd_z=1.5), DecisionRules())
        assert not verdict.promote
        assert any("below the pre-registered threshold" in r for r in verdict.reasons)

    def test_missing_elpd_blocks(self):
        verdict = evaluate_candidate(_passing(elpd_z=None), DecisionRules())
        assert not verdict.promote
        assert any("no paired-ELPD evidence" in r for r in verdict.reasons)

    def test_coverage_out_of_tolerance_blocks(self):
        verdict = evaluate_candidate(_passing(cov95_delta=0.09), DecisionRules())
        assert not verdict.promote
        assert any("95% coverage off nominal" in r for r in verdict.reasons)

    def test_missing_coverage_blocks(self):
        verdict = evaluate_candidate(_passing(cov80_delta=None), DecisionRules())
        assert not verdict.promote
        assert any("no 80% coverage evidence" in r for r in verdict.reasons)

    def test_convergence_failure_blocks(self):
        verdict = evaluate_candidate(_passing(converged=False), DecisionRules())
        assert not verdict.promote
        assert any("convergence gate failed" in r for r in verdict.reasons)

    def test_missing_convergence_blocks(self):
        verdict = evaluate_candidate(_passing(converged=None), DecisionRules())
        assert not verdict.promote
        assert any("no convergence verdict" in r for r in verdict.reasons)

    def test_convergence_ignored_when_not_required(self):
        rules = DecisionRules(require_convergence=False)
        verdict = evaluate_candidate(_passing(converged=False), rules)
        assert verdict.promote

    def test_multiple_failures_all_reported(self):
        verdict = evaluate_candidate(
            _passing(elpd_z=0.0, cov95_delta=0.2, converged=False), DecisionRules()
        )
        assert len(verdict.reasons) == 3


class TestScreenable:
    def test_z_and_coverage_screen_even_when_not_converged(self):
        # The whole point: a non-converged arm can still be a confirmation
        # candidate (convergence is enforced later, at publication scale).
        assert screenable(_passing(converged=False), DecisionRules())

    def test_low_z_is_not_screenable(self):
        assert not screenable(_passing(elpd_z=1.5), DecisionRules())

    def test_missing_elpd_is_not_screenable(self):
        assert not screenable(_passing(elpd_z=None), DecisionRules())

    def test_coverage_out_of_tolerance_is_not_screenable(self):
        assert not screenable(_passing(cov95_delta=0.2), DecisionRules())

    def test_missing_coverage_is_not_screenable(self):
        assert not screenable(_passing(cov80_delta=None), DecisionRules())


class TestPromotable:
    def test_promotable_sorted_first(self):
        scores = [
            _passing(arm="loser", elpd_z=0.5),
            _passing(arm="winner", elpd_z=6.0),
        ]
        verdicts = promotable(scores, DecisionRules())
        assert verdicts[0].arm == "winner"
        assert verdicts[0].promote
        assert not verdicts[1].promote
