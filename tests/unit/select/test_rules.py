"""Pre-registered decision rules: threshold loading and per-arm verdicts."""

from __future__ import annotations

import textwrap

import pytest

from panelcast.select.rules import (
    DecisionRules,
    evaluate_candidate,
    promotable,
    reference_arm,
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

    def test_malformed_yaml_raises_not_defaults(self, tmp_path):
        # A present-but-broken file silently running under shipped defaults
        # would void pre-registration; it must raise.
        path = tmp_path / "select.yaml"
        path.write_text("rules: {promote_z: 4.0", encoding="utf-8")
        with pytest.raises(ValueError, match="malformed select config"):
            DecisionRules.load(path)

    def test_non_mapping_yaml_raises(self, tmp_path):
        path = tmp_path / "select.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a mapping"):
            DecisionRules.load(path)


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


class TestReferenceArm:
    def test_arm_turning_no_knobs_is_the_reference(self):
        ref = _passing(arm="ref")
        arm = _passing(arm="cand", knobs={"gate": True})
        assert reference_arm([arm, ref]) is ref

    def test_none_when_every_arm_turns_a_knob(self):
        assert reference_arm([_passing(arm="a", knobs={"gate": True})]) is None


class TestCoverageNonInferiority:
    """#236: outside the tolerance still clears if it beats the incumbent."""

    def test_beats_reference_promotes_despite_missing_tolerance(self):
        arm = _passing(cov80_delta=0.04, knobs={"gate": True})
        reference = _passing(arm="ref", cov80_delta=0.09, elpd_z=0.0)
        assert evaluate_candidate(arm, DecisionRules(), reference).promote

    def test_worse_than_reference_and_outside_tolerance_is_held(self):
        arm = _passing(cov80_delta=0.09, knobs={"gate": True})
        reference = _passing(arm="ref", cov80_delta=0.04, elpd_z=0.0)
        verdict = evaluate_candidate(arm, DecisionRules(), reference)
        assert not verdict.promote
        assert "no closer than the reference's +0.040" in verdict.reasons[0]

    def test_clauses_are_ord_per_axis_not_across_axes(self):
        # 80% rides non-inferiority, 95% rides the tolerance — both clear.
        arm = _passing(cov80_delta=0.04, cov95_delta=0.02, knobs={"gate": True})
        reference = _passing(arm="ref", cov80_delta=0.09, cov95_delta=0.0, elpd_z=0.0)
        assert evaluate_candidate(arm, DecisionRules(), reference).promote

    def test_one_axis_regressing_past_both_clauses_still_holds(self):
        arm = _passing(cov80_delta=0.04, cov95_delta=0.2, knobs={"gate": True})
        reference = _passing(arm="ref", cov80_delta=0.09, cov95_delta=0.0, elpd_z=0.0)
        assert not evaluate_candidate(arm, DecisionRules(), reference).promote

    def test_disabled_restores_the_absolute_bar(self):
        rules = DecisionRules(coverage_non_inferiority=False)
        arm = _passing(cov80_delta=0.04, knobs={"gate": True})
        reference = _passing(arm="ref", cov80_delta=0.09, elpd_z=0.0)
        assert not evaluate_candidate(arm, rules, reference).promote

    def test_without_a_reference_the_absolute_bar_applies(self):
        assert not evaluate_candidate(_passing(cov80_delta=0.04), DecisionRules()).promote

    def test_screening_honours_non_inferiority(self):
        arm = _passing(cov80_delta=0.04, converged=False, knobs={"gate": True})
        reference = _passing(arm="ref", cov80_delta=0.09, elpd_z=0.0)
        assert screenable(arm, DecisionRules(), reference)
        assert not screenable(arm, DecisionRules())

    def test_entity_obs_confirmation_promotes(self):
        """The #236 decision, pinned to the real s42 numbers.

        The arm misses the 0.03 cov80 tolerance by 1.5e-5 — one album out of
        653 — while the incumbent misses it by 0.023. Under the absolute-only
        bar this held; it must not hold again.
        """
        arm = ArmScore(
            arm="2fee043e3e62",
            knobs={"heteroscedastic_entity_obs": True},
            elpd_z=4.253706968718716,
            cov80_delta=0.030015313935681465,
            cov95_delta=0.017840735068912705,
            converged=True,
            rhat_max=1.0,
            ess_bulk_min=1119.0,
            divergences=0,
        )
        reference = ArmScore(
            arm="750f957a8c71",
            elpd_z=0.0,
            cov80_delta=0.052986217457886675,
            cov95_delta=0.013246554364471752,
            converged=True,
            rhat_max=1.0,
            ess_bulk_min=3085.0,
            divergences=0,
        )
        verdicts = promotable([arm, reference], DecisionRules())
        assert verdicts[0].arm == "2fee043e3e62"
        assert verdicts[0].promote
        # The reference is never its own candidate: z 0.0 is below the bar.
        assert not verdicts[1].promote

    def test_entity_obs_was_held_under_the_absolute_bar(self):
        arm = ArmScore(
            arm="2fee043e3e62",
            knobs={"heteroscedastic_entity_obs": True},
            elpd_z=4.253706968718716,
            cov80_delta=0.030015313935681465,
            cov95_delta=0.017840735068912705,
            converged=True,
        )
        rules = DecisionRules(coverage_non_inferiority=False)
        assert not evaluate_candidate(arm, rules).promote
