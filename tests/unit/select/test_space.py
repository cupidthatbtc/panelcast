"""Behavior of the select candidate space: enumeration, pruning, arm validity."""

from __future__ import annotations

from pathlib import Path

import pytest

from panelcast.config.descriptor import DatasetDescriptor, load_descriptor
from panelcast.models.bayes.likelihoods import REGISTRY
from panelcast.select.space import (
    KNOBS,
    arm_conflicts,
    default_arm,
    enumerate_space,
    knob_is_active,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
AOTY = DatasetDescriptor()


@pytest.fixture(scope="module")
def aero() -> DatasetDescriptor:
    return load_descriptor(REPO_ROOT / "examples" / "aerospace" / "descriptor.yaml")


class TestTable:
    def test_defaults_lead_every_value_tuple(self):
        for knob in KNOBS:
            assert knob.values[0] == knob.default, knob.name

    def test_every_knob_carries_history(self):
        # Past verdicts are report metadata; a knob without provenance would
        # silently drop its history from the report.
        for knob in KNOBS:
            assert knob.history, f"{knob.name} has no history note"

    def test_default_arm_matches_pipeline_defaults(self):
        from panelcast.pipelines.orchestrator import PipelineConfig

        cfg = PipelineConfig()
        arm = default_arm()
        for knob in KNOBS:
            assert arm[knob.name] == getattr(cfg, knob.name)
            assert knob.default == getattr(cfg, knob.name), (
                f"{knob.name}: table default {knob.default!r} drifted from "
                f"PipelineConfig default {getattr(cfg, knob.name)!r}"
            )

    def test_feature_affecting_knobs(self):
        flagged = {k.name for k in KNOBS if k.affects_features}
        assert flagged == {
            "gbm_offset",
            "enable_genre",
            "enable_artist",
            "enable_temporal",
            "impute_missing",
        }


class TestEnumerate:
    def test_aoty_space_is_unpruned(self):
        space = enumerate_space(AOTY)
        assert set(space["likelihood_family"]) == set(REGISTRY)
        assert space["entity_group_pooling"] == (None, True, False)
        for knob in KNOBS:
            assert space[knob.name] == knob.values, knob.name

    def test_aero_prunes_beta_binomial(self, aero):
        space = enumerate_space(aero)
        assert "beta_binomial" not in space["likelihood_family"]
        assert "beta" in space["likelihood_family"]

    def test_aero_prunes_explicit_pooling(self, aero):
        assert aero.entity_group_col is None
        assert enumerate_space(aero)["entity_group_pooling"] == (None, False)

    def test_aero_prunes_genre_ablation(self, aero):
        space = enumerate_space(aero)
        assert space["enable_genre"] == (True,)
        assert space["enable_artist"] == (True, False)
        assert space["enable_temporal"] == (True, False)

    def test_pooling_pruned_when_column_missing_from_data(self):
        space = enumerate_space(AOTY, available_columns={"User_Score", "Artist"})
        assert space["entity_group_pooling"] == (None, False)
        space = enumerate_space(AOTY, available_columns={"primary_genre"})
        assert space["entity_group_pooling"] == (None, True, False)


class TestArmConflicts:
    def test_default_arm_is_valid(self):
        assert arm_conflicts({}, AOTY) == []
        assert arm_conflicts(default_arm(), AOTY) == []

    def test_bounded_family_needs_identity(self):
        for family in ("beta", "beta_ceiling"):
            conflicts = arm_conflicts({"likelihood_family": family}, AOTY)
            assert any("target_transform='identity'" in c for c in conflicts), family
            assert arm_conflicts(
                {"likelihood_family": family, "target_transform": "identity"}, AOTY
            ) == []

    def test_discretize_needs_supporting_family_and_identity(self):
        conflicts = arm_conflicts(
            {"discretize_observation": True, "likelihood_family": "skew_studentt",
             "target_transform": "identity"},
            AOTY,
        )
        assert any("unsupported" in c for c in conflicts)
        conflicts = arm_conflicts({"discretize_observation": True}, AOTY)
        assert any("requires target_transform='identity'" in c for c in conflicts)
        assert arm_conflicts(
            {"discretize_observation": True, "target_transform": "identity"}, AOTY
        ) == []

    def test_beta_binomial_needs_aggregation_count(self, aero):
        conflicts = arm_conflicts(
            {"likelihood_family": "beta_binomial", "target_transform": "identity"}, aero
        )
        assert any("structurally incompatible" in c for c in conflicts)
        assert arm_conflicts(
            {"likelihood_family": "beta_binomial", "target_transform": "identity"}, AOTY
        ) == []

    def test_sigma_knob_at_its_default_does_not_conflict_with_a_beta_family(
        self, monkeypatch
    ):
        """A promoted-on sigma knob must not prune the Beta families out of the
        space: every arm inherits it, so no arm is mislabeled by it."""
        from panelcast.select import space as space_mod

        flipped = dict(space_mod.default_arm())
        flipped["heteroscedastic_entity_obs"] = True
        monkeypatch.setattr(space_mod, "default_arm", lambda: flipped)

        conflicts = arm_conflicts(
            {"likelihood_family": "beta", "target_transform": "identity"}, AOTY
        )
        assert [c for c in conflicts if "heteroscedastic_entity_obs" in c] == []

    def test_sigma_knob_moved_off_a_flipped_default_still_conflicts(self, monkeypatch):
        from panelcast.select import space as space_mod

        flipped = dict(space_mod.default_arm())
        flipped["heteroscedastic_entity_obs"] = True
        monkeypatch.setattr(space_mod, "default_arm", lambda: flipped)

        conflicts = arm_conflicts(
            {
                "likelihood_family": "beta",
                "target_transform": "identity",
                "heteroscedastic_entity_obs": False,
            },
            AOTY,
        )
        assert [c for c in conflicts if "heteroscedastic_entity_obs" in c] != []

    def test_unknown_knob_and_value_flagged(self):
        assert arm_conflicts({"no_such_gate": True}, AOTY) == ["unknown knob: no_such_gate"]
        conflicts = arm_conflicts({"latent_process": "arma"}, AOTY)
        assert any("not a candidate value" in c for c in conflicts)

    def test_sigma_knobs_conflict_with_sigma_ignoring_families(self):
        """Arms pairing beta families with sigma-side knobs score identically
        to the base arm while labeled as different models — pruned. The
        conflicting value is whichever moves the knob off its shipped default:
        True for the default-off learn_n_exponent, False for the default-on
        (since 0.13.0) heteroscedastic_entity_obs."""
        moved_off_default = {"learn_n_exponent": True, "heteroscedastic_entity_obs": False}
        for family in ("beta", "beta_ceiling", "beta_binomial"):
            for knob, value in moved_off_default.items():
                conflicts = arm_conflicts(
                    {
                        "likelihood_family": family,
                        "target_transform": "identity",
                        knob: value,
                    },
                    AOTY,
                )
                assert any("ignores sigma" in c for c in conflicts), (family, knob)
        # sigma-using families keep those knobs as real arms.
        assert arm_conflicts({"learn_n_exponent": True}, AOTY) == []
        assert arm_conflicts({"heteroscedastic_entity_obs": False}, AOTY) == []


class TestActivation:
    def test_n_exponent_prior_inert_without_learning(self):
        (knob,) = [k for k in KNOBS if k.name == "n_exponent_prior"]
        assert not knob_is_active(knob, {"learn_n_exponent": False})
        assert not knob_is_active(knob, {})
        assert knob_is_active(knob, {"learn_n_exponent": True})

    def test_all_other_knobs_always_active(self):
        for knob in KNOBS:
            if knob.name != "n_exponent_prior":
                assert knob_is_active(knob, {}), knob.name
