"""Effort tiers: shipped defaults, YAML overrides, and the SweepConfig mapping."""

from __future__ import annotations

import textwrap

import pytest

from panelcast.select.tiers import (
    EffortTier,
    load_tiers,
    resolve_tier,
    tier_to_sweep_config,
)


class TestLoad:
    def test_shipped_defaults_when_file_missing(self, tmp_path):
        tiers = load_tiers(tmp_path / "nope.yaml")
        assert set(tiers) == {"quick", "standard", "thorough"}
        assert tiers["quick"].stages == (1,)
        assert not tiers["quick"].confirm
        assert tiers["standard"].include_stage2
        assert tiers["thorough"].stage3_fits == 8
        assert tiers["thorough"].publication_confirm["num_samples"] == 5000

    def test_yaml_overrides_field_by_field(self, tmp_path):
        path = tmp_path / "select.yaml"
        path.write_text(
            textwrap.dedent(
                """
                tiers:
                  quick:
                    num_samples: 250
                """
            ),
            encoding="utf-8",
        )
        tiers = load_tiers(path)
        assert tiers["quick"].num_samples == 250
        # Unspecified fields keep the shipped values.
        assert tiers["quick"].stages == (1,)
        assert tiers["standard"].num_samples == 1000

    def test_custom_tier_added(self, tmp_path):
        path = tmp_path / "select.yaml"
        path.write_text(
            "tiers:\n  exhaustive:\n    stages: [1, 2, 3]\n    stage3_fits: 40\n",
            encoding="utf-8",
        )
        tiers = load_tiers(path)
        assert "exhaustive" in tiers
        assert tiers["exhaustive"].stage3_fits == 40


class TestResolve:
    def test_unknown_tier_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown effort tier"):
            resolve_tier("turbo", tmp_path / "nope.yaml")

    def test_resolves_shipped(self, tmp_path):
        assert resolve_tier("standard", tmp_path / "nope.yaml").name == "standard"


class TestSweepConfigMapping:
    def test_quick_is_stage1_only(self, tmp_path):
        cfg = tier_to_sweep_config(
            EffortTier("quick", (1,), 2, 500, 500), sweep_id="s", output_root=tmp_path
        )
        assert cfg.include_stage2 is False
        assert cfg.stage3_fits == 0
        assert cfg.num_samples == 500

    def test_thorough_carries_stage3(self, tmp_path):
        tier = EffortTier("thorough", (1, 2, 3), 4, 1000, 1000, stage3_fits=8, confirm=True)
        cfg = tier_to_sweep_config(tier, sweep_id="s", output_root=tmp_path, max_fits=20)
        assert cfg.include_stage2 is True
        assert cfg.stage3_fits == 8
        assert cfg.max_fits == 20

    def test_stage3_fits_zeroed_when_stage3_absent(self, tmp_path):
        # A tier that sets stage3_fits but doesn't list stage 3 shouldn't run it.
        tier = EffortTier("weird", (1, 2), 4, 1000, 1000, stage3_fits=8)
        cfg = tier_to_sweep_config(tier, sweep_id="s", output_root=tmp_path)
        assert cfg.stage3_fits == 0

    def test_promote_z_flows_to_winner_gate(self, tmp_path):
        cfg = tier_to_sweep_config(
            EffortTier("standard", (1, 2), 4, 1000, 1000),
            sweep_id="s",
            output_root=tmp_path,
            promote_z=3.5,
        )
        assert cfg.winner_z == 3.5
