"""Successive-halving rung ladder (#164): promotion, ledger v2, sweep flow."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.select.runner import (
    ArmRecord,
    SweepConfig,
    SweepLedger,
    arm_id,
    ofat_arms,
    record_key,
    run_sweep,
    rung_survivors,
)
from panelcast.select.tiers import Rung, load_tiers, tier_to_sweep_config

AOTY = DatasetDescriptor()


# --- ladder config parsing ---------------------------------------------------


class TestRungParsing:
    def _tiers(self, tmp_path, tier_yaml: str):
        path = tmp_path / "select.yaml"
        path.write_text(tier_yaml, encoding="utf-8")
        return load_tiers(path)

    def test_rungs_parsed_from_yaml(self, tmp_path):
        tiers = self._tiers(
            tmp_path,
            yaml.safe_dump(
                {
                    "tiers": {
                        "standard": {
                            "rungs": [
                                {
                                    "num_chains": 2,
                                    "num_samples": 500,
                                    "num_warmup": 500,
                                    "keep_fraction": 0.4,
                                },
                                {"num_chains": 4, "num_samples": 1000, "num_warmup": 1000},
                            ]
                        }
                    }
                }
            ),
        )
        rungs = tiers["standard"].rungs
        assert rungs == (
            Rung(2, 500, 500, 0.4),
            Rung(4, 1000, 1000, None),
        )

    def test_missing_keep_fraction_on_screening_rung_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="keep_fraction"):
            self._tiers(
                tmp_path,
                yaml.safe_dump(
                    {
                        "tiers": {
                            "quick": {
                                "rungs": [
                                    {"num_chains": 2, "num_samples": 500, "num_warmup": 500},
                                    {"num_chains": 4, "num_samples": 1000, "num_warmup": 1000},
                                ]
                            }
                        }
                    }
                ),
            )

    def test_malformed_rung_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="rung"):
            self._tiers(
                tmp_path,
                yaml.safe_dump(
                    {"tiers": {"quick": {"rungs": [{"num_chains": 2}]}}}
                ),
            )

    def test_tier_without_rungs_stays_legacy(self, tmp_path):
        tiers = self._tiers(tmp_path, yaml.safe_dump({"tiers": {"quick": {"num_chains": 2}}}))
        assert tiers["quick"].rungs == ()

    def test_shipped_config_registers_the_ladder(self):
        tiers = load_tiers()  # repo configs/select.yaml
        assert tiers["standard"].rungs[0].keep_fraction == 0.4
        assert tiers["standard"].rungs[-1].keep_fraction is None

    def test_rung_instances_pass_through(self):
        from panelcast.select.tiers import _parse_rungs

        rungs = (Rung(2, 500, 500, 0.4), Rung(4, 1000, 1000, None))
        assert _parse_rungs("t", rungs) == rungs

    def test_tier_to_sweep_config_carries_rungs(self, tmp_path):
        tiers = load_tiers()
        cfg = tier_to_sweep_config(tiers["standard"], "s1")
        assert cfg.rungs is not None
        assert cfg.rungs[0]["num_samples"] == 500
        assert cfg.rungs[-1].get("keep_fraction") is None
        cfg_quick = tier_to_sweep_config(tiers["quick"], "s2")
        assert cfg_quick.rungs is None


# --- promotion rule ----------------------------------------------------------


def _rec(knobs, z, rung=0, stage=1, status="completed"):
    return ArmRecord(
        arm_id=arm_id(knobs),
        knobs=knobs,
        stage=stage,
        status=status,
        rung=rung,
        score={"z": z} if z is not None else None,
    )


class TestRungSurvivors:
    def test_top_fraction_by_z(self):
        records = [_rec({"a": i}, z=float(i)) for i in range(10)]
        out = rung_survivors(records, 0, 0.3, promote_z=2.0, screen_margin=0.0)
        kept = [knobs["a"] for knobs, _ in out]
        # top ceil(0.3 * 10) = 3 by z, plus z >= 2.0 rescues (a=2..6 overlap)
        assert kept[:3] == [9, 8, 7]

    def test_margin_rescues_near_threshold_arms(self):
        records = [
            _rec({"a": 1}, z=5.0),
            _rec({"a": 2}, z=1.8),  # below keep cut, inside promote_z - 0.5
            _rec({"a": 3}, z=0.2),
        ]
        out = rung_survivors(records, 0, 1 / 3, promote_z=2.0, screen_margin=0.5)
        kept = {knobs["a"] for knobs, _ in out}
        assert kept == {1, 2}
        notes = {knobs["a"]: note for knobs, note in out}
        assert "margin-rescued" in notes[2]

    def test_unscored_arms_never_promote(self):
        records = [_rec({"a": 1}, z=None), _rec({"a": 2}, z=0.1)]
        out = rung_survivors(records, 0, 1.0, promote_z=2.0, screen_margin=0.5)
        assert [knobs["a"] for knobs, _ in out] == [2]

    def test_nothing_scored_promotes_nothing(self):
        records = [_rec({"a": 1}, z=None), _rec({"a": 2}, z=None)]
        assert rung_survivors(records, 0, 0.5, promote_z=2.0, screen_margin=0.5) == []

    def test_keep_fraction_floor_is_one(self):
        records = [_rec({"a": 1}, z=-3.0)]
        out = rung_survivors(records, 0, 0.01, promote_z=2.0, screen_margin=0.0)
        assert len(out) == 1

    def test_only_matching_rung_and_stage_counted(self):
        records = [
            _rec({"a": 1}, z=9.0, rung=1),
            _rec({"a": 2}, z=1.0, rung=0),
            _rec({"a": 3}, z=8.0, rung=0, stage=2),
        ]
        out = rung_survivors(records, 0, 1.0, promote_z=2.0, screen_margin=0.0)
        assert [knobs["a"] for knobs, _ in out] == [2]


# --- ledger v2 ---------------------------------------------------------------


class TestLedgerV2:
    def test_v2_roundtrip_with_rungs(self, tmp_path):
        path = tmp_path / "ledger.json"
        ledger = SweepLedger(path)
        ledger.upsert(_rec({"a": 1}, z=1.0, rung=0))
        ledger.upsert(_rec({"a": 1}, z=2.0, rung=1))
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["version"] == 2
        reloaded = SweepLedger(path)
        aid = arm_id({"a": 1})
        assert set(reloaded.records) == {aid, f"{aid}@r1"}
        assert reloaded.records[f"{aid}@r1"].score == {"z": 2.0}

    def test_v1_payload_loads_as_rung_zero(self, tmp_path):
        path = tmp_path / "ledger.json"
        v1 = {
            "arms": [
                {
                    "arm_id": "abc123",
                    "knobs": {"a": 1},
                    "stage": 1,
                    "status": "completed",
                    "run_dir": "outputs/run_001",
                }
            ]
        }
        path.write_text(json.dumps(v1), encoding="utf-8")
        ledger = SweepLedger(path)
        assert ledger.records["abc123"].rung == 0
        assert ledger.completed_ids() == {"abc123"}

    def test_completed_ids_are_bare_arm_ids(self, tmp_path):
        ledger = SweepLedger(tmp_path / "ledger.json")
        ledger.upsert(_rec({"a": 1}, z=1.0, rung=1))
        assert ledger.completed_ids() == {arm_id({"a": 1})}

    def test_checkpoint_writes_current_state(self, tmp_path):
        path = tmp_path / "ledger.json"
        ledger = SweepLedger(path)
        rec = _rec({"a": 1}, z=None)
        ledger.records[arm_id({"a": 1})] = rec
        ledger.checkpoint()
        assert json.loads(path.read_text(encoding="utf-8"))["arms"]

    def test_ladder_requires_reference_first(self):
        with pytest.raises(ValueError, match="reference_first"):
            SweepConfig(
                sweep_id="p",
                reference_first=False,
                rungs=[{"num_chains": 2, "num_samples": 500, "keep_fraction": 0.4}],
            )


# --- plan arithmetic ---------------------------------------------------------


class TestPlanArithmetic:
    def test_ladder_fit_accounting(self):
        import math

        from panelcast.select.orchestrate import build_plan
        from panelcast.select.tiers import EffortTier

        tier = EffortTier(
            "standard",
            (1, 2),
            4,
            1000,
            1000,
            confirm=True,
            rungs=(Rung(2, 500, 500, 0.4), Rung(4, 1000, 1000, None)),
        )
        cfg = SweepConfig(sweep_id="p")
        plan = build_plan(AOTY, tier, cfg, "aoty")
        n_ofat = len(ofat_arms(AOTY))
        expected_rung1 = max(1, math.ceil(0.4 * n_ofat))
        confirm = 2 * 3
        assert plan.min_fits == (1 + n_ofat) + (1 + expected_rung1) + confirm
        # worst case: margin rescues promote everything at every rung
        stage2_upper = 1 + math.comb(3, 2)
        assert plan.max_fits_planned == 2 * (1 + n_ofat) + stage2_upper + confirm
        assert any("rung ladder" in n for n in plan.notes)

    def test_legacy_plan_unchanged(self):
        from panelcast.select.orchestrate import build_plan
        from panelcast.select.tiers import EffortTier

        tier = EffortTier("quick", (1,), 2, 500, 500)
        plan = build_plan(AOTY, tier, SweepConfig(sweep_id="p"), "aoty")
        n_ofat = len(ofat_arms(AOTY))
        assert plan.min_fits == 1 + n_ofat
        assert plan.max_fits_planned == 1 + n_ofat


# --- sweep flow --------------------------------------------------------------


def _write_manifest(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"created_at": datetime.now().isoformat()}), encoding="utf-8"
    )


def _fake_env(tmp_path, monkeypatch):
    launches: list[Path] = []

    def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None) -> tuple[int, str]:
        launches.append(Path(config_path))
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        _write_manifest(tmp_path / "outputs" / payload["run_id"])
        return 0, "ok"

    cfg = SweepConfig(
        sweep_id="t",
        output_root=tmp_path / "select",
        panelcast_bin="pc",
        include_stage2=False,
        pipeline_output_base=tmp_path / "outputs",
        rungs=[
            {"num_chains": 2, "num_samples": 500, "num_warmup": 500, "keep_fraction": 0.25},
            {"num_chains": 4, "num_samples": 1000, "num_warmup": 1000},
        ],
    )
    return cfg, launches, launch


def _z_scorer():
    """Deterministic z per arm id — stable across rungs, unique per arm."""

    def scorer(run_dir: Path, reference_run: Path | None) -> dict:
        return {"z": None, "elpd_diff": None} if reference_run is None else {"z": 1.0}

    return scorer


class TestLadderSweep:
    def test_two_rung_flow_and_keys(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        n_ofat = len(ofat_arms(AOTY))
        zs = iter(range(1000))

        def scorer(run_dir, reference_run):
            # Reference scores itself first (no reference yet) -> None z.
            if reference_run is None:
                return {"z": None}
            return {"z": float(next(zs) % 7)}  # spread of z values 0..6

        ledger = run_sweep(cfg, AOTY, launch=launch, scorer=scorer)

        rung0 = [r for r in ledger.records.values() if r.rung == 0]
        rung1 = [r for r in ledger.records.values() if r.rung == 1]
        assert len(rung0) == 1 + n_ofat
        # promoted: top ceil(.25 * scored) plus z >= 2.0 - 0.5 rescues, + reference refit
        assert 1 < len(rung1) < len(rung0)
        assert all(record_key(r.arm_id, 1) in ledger.records for r in rung1)
        # every rung-1 record's key carries the suffix
        for r in rung1:
            assert ledger.records[f"{r.arm_id}@r1"] is r

    def test_rung_sampler_overrides_written_to_configs(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        run_sweep(cfg, AOTY, launch=launch, scorer=_z_scorer())
        rung0_cfg = yaml.safe_load(
            next(p for p in launches if p.name.endswith(".yaml") and "@r" not in p.name)
            .read_text(encoding="utf-8")
        )
        assert rung0_cfg["num_samples"] == 500
        rung1_paths = [p for p in launches if "@r1" in p.name]
        assert rung1_paths, "no rung-1 arm was launched"
        rung1_cfg = yaml.safe_load(rung1_paths[0].read_text(encoding="utf-8"))
        assert rung1_cfg["num_samples"] == 1000

    def test_resume_skips_completed_rungs(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        run_sweep(cfg, AOTY, launch=launch, scorer=_z_scorer())
        first = len(launches)
        run_sweep(cfg, AOTY, launch=launch, scorer=_z_scorer())
        assert len(launches) == first

    def test_legacy_single_scale_untouched(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.rungs = None
        ledger = run_sweep(cfg, AOTY, launch=launch, scorer=_z_scorer())
        assert len(launches) == 1 + len(ofat_arms(AOTY))
        assert all(r.rung == 0 for r in ledger.records.values())
        assert all("@r" not in key for key in ledger.records)

    def test_ladder_halts_when_nothing_scores(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        ledger = run_sweep(cfg, AOTY, launch=launch, scorer=lambda run_dir, ref: {"z": None})
        assert not [r for r in ledger.records.values() if r.rung == 1]
        assert len(launches) == 1 + len(ofat_arms(AOTY))

    def test_ladder_truncates_at_rung_boundary(self, tmp_path, monkeypatch):
        import panelcast.select.runner as runner_mod

        monkeypatch.setattr(
            runner_mod, "ofat_arms",
            lambda d, available_columns=None: [
                ({"ar_center": "none"}, None),
                ({"latent_process": "ar1"}, None),
            ],
        )
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.max_fits = 3  # rung 0 exactly (ref + 2 arms); the rung-1 refit hits the cap
        ledger = run_sweep(cfg, AOTY, launch=launch, scorer=_z_scorer())
        assert ledger.fits_done() == 3
        assert not [r for r in ledger.records.values() if r.rung == 1]

    def test_failed_reference_stops_the_ladder(self, tmp_path, monkeypatch):
        # A dead rung-0 reference means nothing can score or promote; the
        # ladder must stop rather than spend the whole screening budget
        # (release-audit Y1).
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)

        def failing_launch(config_path, panelcast_bin, timeout_seconds=None):
            return 1, "CUDA error: device-side assert triggered"

        ledger = run_sweep(cfg, AOTY, launch=failing_launch, scorer=_z_scorer())
        assert len(ledger.records) == 1
        assert next(iter(ledger.records.values())).status == "failed"
