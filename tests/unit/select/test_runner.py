"""Sweep runner: arm generation, ledger resume, budget truncation, serial cache discipline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.select.runner import (
    ArmRecord,
    SweepConfig,
    SweepLedger,
    arm_id,
    complete_arm,
    diagnose_data,
    feature_signature,
    ofat_arms,
    reorder_arms,
    run_sweep,
    stage2_arms,
    stage3_arms,
)
from panelcast.select.space import default_arm

AOTY = DatasetDescriptor()


class TestArmIdentity:
    def test_key_order_is_irrelevant(self):
        assert arm_id({"latent_process": "ar1", "gbm_offset": False}) == arm_id(
            {"gbm_offset": False, "latent_process": "ar1"}
        )

    def test_explicit_default_equals_omitted(self):
        base = default_arm()
        assert arm_id({}) == arm_id({"latent_process": base["latent_process"]})

    def test_different_arms_differ(self):
        assert arm_id({"latent_process": "ar1"}) != arm_id({})


class TestCompletion:
    def test_bounded_family_gets_identity_companion(self):
        arm, note = complete_arm({"likelihood_family": "beta"}, AOTY)
        assert arm["target_transform"] == "identity"
        assert "structural" in note

    def test_valid_arm_untouched(self):
        arm, note = complete_arm({"latent_process": "ar1"}, AOTY)
        assert arm == {"latent_process": "ar1"}
        assert note is None


class TestOfat:
    def test_every_arm_is_valid_and_non_base(self):
        from panelcast.select.space import arm_conflicts

        arms = ofat_arms(AOTY)
        assert arms
        for arm, _note in arms:
            assert arm, "empty arm generated"
            assert arm_conflicts(arm, AOTY) == []

    def test_covers_the_whole_surface(self):
        arms = ofat_arms(AOTY)
        varied = {name for arm, _ in arms for name in arm}
        assert varied == {k for k, _ in _expected_variations()}

    def test_expected_arm_count(self):
        assert len(ofat_arms(AOTY)) == sum(n for _, n in _expected_variations())

    def test_inert_prior_knob_paired_with_enabler(self):
        arms = [a for a, _ in ofat_arms(AOTY) if "n_exponent_prior" in a]
        assert arms == [{"learn_n_exponent": True, "n_exponent_prior": "beta"}]

    def test_ids_are_unique(self):
        ids = [arm_id(a) for a, _ in ofat_arms(AOTY)]
        assert len(ids) == len(set(ids))


def _expected_variations() -> list[tuple[str, int]]:
    # (knob, expected stage-1 arms) on the AOTY descriptor.
    return [
        ("target_transform", 1),
        ("latent_process", 1),
        ("likelihood_family", 8),
        ("sigma_obs_prior_type", 1),
        ("ar_center", 2),
        ("debut_prev_score_source", 1),
        ("n_exponent_prior", 1),
        ("learn_n_exponent", 1),
        ("discretize_observation", 1),
        ("heteroscedastic_entity_obs", 1),
        ("errors_in_variables", 1),
        ("propagate_rw_horizon", 1),
        ("entity_group_pooling", 2),
        ("gbm_offset", 1),
        ("enable_genre", 1),
        ("enable_artist", 1),
        ("enable_temporal", 1),
    ]


class TestDiagnostics:
    def _frame(self, y, entities=None, n_obs=None) -> pd.DataFrame:
        n = len(y)
        return pd.DataFrame(
            {
                "Artist": entities if entities is not None else [f"a{i % 10}" for i in range(n)],
                "User_Score": y,
                "User_Ratings": n_obs if n_obs is not None else np.full(n, 100),
            }
        )

    def test_skew_detected(self):
        rng = np.random.default_rng(0)
        skewed = 100 - rng.exponential(10, 400)
        assert diagnose_data(self._frame(skewed), AOTY)["target_skewed"]

    def test_integer_heaping_detected(self):
        y = np.round(np.random.default_rng(1).normal(70, 8, 300))
        assert diagnose_data(self._frame(y), AOTY)["integer_heaped"]

    def test_sparse_histories_detected(self):
        y = np.random.default_rng(2).normal(70, 8, 40)
        entities = [f"e{i}" for i in range(40)]
        assert diagnose_data(self._frame(y, entities=entities), AOTY)["sparse_histories"]

    def test_obs_count_spread_detected(self):
        rng = np.random.default_rng(3)
        y = rng.normal(70, 8, 300)
        counts = rng.lognormal(3, 1.5, 300).astype(int) + 1
        assert diagnose_data(self._frame(y, n_obs=counts), AOTY)["obs_count_spread"]

    def test_reorder_floats_prioritized_knobs(self):
        arms = ofat_arms(AOTY)
        ordered = reorder_arms(arms, {"integer_heaped": True})
        assert "discretize_observation" in ordered[0][0]


class TestStage2And3:
    def test_stage2_composes_and_probes_pairwise(self):
        winners = [{"latent_process": "ar1"}, {"gbm_offset": False}, {"ar_center": "none"}]
        arms = stage2_arms(winners, AOTY)
        combined = {"latent_process": "ar1", "gbm_offset": False, "ar_center": "none"}
        assert any(a == combined for a, _ in arms)
        assert sum(1 for a, _ in arms if len(a) == 2) == 3

    def test_stage2_skips_already_seen(self):
        winners = [{"latent_process": "ar1"}, {"gbm_offset": False}]
        seen = {arm_id({"latent_process": "ar1", "gbm_offset": False})}
        assert stage2_arms(winners, AOTY, seen=seen) == []

    def test_stage3_is_deterministic_and_valid(self):
        from panelcast.select.space import arm_conflicts

        first = stage3_arms(AOTY, 5, "sweep-42")
        second = stage3_arms(AOTY, 5, "sweep-42")
        assert [a for a, _ in first] == [a for a, _ in second]
        for arm, _ in first:
            assert arm_conflicts(arm, AOTY) == []


class TestFeatureSignature:
    def test_feature_knobs_change_signature(self):
        base = default_arm()
        assert feature_signature({**base, "gbm_offset": False}) != feature_signature(base)
        assert feature_signature({**base, "enable_genre": False}) != feature_signature(base)

    def test_model_knobs_do_not(self):
        base = default_arm()
        assert feature_signature({**base, "latent_process": "ar1"}) == feature_signature(base)


class TestLedger:
    def test_roundtrip(self, tmp_path):
        ledger = SweepLedger(tmp_path / "ledger.json")
        ledger.upsert(ArmRecord(arm_id="abc", knobs={"x": 1}, stage=1, status="completed"))
        reloaded = SweepLedger(tmp_path / "ledger.json")
        assert reloaded.completed_ids() == {"abc"}
        assert reloaded.records["abc"].knobs == {"x": 1}

    def test_hours_and_fits_accounting(self, tmp_path):
        ledger = SweepLedger(tmp_path / "ledger.json")
        ledger.upsert(
            ArmRecord("a", {}, 1, status="completed", wall_clock_seconds=1800.0)
        )
        ledger.upsert(ArmRecord("b", {}, 1, status="failed", wall_clock_seconds=1800.0))
        assert ledger.fits_done() == 2
        assert ledger.hours_spent() == 1.0


def _fake_env(tmp_path, monkeypatch):
    """Fake launcher + latest-run plumbing; returns (cfg, launches list)."""
    launches: list[Path] = []
    run_dirs = iter(f"run_{i:03d}" for i in range(999))

    def launch(config_path: Path, panelcast_bin: str) -> tuple[int, str]:
        launches.append(Path(config_path))
        (tmp_path / "outputs").mkdir(exist_ok=True)
        current = next(run_dirs)
        (tmp_path / "outputs" / current).mkdir()
        (tmp_path / "outputs" / "latest.json").write_text(
            json.dumps({"run_dir": current}), encoding="utf-8"
        )
        return 0, "ok"

    import panelcast.paths as paths_mod

    monkeypatch.setattr(
        paths_mod, "resolve_latest", lambda output_base=Path("outputs"): _latest(tmp_path)
    )
    cfg = SweepConfig(sweep_id="t", output_root=tmp_path / "select", panelcast_bin="pc")
    return cfg, launches, launch


def _latest(tmp_path: Path) -> Path | None:
    try:
        data = json.loads((tmp_path / "outputs" / "latest.json").read_text(encoding="utf-8"))
        return tmp_path / "outputs" / data["run_dir"]
    except OSError:
        return None


class TestRunSweep:
    def test_reference_plus_ofat_all_complete(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        ledger = run_sweep(cfg, AOTY, launch=launch)
        statuses = {r.status for r in ledger.records.values()}
        assert statuses == {"completed"}
        assert len(launches) == len(ofat_arms(AOTY)) + 1
        assert all(r.run_dir for r in ledger.records.values())

    def test_resume_skips_completed(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        run_sweep(cfg, AOTY, launch=launch)
        first_count = len(launches)
        run_sweep(cfg, AOTY, launch=launch)
        assert len(launches) == first_count

    def test_max_fits_truncates(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.max_fits = 3
        cfg.include_stage2 = False
        ledger = run_sweep(cfg, AOTY, launch=launch)
        assert ledger.fits_done() == 3

    def test_failed_arm_recorded_and_sweep_continues(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        calls = {"n": 0}

        def flaky(config_path: Path, panelcast_bin: str) -> tuple[int, str]:
            calls["n"] += 1
            if calls["n"] == 2:
                return 1, "boom"
            return launch(config_path, panelcast_bin)

        ledger = run_sweep(cfg, AOTY, launch=flaky)
        failed = [r for r in ledger.records.values() if r.status == "failed"]
        assert len(failed) == 1
        assert failed[0].error == "boom"
        assert len(ledger.records) == len(ofat_arms(AOTY)) + 1

    def test_stage2_runs_for_scored_winners(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = True
        cfg.max_fits = len(ofat_arms(AOTY)) + 4

        def scorer(run_dir: Path, reference: Path | None) -> dict:
            return {"z": 5.0}

        ledger = run_sweep(cfg, AOTY, launch=launch, scorer=scorer)
        assert any(r.stage == 2 for r in ledger.records.values())

    def test_no_stage2_without_winners(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = True
        ledger = run_sweep(cfg, AOTY, launch=launch, scorer=lambda r, ref: {"z": 0.0})
        assert not any(r.stage == 2 for r in ledger.records.values())

    def test_arm_config_written_with_stages_and_knobs(self, tmp_path, monkeypatch):
        import yaml

        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.max_fits = 2
        cfg.num_samples = 111
        run_sweep(cfg, AOTY, launch=launch)
        payload = yaml.safe_load(launches[0].read_text(encoding="utf-8"))
        assert payload["stages"] == ["splits", "features", "train", "evaluate"]
        assert payload["num_samples"] == 111
        assert "likelihood_family" in payload

    def test_model_only_arm_reuses_feature_cache(self, tmp_path, monkeypatch):
        import yaml

        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.max_fits = 3
        run_sweep(cfg, AOTY, launch=launch)
        stage_lists = [
            yaml.safe_load(p.read_text(encoding="utf-8"))["stages"] for p in launches
        ]
        assert stage_lists[0] == ["splits", "features", "train", "evaluate"]
        # Reference and the first OFAT arms are model-only under default features.
        assert ["train", "evaluate"] in stage_lists[1:]


class TestDiagnosticsAreOrderingOnly:
    def test_diagnostics_never_change_the_arm_set(self):
        arms = ofat_arms(AOTY)
        reordered = reorder_arms(arms, {k: True for k in ("target_skewed", "integer_heaped")})
        assert sorted(str(a) for a, _ in arms) == sorted(str(a) for a, _ in reordered)
