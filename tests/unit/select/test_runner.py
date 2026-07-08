"""Sweep runner: arm generation, ledger resume, budget truncation, serial cache discipline."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.select.runner import (
    ARM_TIMEOUT_RETURNCODE,
    STAGE2_MAX_WINNERS,
    ArmRecord,
    SweepConfig,
    SweepLedger,
    arm_id,
    complete_arm,
    diagnose_data,
    feature_signature,
    ofat_arms,
    reorder_arms,
    resolve_arm_timeout,
    run_sweep,
    stage2_arms,
    stage2_winners,
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


def _write_manifest(run_dir: Path, created_at: str | None = None, **extra) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": created_at or datetime.now().isoformat(), **extra}
    (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _fake_env(tmp_path, monkeypatch):
    """Fake launcher + latest-run plumbing; returns (cfg, launches list)."""
    launches: list[Path] = []
    run_dirs = iter(f"run_{i:03d}" for i in range(999))

    def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None) -> tuple[int, str]:
        launches.append(Path(config_path))
        (tmp_path / "outputs").mkdir(exist_ok=True)
        current = next(run_dirs)
        _write_manifest(tmp_path / "outputs" / current)
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

        def flaky(config_path: Path, panelcast_bin: str, timeout_seconds=None) -> tuple[int, str]:
            calls["n"] += 1
            if calls["n"] == 2:
                return 1, "boom"
            return launch(config_path, panelcast_bin, timeout_seconds)

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


class TestArmTimeout:
    def test_launch_arm_passes_timeout_through(self, tmp_path, monkeypatch):
        import panelcast.select.runner as runner_mod

        captured: dict = {}

        class _Proc:
            returncode = 0
            stdout = "out"
            stderr = "err"

        def fake_run(cmd, **kwargs):
            captured.update(kwargs)
            return _Proc()

        monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)
        code, tail = runner_mod.launch_arm(tmp_path / "arm.yaml", "pc", timeout_seconds=123)
        assert captured["timeout"] == 123
        assert code == 0

    def test_launch_arm_timeout_returns_kill_tuple(self, tmp_path, monkeypatch):
        import panelcast.select.runner as runner_mod

        def fake_run(cmd, **kwargs):
            raise runner_mod.subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

        monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)
        code, tail = runner_mod.launch_arm(tmp_path / "arm.yaml", "pc", timeout_seconds=5)
        assert code == ARM_TIMEOUT_RETURNCODE
        assert "exceeded timeout of 5s" in tail

    def test_timed_out_arm_is_marked_timeout_and_sweep_continues(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.arm_timeout_seconds = 1800.0
        calls = {"n": 0}

        def timing_out(config_path, panelcast_bin, timeout_seconds=None):
            calls["n"] += 1
            if calls["n"] == 2:
                return (
                    ARM_TIMEOUT_RETURNCODE,
                    f"arm exceeded timeout of {timeout_seconds}s and was killed",
                )
            return launch(config_path, panelcast_bin, timeout_seconds)

        ledger = run_sweep(cfg, AOTY, launch=timing_out)
        timed_out = [r for r in ledger.records.values() if r.status == "timeout"]
        assert len(timed_out) == 1
        assert "exceeded timeout of 1800.0s" in timed_out[0].error
        assert not any(r.status == "failed" for r in ledger.records.values())
        # The sweep did not stall: every planned arm still has a record.
        assert len(ledger.records) == len(ofat_arms(AOTY)) + 1

    def test_resume_skips_timeout_but_reruns_failed(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False

        # Two OFAT arms stand in for the terminal (timeout) and retryable (failed)
        # cases; seed the ledger directly so resume decides purely on status.
        ofat = ofat_arms(AOTY)
        timeout_arm = ofat[0][0]
        failed_arm = ofat[1][0]
        timeout_id = arm_id(timeout_arm)
        failed_id = arm_id(failed_arm)
        ledger = SweepLedger(cfg.sweep_dir / "ledger.json")
        cfg.sweep_dir.mkdir(parents=True, exist_ok=True)
        ledger.upsert(ArmRecord(timeout_id, timeout_arm, 1, status="timeout"))
        ledger.upsert(ArmRecord(failed_id, failed_arm, 1, status="failed"))

        def guarded(config_path, panelcast_bin, timeout_seconds=None):
            aid = config_path.stem.removeprefix("arm_")
            if aid == timeout_id:
                raise AssertionError("timed-out arm must not be re-launched on resume")
            return launch(config_path, panelcast_bin, timeout_seconds)

        relaunched = run_sweep(cfg, AOTY, launch=guarded)
        assert relaunched.records[timeout_id].status == "timeout"
        # The failed arm is retryable: resume re-ran it and it completed this time.
        assert relaunched.records[failed_id].status == "completed"


class TestAdaptiveTimeout:
    DIMS = {"n_observations": 5000, "n_features": 40, "n_artists": 900, "max_seq": 30}

    def _stub_predict(self, monkeypatch, seconds, source="local history (offset_logit), n=3"):
        import panelcast.gpu_memory.runtime_predictor as rp

        calls: list[dict] = []

        def fake(num_chains, num_samples, num_warmup, n_obs, transform=None, store_path=None):
            calls.append({"transform": transform, "n_obs": n_obs})
            return rp.RuntimePrediction(seconds=seconds, source=source)

        monkeypatch.setattr(rp, "predict_fit_seconds", fake)
        return calls

    def _auto_cfg(self, tmp_path) -> SweepConfig:
        return SweepConfig(
            sweep_id="t", output_root=tmp_path / "select", panelcast_bin="pc",
            arm_timeout_seconds="auto", num_chains=4, num_samples=1000, num_warmup=1000,
        )

    def test_auto_is_multiplier_times_predicted(self, tmp_path, monkeypatch):
        self._stub_predict(monkeypatch, seconds=6000.0)
        timeout, prediction = resolve_arm_timeout(self._auto_cfg(tmp_path), default_arm(), self.DIMS)
        assert timeout == 3.0 * 6000.0
        assert prediction.seconds == 6000.0

    def test_floor_wins_for_cheap_arms(self, tmp_path, monkeypatch):
        self._stub_predict(monkeypatch, seconds=100.0)
        timeout, _ = resolve_arm_timeout(self._auto_cfg(tmp_path), default_arm(), self.DIMS)
        assert timeout == 1800.0

    def test_explicit_numeric_never_predicts(self, tmp_path, monkeypatch):
        calls = self._stub_predict(monkeypatch, seconds=6000.0)
        cfg = self._auto_cfg(tmp_path)
        cfg.arm_timeout_seconds = 900.0
        assert resolve_arm_timeout(cfg, default_arm(), self.DIMS) == (900.0, None)
        assert calls == []

    def test_auto_without_dims_falls_back_to_floor(self, tmp_path, monkeypatch):
        calls = self._stub_predict(monkeypatch, seconds=6000.0)
        assert resolve_arm_timeout(self._auto_cfg(tmp_path), default_arm(), None) == (1800.0, None)
        assert calls == []

    def test_prediction_uses_the_arms_own_transform(self, tmp_path, monkeypatch):
        calls = self._stub_predict(monkeypatch, seconds=6000.0)
        merged = {**default_arm(), "target_transform": "identity"}
        resolve_arm_timeout(self._auto_cfg(tmp_path), merged, self.DIMS)
        assert calls[-1]["transform"] == "identity"

    def test_sweep_records_prediction_and_timeout(self, tmp_path, monkeypatch):
        self._stub_predict(monkeypatch, seconds=6000.0)
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.max_fits = 2
        cfg.arm_timeout_seconds = "auto"
        seen_timeouts: list = []

        def capturing(config_path, panelcast_bin, timeout_seconds=None):
            seen_timeouts.append(timeout_seconds)
            return launch(config_path, panelcast_bin, timeout_seconds)

        ledger = run_sweep(cfg, AOTY, launch=capturing, dims=self.DIMS)
        assert seen_timeouts == [18000.0, 18000.0]
        for record in ledger.records.values():
            assert record.predicted_seconds == 6000.0
            assert record.timeout_seconds_used == 18000.0

    def test_prediction_source_is_logged(self, tmp_path, monkeypatch):
        from structlog.testing import capture_logs

        source = "cold-start planning numbers (RTX 5090, offset_logit)"
        self._stub_predict(monkeypatch, seconds=6000.0, source=source)
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.max_fits = 1
        cfg.arm_timeout_seconds = "auto"
        with capture_logs() as logs:
            run_sweep(cfg, AOTY, launch=launch, dims=self.DIMS)
        events = [e for e in logs if e["event"] == "arm_timeout_auto"]
        assert events and events[0]["source"] == source
        assert events[0]["timeout_seconds"] == 18000.0

    def test_old_ledger_entry_without_timeout_fields_loads(self, tmp_path):
        # A 0.7.x ledger predates predicted_seconds/timeout_seconds_used;
        # resume must load it unchanged.
        old_entry = {
            "arm_id": "abc", "knobs": {"latent_process": "ar1"}, "stage": 1,
            "status": "completed", "run_dir": None, "wall_clock_seconds": 12.0,
            "error": None, "score": {"z": 3.0}, "note": None,
        }
        record = ArmRecord(**old_entry)
        assert record.predicted_seconds is None
        assert record.timeout_seconds_used is None
        path = tmp_path / "ledger.json"
        path.write_text(json.dumps({"arms": [old_entry]}), encoding="utf-8")
        assert SweepLedger(path).records["abc"].status == "completed"


class TestStage2Winners:
    def _record(self, knobs, z, status="completed", stage=1) -> ArmRecord:
        score = None if z == "absent" else {"z": z}
        return ArmRecord(arm_id(knobs), knobs, stage, status=status, score=score)

    def test_none_z_is_never_a_winner(self):
        records = [
            self._record({"latent_process": "ar1"}, None),
            self._record({"gbm_offset": False}, "absent"),
        ]
        assert stage2_winners(records, winner_z=2.0) == []

    def test_five_winners_capped_to_top_three_by_z(self):
        knobs = [
            {"latent_process": "ar1"},
            {"gbm_offset": False},
            {"ar_center": "none"},
            {"learn_n_exponent": True},
            {"errors_in_variables": True},
        ]
        records = [self._record(k, z) for k, z in zip(knobs, (3.0, 7.0, 5.0, 4.0, 6.0))]
        winners = stage2_winners(records, winner_z=2.0)
        assert len(winners) == STAGE2_MAX_WINNERS == 3
        assert winners == [knobs[1], knobs[4], knobs[2]]  # z 7, 6, 5
        # The composed+pairwise bound the plan promises: 1 + C(3, 2).
        assert len(stage2_arms(winners, AOTY)) <= 1 + 3

    def test_tie_breaks_on_arm_id(self):
        a, b = {"latent_process": "ar1"}, {"gbm_offset": False}
        records = [self._record(a, 5.0), self._record(b, 5.0)]
        winners = stage2_winners(records, winner_z=2.0, cap=1)
        expected_first = min((arm_id(a), a), (arm_id(b), b))[1]
        assert winners == [expected_first]

    def test_sweep_survives_none_z_scores(self, tmp_path, monkeypatch):
        # The production scorer returns {"z": None} whenever the reference
        # snapshot is missing; the stage-2 gate must not TypeError on it.
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = True
        ledger = run_sweep(cfg, AOTY, launch=launch, scorer=lambda r, ref: {"z": None})
        assert all(r.status == "completed" for r in ledger.records.values())
        assert not any(r.stage == 2 for r in ledger.records.values())

    def test_sweep_stage2_count_stays_within_plan_bound(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = True
        counter = {"n": 0}

        def scorer(run_dir, reference):
            counter["n"] += 1
            return {"z": float(counter["n"])}  # every arm clears the bar

        ledger = run_sweep(cfg, AOTY, launch=launch, scorer=scorer)
        n_stage2 = sum(1 for r in ledger.records.values() if r.stage == 2)
        assert 0 < n_stage2 <= 1 + 3


class TestAttribution:
    def test_run_dir_is_dereferenced(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.max_fits = 1

        import panelcast.paths as paths_mod

        real = tmp_path / "outputs" / "run_000"

        def launch_indirect(config_path, panelcast_bin, timeout_seconds=None):
            _write_manifest(real)
            return 0, "ok"

        monkeypatch.setattr(
            paths_mod,
            "resolve_latest",
            lambda output_base=Path("outputs"): tmp_path / "outputs" / ".." / "outputs" / "run_000",
        )
        ledger = run_sweep(cfg, AOTY, launch=launch_indirect)
        (record,) = ledger.records.values()
        assert record.run_dir == str(real.resolve())

    def test_stale_run_is_not_scored(self, tmp_path, monkeypatch):
        # latest.json still points at a run created BEFORE this arm launched
        # (failed pointer write / foreign run): the arm must fail, not score it.
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.max_fits = 1
        stale = tmp_path / "outputs" / "stale_run"
        _write_manifest(stale, created_at=(datetime.now() - timedelta(hours=2)).isoformat())
        (tmp_path / "outputs" / "latest.json").write_text(
            json.dumps({"run_dir": "stale_run"}), encoding="utf-8"
        )

        def launch_no_pointer(config_path, panelcast_bin, timeout_seconds=None):
            return 0, "ok"  # succeeds but never re-points latest.json

        ledger = run_sweep(cfg, AOTY, launch=launch_no_pointer)
        (record,) = ledger.records.values()
        assert record.status == "failed"
        assert "attribution failed" in record.error
        assert record.run_dir is None

    def test_config_mismatch_is_not_scored(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.max_fits = 2

        def launch_foreign(config_path, panelcast_bin, timeout_seconds=None):
            code, tail = launch(config_path, panelcast_bin, timeout_seconds)
            run_dir = tmp_path / "outputs" / json.loads(
                (tmp_path / "outputs" / "latest.json").read_text()
            )["run_dir"]
            # A foreign run's manifest records different knob values.
            _write_manifest(run_dir, flags={"likelihood_family": "not-a-real-family"})
            return code, tail

        ledger = run_sweep(cfg, AOTY, launch=launch_foreign)
        assert all(r.status == "failed" for r in ledger.records.values())
        assert all("disagrees with the arm" in r.error for r in ledger.records.values())

    def test_two_arms_resolving_to_same_run_fail_the_second(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.max_fits = 2
        state = {"n": 0}

        def launch_once(config_path, panelcast_bin, timeout_seconds=None):
            state["n"] += 1
            if state["n"] == 1:
                return launch(config_path, panelcast_bin, timeout_seconds)
            # Second arm: pointer write fails silently, latest still points at
            # arm 1's run — but a fresh manifest timestamp alone must not pass.
            run_dir = tmp_path / "outputs" / json.loads(
                (tmp_path / "outputs" / "latest.json").read_text()
            )["run_dir"]
            _write_manifest(run_dir)
            return 0, "ok"

        ledger = run_sweep(cfg, AOTY, launch=launch_once)
        statuses = [r.status for r in ledger.records.values()]
        assert statuses.count("completed") == 1
        assert statuses.count("failed") == 1
        failed = next(r for r in ledger.records.values() if r.status == "failed")
        assert "already belongs to another arm" in failed.error


class _FakePrediction:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.source = "fake"


class TestBudgetLookahead:
    def _expensive_ar1(self, monkeypatch):
        import panelcast.select.runner as runner_mod

        def fake_predict(cfg, merged, dims):
            if dims is None:
                return None
            return _FakePrediction(7200.0 if merged.get("latent_process") == "ar1" else 60.0)

        monkeypatch.setattr(runner_mod, "_predict_arm_seconds", fake_predict)

    def test_expensive_arm_skipped_cheap_arms_run(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.budget_hours = 1.0
        self._expensive_ar1(monkeypatch)
        ledger = run_sweep(cfg, AOTY, dims={"n_observations": 100}, launch=launch)
        skipped = [r for r in ledger.records.values() if r.status == "skipped_budget"]
        assert [r.knobs for r in skipped] == [{"latent_process": "ar1"}]
        assert "exceeds remaining budget" in skipped[0].error
        assert skipped[0].predicted_seconds == 7200.0
        completed = [r for r in ledger.records.values() if r.status == "completed"]
        assert len(completed) == len(ofat_arms(AOTY))  # reference + all but the skipped arm
        assert len(launches) == len(completed)

    def test_skipped_budget_retried_under_bigger_budget(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.budget_hours = 1.0
        self._expensive_ar1(monkeypatch)
        run_sweep(cfg, AOTY, dims={"n_observations": 100}, launch=launch)
        cfg.budget_hours = 10.0
        ledger = run_sweep(cfg, AOTY, dims={"n_observations": 100}, launch=launch)
        record = ledger.records[arm_id({"latent_process": "ar1"})]
        assert record.status == "completed"

    def test_exhausted_budget_marks_all_remaining_skipped(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.budget_hours = 0.5
        cfg.sweep_dir.mkdir(parents=True, exist_ok=True)
        (cfg.sweep_dir / "ledger.json").write_text(
            json.dumps(
                {"arms": [{"arm_id": "prior", "knobs": {"x": 1}, "stage": 1,
                           "status": "completed", "wall_clock_seconds": 3600.0}]}
            ),
            encoding="utf-8",
        )
        ledger = run_sweep(cfg, AOTY, launch=launch)
        assert launches == []
        skipped = [r for r in ledger.records.values() if r.status == "skipped_budget"]
        assert len(skipped) == len(ofat_arms(AOTY)) + 1  # reference included
        assert all("budget exhausted" in r.error for r in skipped)

    def test_failed_record_not_clobbered_by_budget_skip(self, tmp_path, monkeypatch):
        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.budget_hours = 0.5
        aid = arm_id({"latent_process": "ar1"})
        cfg.sweep_dir.mkdir(parents=True, exist_ok=True)
        (cfg.sweep_dir / "ledger.json").write_text(
            json.dumps(
                {"arms": [
                    {"arm_id": "prior", "knobs": {"x": 1}, "stage": 1,
                     "status": "completed", "wall_clock_seconds": 3600.0},
                    {"arm_id": aid, "knobs": {"latent_process": "ar1"}, "stage": 1,
                     "status": "failed", "error": "boom"},
                ]}
            ),
            encoding="utf-8",
        )
        ledger = run_sweep(cfg, AOTY, launch=launch)
        assert ledger.records[aid].status == "failed"
        assert ledger.records[aid].error == "boom"


class TestReorderCost:
    def test_cost_breaks_ties_within_priority(self):
        arms = [({"a": 1}, None), ({"b": 1}, None), ({"c": 1}, None)]
        cost = {"a": 30.0, "b": 10.0, "c": 20.0}
        ordered = reorder_arms(arms, {}, cost=lambda arm: cost[next(iter(arm))])
        assert [next(iter(a)) for a, _ in ordered] == ["b", "c", "a"]

    def test_diagnostics_dominate_cost(self):
        arms = [({"target_transform": "identity"}, None), ({"gbm_offset": False}, None)]
        ordered = reorder_arms(
            arms,
            {"target_skewed": True},
            cost=lambda arm: 100.0 if "target_transform" in arm else 1.0,
        )
        assert "target_transform" in ordered[0][0]

    def test_no_cost_keeps_enumeration_order(self):
        arms = [({"a": 1}, None), ({"b": 1}, None)]
        assert reorder_arms(arms, {}) == arms


class TestWarmupTransferWiring:
    """Reference exports adaptation; later arms import at reduced warmup (#178)."""

    def _configs_written(self, tmp_path, monkeypatch, transfer: bool):
        import yaml

        cfg, launches, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.max_fits = 3
        cfg.warmup_transfer = transfer

        def launch_and_touch(config_path, panelcast_bin, timeout_seconds=None):
            code, tail = launch(config_path, panelcast_bin, timeout_seconds)
            # The reference subprocess would write the export; fake it.
            payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            if "warmup_export_path" in payload:
                Path(payload["warmup_export_path"]).write_bytes(b"x")
            return code, tail

        ledger = run_sweep(cfg, AOTY, launch=launch_and_touch)
        configs = {
            p.name: __import__("yaml").safe_load(p.read_text(encoding="utf-8"))
            for p in cfg.sweep_dir.glob("arm_*.yaml")
        }
        return ledger, configs

    def test_reference_exports_arms_import(self, tmp_path, monkeypatch):
        ledger, configs = self._configs_written(tmp_path, monkeypatch, transfer=True)
        reference = next(r for r in ledger.records.values() if not r.knobs)
        ref_config = configs[f"arm_{reference.arm_id}.yaml"]
        assert "warmup_export_path" in ref_config
        assert "warmup_import_path" not in ref_config

        arm_records = [r for r in ledger.records.values() if r.knobs]
        assert arm_records
        for r in arm_records:
            payload = configs[f"arm_{r.arm_id}.yaml"]
            assert payload["warmup_import_path"] == ref_config["warmup_export_path"]
            assert payload["num_warmup"] == 200
            assert r.warm_started is True
        assert reference.warm_started is None

    def test_default_off_writes_no_warmup_keys(self, tmp_path, monkeypatch):
        _, configs = self._configs_written(tmp_path, monkeypatch, transfer=False)
        for payload in configs.values():
            assert "warmup_export_path" not in payload
            assert "warmup_import_path" not in payload
