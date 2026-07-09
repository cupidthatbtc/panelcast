"""#167 parallel arms: admission control, bucketing, OOM fallback, default parity."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import yaml

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.gpu_memory.admission import GpuAdmission
from panelcast.select.runner import SweepConfig, ofat_arms, run_sweep

AOTY = DatasetDescriptor()
_GIB = 1024**3


class TestGpuAdmission:
    def test_never_over_commits(self):
        adm = GpuAdmission(headroom=0.5, free_bytes_fn=lambda: 20 * _GIB)
        assert adm.try_admit(4.0)
        assert adm.try_admit(4.0)
        assert not adm.try_admit(4.0)  # 12 > 20 * 0.5
        adm.release(4.0)
        assert adm.try_admit(4.0)

    def test_thread_safe_reservations(self):
        adm = GpuAdmission(headroom=1.0, free_bytes_fn=lambda: 10 * _GIB)
        granted = []

        def worker():
            granted.append(adm.try_admit(3.0))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(granted) == 3  # 3 * 3.0 <= 10, a 4th would over-commit

    def test_unknown_gpu_serializes(self):
        adm = GpuAdmission(free_bytes_fn=lambda: None)
        assert adm.try_admit(2.0)
        assert not adm.try_admit(2.0)
        adm.release(2.0)
        assert adm.try_admit(2.0)

    def test_single_oversized_arm_admits_alone(self):
        # An arm priced above the whole headroom budget still runs (alone);
        # admit() must never spin forever on it.
        adm = GpuAdmission(headroom=0.8, free_bytes_fn=lambda: 12 * _GIB)
        assert adm.try_admit(20.0)
        assert not adm.try_admit(0.5)
        adm.release(20.0)
        assert adm.try_admit(0.5)


def _env(tmp_path, parallel_arms=2, fail_first_oom=None):
    """Fake launcher tracking concurrency; optional one-shot OOM per arm id."""
    state = {"live": 0, "max_live": 0, "oomed": set()}
    lock = threading.Lock()

    def launch(config_path, panelcast_bin, timeout_seconds=None, env_overrides=None):
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        with lock:
            state["live"] += 1
            state["max_live"] = max(state["max_live"], state["live"])
        try:
            if fail_first_oom and fail_first_oom(payload) and "hit" not in state["oomed"]:
                state["oomed"].add("hit")  # only the FIRST attempt OOMs
                return 1, "XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory"
            run_dir = tmp_path / "outputs" / payload["run_id"]
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps({"created_at": datetime.now().isoformat()}), encoding="utf-8"
            )
            return 0, "ok"
        finally:
            with lock:
                state["live"] -= 1

    cfg = SweepConfig(
        sweep_id="p",
        output_root=tmp_path / "select",
        panelcast_bin="pc",
        include_stage2=False,
        pipeline_output_base=tmp_path / "outputs",
        parallel_arms=parallel_arms,
    )
    return cfg, launch, state


class TestParallelBuckets:
    def test_concurrency_bounded_and_all_complete(self, tmp_path):
        cfg, launch, state = _env(tmp_path, parallel_arms=3)
        ledger = run_sweep(cfg, AOTY, launch=launch)
        assert all(r.status == "completed" for r in ledger.records.values())
        assert len(ledger.records) == 1 + len(ofat_arms(AOTY))
        assert state["max_live"] <= 3

    def test_oom_arm_is_serialized_and_recovers(self, tmp_path):
        def oom_on_ar1(payload):
            return payload.get("latent_process") == "ar1"

        cfg, launch, state = _env(tmp_path, parallel_arms=2, fail_first_oom=oom_on_ar1)
        ledger = run_sweep(cfg, AOTY, launch=launch)
        ar1 = next(r for r in ledger.records.values() if r.knobs == {"latent_process": "ar1"})
        assert ar1.status == "completed"
        assert "serialized after OOM" in (ar1.note or "")

    def test_failed_head_falls_back_to_serial(self, tmp_path, monkeypatch):
        """A non-OOM head failure leaves the flat cache untrusted — the tail
        must serialize rather than race N feature rebuilds into it."""
        import panelcast.select.runner as runner_mod

        monkeypatch.setattr(
            runner_mod, "ofat_arms",
            lambda d, available_columns=None: [
                ({"ar_center": "none"}, None),
                ({"latent_process": "ar1"}, None),
                ({"debut_prev_score_source": "dataset_stats"}, None),
            ],
        )
        cfg, launch, state = _env(tmp_path, parallel_arms=2)
        calls = {"n": 0}

        def failing_launch(config_path, panelcast_bin, timeout_seconds=None, env_overrides=None):
            calls["n"] += 1
            if calls["n"] == 2:  # the bucket head, right after the reference
                return 1, "RuntimeError: boom"
            return launch(config_path, panelcast_bin, timeout_seconds, env_overrides)

        ledger = run_sweep(cfg, AOTY, launch=failing_launch)
        assert state["max_live"] == 1
        assert len([r for r in ledger.records.values() if r.status == "failed"]) == 1
        assert len([r for r in ledger.records.values() if r.knobs and r.status == "completed"]) == 2

    def test_default_serial_uses_three_arg_launch(self, tmp_path):
        """--parallel-arms 1 keeps the legacy launch protocol byte-identical:
        a fake WITHOUT env_overrides support must still work."""
        calls = []

        def legacy_launch(config_path, panelcast_bin, timeout_seconds=None):
            calls.append(config_path)
            payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            run_dir = tmp_path / "outputs" / payload["run_id"]
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps({"created_at": datetime.now().isoformat()}), encoding="utf-8"
            )
            return 0, "ok"

        cfg = SweepConfig(
            sweep_id="s1",
            output_root=tmp_path / "select",
            panelcast_bin="pc",
            include_stage2=False,
            max_fits=3,
            pipeline_output_base=tmp_path / "outputs",
        )
        ledger = run_sweep(cfg, AOTY, launch=legacy_launch)
        assert len(calls) == 3
        assert all(r.status == "completed" for r in ledger.records.values())
