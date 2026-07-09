"""#167 prerequisites: run-dir handshake and per-child env overrides."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from panelcast.pipelines.orchestrator import PipelineConfig
from panelcast.select.runner import SweepConfig, launch_arm, run_sweep

from panelcast.config.descriptor import DatasetDescriptor

AOTY = DatasetDescriptor()


def _fake_env(tmp_path):
    launches: list[Path] = []

    def launch(config_path, panelcast_bin, timeout_seconds=None):
        launches.append(Path(config_path))
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        run_dir = tmp_path / "outputs" / payload["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"created_at": datetime.now().isoformat()}), encoding="utf-8"
        )
        return 0, "ok"

    cfg = SweepConfig(
        sweep_id="hs",
        output_root=tmp_path / "select",
        panelcast_bin="pc",
        include_stage2=False,
        pipeline_output_base=tmp_path / "outputs",
    )
    return cfg, launches, launch


class TestRunIdHandshake:
    def test_every_arm_config_names_a_unique_run(self, tmp_path):
        cfg, launches, launch = _fake_env(tmp_path)
        cfg.max_fits = 4
        ledger = run_sweep(cfg, AOTY, launch=launch)
        run_ids = [
            yaml.safe_load(p.read_text(encoding="utf-8"))["run_id"] for p in launches
        ]
        assert len(run_ids) == len(set(run_ids))
        assert all(rid.startswith("sel_hs_") for rid in run_ids)
        # And the ledger records point at exactly those named dirs.
        recorded = {Path(r.run_dir).name for r in ledger.records.values() if r.run_dir}
        assert recorded == set(run_ids)

    def test_no_resolve_latest_in_the_sweep_path(self, tmp_path, monkeypatch):
        """The acceptance criterion: a sweep never touches resolve_latest,
        even when the pointer is poisoned mid-sweep."""
        import panelcast.paths as paths_mod

        def _boom(*a, **kw):
            raise AssertionError("resolve_latest called from the sweep path")

        monkeypatch.setattr(paths_mod, "resolve_latest", _boom)
        cfg, launches, launch = _fake_env(tmp_path)
        cfg.max_fits = 2
        ledger = run_sweep(cfg, AOTY, launch=launch)
        assert all(r.status == "completed" for r in ledger.records.values())


class TestLaunchArmEnv:
    def test_env_overrides_reach_the_child(self, tmp_path, monkeypatch):
        import panelcast.select.runner as runner_mod

        seen = {}

        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, capture_output, text, env, timeout):
            seen.update(env)
            return _Proc()

        monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)
        config = tmp_path / "arm.yaml"
        config.write_text("{}", encoding="utf-8")
        launch_arm(
            config, "pc", env_overrides={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30"}
        )
        assert seen["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.30"
        assert seen["PANELCAST_SAVE_LOG_LIKELIHOOD"] == "1"

    def test_default_env_unchanged(self, tmp_path, monkeypatch):
        import panelcast.select.runner as runner_mod

        seen = {}

        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, capture_output, text, env, timeout):
            seen.update(env)
            return _Proc()

        monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)
        config = tmp_path / "arm.yaml"
        config.write_text("{}", encoding="utf-8")
        launch_arm(config, "pc")
        assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in seen


class TestRunIdConfigKnob:
    def test_bare_name_accepted(self):
        cfg = PipelineConfig(run_id="sel_hs_abc123_20260709T120000")
        assert cfg.run_id == "sel_hs_abc123_20260709T120000"

    @pytest.mark.parametrize("bad", ["a/b", "..", "", "a\\b"])
    def test_path_like_run_id_rejected(self, bad):
        with pytest.raises(ValueError, match="run_id"):
            PipelineConfig(run_id=bad)
