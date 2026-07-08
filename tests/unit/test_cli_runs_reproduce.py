"""resolved_config.yaml + `panelcast runs reproduce` (#170)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.config.pipeline_yaml import (
    PIPELINE_YAML_MAPPING,
    dump_resolved_config,
    load_resolved_config,
)
from panelcast.pipelines.manifest import (
    EnvironmentInfo,
    GitStateModel,
    RunManifest,
    save_run_manifest,
)
from panelcast.pipelines.orchestrator import PipelineConfig

runner = CliRunner()


class TestResolvedConfigRoundTrip:
    def test_non_default_config_round_trips_identically(self, tmp_path):
        original = PipelineConfig(
            seed=7,
            num_samples=2000,
            num_chains=2,
            target_transform="identity",
            entity_group_pooling="on",
            errors_in_variables=True,
            calibration_intervals=(0.5, 0.8, 0.95),
            min_ratings=25,
        )
        path = tmp_path / "resolved_config.yaml"
        path.write_text(dump_resolved_config(original), encoding="utf-8")
        rebuilt = PipelineConfig(**load_resolved_config(path))
        for spec in PIPELINE_YAML_MAPPING.values():
            assert getattr(rebuilt, spec.config_field) == getattr(
                original, spec.config_field
            ), f"round-trip drift on {spec.config_field}"

    def test_every_mapped_field_exists_on_config(self):
        # A mapping entry pointing at a removed field would silently emit null.
        config = PipelineConfig()
        for yaml_key, spec in PIPELINE_YAML_MAPPING.items():
            assert hasattr(config, spec.config_field), f"{yaml_key} -> {spec.config_field}"


class TestSetupWritesResolvedConfig:
    def test_fresh_run_writes_the_yaml(self, tmp_path, monkeypatch):
        from panelcast.pipelines.orchestrator import PipelineOrchestrator
        from panelcast.utils.git_state import GitState

        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.capture_git_state",
            lambda: GitState(commit="abc", branch="t", dirty=False, untracked_count=0),
        )
        orch = PipelineOrchestrator(
            PipelineConfig(num_samples=1234), output_base=tmp_path / "outputs"
        )
        orch._setup_run()
        path = orch.run_dir / "resolved_config.yaml"
        assert path.exists()
        rebuilt = PipelineConfig(**load_resolved_config(path))
        assert rebuilt.num_samples == 1234


def _write_run(base: Path, run_id: str, with_resolved: bool = True) -> Path:
    run_dir = base / run_id
    (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation" / "metrics.json").write_text(
        json.dumps({"point_metrics": {"mae": 5.3}}), encoding="utf-8"
    )
    manifest = RunManifest(
        run_id=run_id,
        created_at="2026-07-08T00:00:00Z",
        command="panelcast run",
        flags={"num_samples": 1234, "dataset": None},
        seed=42,
        git=GitStateModel(commit="a" * 40, branch="main", dirty=False, untracked_count=0),
        environment=EnvironmentInfo(
            python_version="3.14",
            jax_version="0.8.2",
            numpyro_version=None,
            arviz_version=None,
            platform="Linux",
            pixi_lock_hash=None,
            fingerprint="0000000000000000",  # never matches the live env
        ),
        input_hashes={},
        stage_hashes={},
        stages_completed=["train"],
        stages_skipped=[],
        outputs={},
        success=True,
    )
    save_run_manifest(manifest, run_dir)
    if with_resolved:
        (run_dir / "resolved_config.yaml").write_text(
            dump_resolved_config(PipelineConfig(num_samples=1234)), encoding="utf-8"
        )
    return run_dir


class TestRunsReproduce:
    def _invoke(self, base, monkeypatch, run_id="run_a"):
        launched: dict = {}

        def fake_run_pipeline(config, output_base=Path("outputs")):
            launched["config"] = config
            new_dir = Path(output_base) / "run_new"
            (new_dir / "evaluation").mkdir(parents=True, exist_ok=True)
            (new_dir / "evaluation" / "metrics.json").write_text(
                json.dumps({"point_metrics": {"mae": 5.4}}), encoding="utf-8"
            )
            manifest = RunManifest(
                run_id="run_new",
                created_at="2026-07-08T01:00:00Z",
                command="panelcast run",
                flags={},
                seed=config.seed,
                git=GitStateModel(
                    commit="a" * 40, branch="main", dirty=False, untracked_count=0
                ),
                environment=EnvironmentInfo(
                    python_version="3.14",
                    jax_version="0.8.2",
                    numpyro_version=None,
                    arviz_version=None,
                    platform="Linux",
                    pixi_lock_hash=None,
                ),
                input_hashes={},
                stage_hashes={},
                stages_completed=["train"],
                stages_skipped=[],
                outputs={},
                success=True,
            )
            save_run_manifest(manifest, new_dir)
            (Path(output_base) / "latest.json").write_text(
                json.dumps({"run_dir": "run_new"}), encoding="utf-8"
            )
            return 0

        import panelcast.paths as paths_mod
        import panelcast.pipelines.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(
            paths_mod, "resolve_latest", lambda output_base=Path("outputs"): base / "run_new"
        )
        result = runner.invoke(
            app, ["runs", "reproduce", run_id, "--output-base", str(base)]
        )
        return result, launched

    def test_reproduces_from_resolved_config(self, tmp_path, monkeypatch):
        base = tmp_path / "outputs"
        _write_run(base, "run_a")
        result, launched = self._invoke(base, monkeypatch)
        assert result.exit_code == 0, result.output
        assert "resolved_config.yaml" in result.output
        assert launched["config"].num_samples == 1234
        assert launched["config"].skip_existing is False
        assert "statistical reproduction" in result.output
        assert "mae: 5.3 -> 5.4" in result.output

    def test_falls_back_to_flags_for_old_runs(self, tmp_path, monkeypatch):
        base = tmp_path / "outputs"
        _write_run(base, "run_a", with_resolved=False)
        result, launched = self._invoke(base, monkeypatch)
        assert result.exit_code == 0, result.output
        assert "manifest flags" in result.output
        assert launched["config"].num_samples == 1234

    def test_descriptor_drift_aborts_before_compute(self, tmp_path, monkeypatch):
        base = tmp_path / "outputs"
        run_dir = _write_run(base, "run_a")
        manifest_path = run_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["flags"]["dataset_descriptor_hash"] = "not-the-current-hash"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        result, launched = self._invoke(base, monkeypatch)
        assert result.exit_code == 1
        assert "descriptor changed" in result.output
        assert "config" not in launched  # never launched

    def test_changed_input_aborts(self, tmp_path, monkeypatch):
        base = tmp_path / "outputs"
        run_dir = _write_run(base, "run_a")
        raw = tmp_path / "raw.csv"
        raw.write_text("x\n", encoding="utf-8")
        manifest_path = run_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["input_hashes"] = {str(raw): "0" * 64}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        result, launched = self._invoke(base, monkeypatch)
        assert result.exit_code == 1
        assert "raw input changed" in result.output
