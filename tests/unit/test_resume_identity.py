"""Canonical experiment identity for resume (#296)."""

from __future__ import annotations

import json
from dataclasses import fields as dataclass_fields

import pytest
import yaml

from panelcast.config.pipeline_yaml import (
    EXPERIMENT_EXCLUDED_KEYS,
    PIPELINE_YAML_MAPPING,
    dump_resolved_config,
    experiment_config_hash,
    experiment_config_payload,
)
from panelcast.pipelines.errors import PipelineError
from panelcast.pipelines.manifest import (
    EnvironmentInfo,
    GitStateModel,
    RunManifest,
    save_run_manifest,
)
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator
from panelcast.utils.git_state import GitState


class TestExperimentConfigHash:
    def test_stable_for_equal_configs(self):
        assert experiment_config_hash(PipelineConfig()) == experiment_config_hash(
            PipelineConfig()
        )

    def test_changes_with_any_output_affecting_knob(self):
        base = experiment_config_hash(PipelineConfig())
        assert experiment_config_hash(PipelineConfig(num_samples=2000)) != base
        assert experiment_config_hash(PipelineConfig(enable_genre=False)) != base
        assert experiment_config_hash(PipelineConfig(stages=["train"])) != base

    def test_execution_mechanics_do_not_change_the_hash(self):
        base = experiment_config_hash(PipelineConfig())
        assert experiment_config_hash(PipelineConfig(dry_run=True)) == base
        assert experiment_config_hash(PipelineConfig(verbose=True)) == base
        assert experiment_config_hash(PipelineConfig(skip_existing=True)) == base
        assert experiment_config_hash(PipelineConfig(run_id="named-run")) == base
        assert experiment_config_hash(PipelineConfig(strict=True)) == base

    def test_excluded_keys_are_all_mapped(self):
        assert EXPERIMENT_EXCLUDED_KEYS <= set(PIPELINE_YAML_MAPPING)

    def test_payload_covers_every_non_excluded_mapped_key(self):
        config = PipelineConfig(stages=["train"], checkpoint_every_draws=100)
        payload = experiment_config_payload(config)
        for yaml_key in PIPELINE_YAML_MAPPING:
            if yaml_key in EXPERIMENT_EXCLUDED_KEYS:
                assert yaml_key not in payload
        assert json.dumps(payload, sort_keys=True)  # JSON-able


class TestManifestFlagsCompleteness:
    def test_every_config_field_reaches_the_manifest_flags(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.capture_git_state",
            lambda: GitState(commit="abc", branch="t", dirty=False, untracked_count=0),
        )
        orch = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path / "outputs")
        orch._setup_run()
        for f in dataclass_fields(PipelineConfig):
            assert f.name in orch.manifest.flags, f.name
        assert "dataset_descriptor_hash" in orch.manifest.flags

    def test_experiment_identity_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.capture_git_state",
            lambda: GitState(commit="abc", branch="t", dirty=True, untracked_count=1),
        )
        config = PipelineConfig(num_samples=1234)
        orch = PipelineOrchestrator(config, output_base=tmp_path / "outputs")
        orch._setup_run()
        identity = orch.manifest.experiment_identity
        assert identity["config_hash"] == experiment_config_hash(config)
        assert identity["descriptor_hash"] == orch.descriptor.descriptor_hash()
        assert identity["source"] == {"commit": "abc", "dirty": True}
        assert identity["package_version"]


class TestResumeKeyDerivation:
    def test_previously_omitted_controls_are_now_restored(self):
        for key in ("enable_genre", "enable_artist", "enable_temporal", "stages",
                    "warmup_import_path", "warmup_export_path"):
            assert key in PipelineOrchestrator.RESUME_CONFIG_KEYS, key

    def test_execution_mechanics_stay_excluded(self):
        for key in ("resume", "skip_existing", "dry_run", "verbose", "progress_bar",
                    "tag", "run_id", "strict"):
            assert key not in PipelineOrchestrator.RESUME_CONFIG_KEYS, key

    def test_every_field_is_restored_or_explicitly_excluded(self):
        covered = set(PipelineOrchestrator.RESUME_CONFIG_KEYS) | set(
            PipelineOrchestrator.RESUME_EXCLUDED_KEYS
        )
        assert {f.name for f in dataclass_fields(PipelineConfig)} <= covered


def _write_resumable_run(base, run_id, config, identity=None, flags=None):
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(
        run_id=run_id,
        created_at="2026-07-23T00:00:00Z",
        command="panelcast run",
        flags=flags if flags is not None else {"dataset": None},
        seed=config.seed,
        git=GitStateModel(commit="a" * 40, branch="main", dirty=False, untracked_count=0),
        environment=EnvironmentInfo(
            python_version="3.14",
            jax_version="0.8.2",
            numpyro_version=None,
            arviz_version=None,
            platform="Linux",
            pixi_lock_hash=None,
            fingerprint="0000000000000000",
        ),
        input_hashes={},
        stage_hashes={},
        stages_completed=["data"],
        stages_skipped=[],
        outputs={},
        success=False,
        experiment_identity=identity or {},
    )
    save_run_manifest(manifest, run_dir)
    (run_dir / "resolved_config.yaml").write_text(
        dump_resolved_config(config), encoding="utf-8"
    )
    return run_dir


class TestResumeRestoresCompleteExperiment:
    def _resume(self, tmp_path, original, identity=None, tamper=None):
        base = tmp_path / "outputs"
        run_dir = _write_resumable_run(base, "run_a", original, identity=identity)
        if tamper:
            tamper(run_dir)
        config = PipelineConfig(resume="run_a")
        orch = PipelineOrchestrator(config, output_base=base)
        orch._setup_resume()
        return config

    def test_resume_restores_feature_family_and_stage_selection(self, tmp_path):
        original = PipelineConfig(
            enable_genre=False,
            enable_temporal=False,
            stages=["data", "splits", "train"],
            num_samples=1234,
            calibration_intervals=(0.5, 0.9),
        )
        config = self._resume(tmp_path, original)
        assert config.enable_genre is False
        assert config.enable_temporal is False
        assert config.stages == ["data", "splits", "train"]
        assert config.num_samples == 1234
        assert tuple(config.calibration_intervals) == (0.5, 0.9)

    def test_resume_refuses_on_identity_mismatch_with_actionable_diff(self, tmp_path):
        # Recorded identities hold post-resolution values: resolve exactly the
        # way a real run does instead of hand-pinning each sentinel.
        original = PipelineConfig(num_samples=1234)
        PipelineOrchestrator(original, output_base=tmp_path / "resolve_a")
        identity = {
            "config_hash": experiment_config_hash(original),
            "config_payload": experiment_config_payload(original),
        }

        def tamper(run_dir):
            path = run_dir / "resolved_config.yaml"
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["num_samples"] = 2000
            path.write_text(yaml.safe_dump(data), encoding="utf-8")

        with pytest.raises(PipelineError, match=r"(?s)Resume identity mismatch.*num_samples"):
            self._resume(tmp_path, original, identity=identity, tamper=tamper)

    def test_resume_proceeds_when_identity_matches(self, tmp_path):
        original = PipelineConfig(num_samples=1234, enable_artist=False)
        PipelineOrchestrator(original, output_base=tmp_path / "resolve_b")
        identity = {
            "config_hash": experiment_config_hash(original),
            "config_payload": experiment_config_payload(original),
        }
        config = self._resume(tmp_path, original, identity=identity)
        assert config.num_samples == 1234
        assert config.enable_artist is False

    def test_legacy_manifest_without_identity_warns_and_proceeds(self, tmp_path):
        original = PipelineConfig(num_samples=1234)
        config = self._resume(tmp_path, original, identity=None)
        assert config.num_samples == 1234
