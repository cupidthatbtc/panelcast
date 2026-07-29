"""Run/resume identifiers stay inside the output root (#365).

Every lookup, move, and delete keyed by a run identifier goes through
``panelcast.paths.safe_run_dir``; these are the escapes it has to refuse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from panelcast.config.pipeline_yaml import dump_resolved_config
from panelcast.paths import (
    RunPathError,
    path_is_within,
    resolve_latest,
    safe_run_dir,
    validate_run_id,
)
from panelcast.pipelines.manifest import (
    EnvironmentInfo,
    GitStateModel,
    RunManifest,
    generate_run_id,
    save_run_manifest,
)

ESCAPING_IDS = [
    "..",
    "../evil",
    "../../etc",
    "..\\evil",
    "a/b",
    "a\\b",
    "/etc/passwd",
    "\\\\server\\share",
    "C:/Windows",
    "C:evil",
    "sub/../../out",
    "./here",
]

MALFORMED_IDS = [
    "",
    " ",
    ".",
    ".hidden",
    "trailing.",
    "trailing ",
    " leading",
    "with\x00null",
    "with\nnewline",
    "stream:name",
    'bad<name',
    'bad>name',
    'bad"name',
    "bad|name",
    "bad?name",
    "bad*name",
    "x" * 256,
    "é" * 128,
]

RESERVED_IDS = [
    "latest",
    "latest.json",
    "latest.json.tmp",
    "failed",
    "failed.log",
    "LATEST",
    "Failed",
    "nul",
    "CON",
    "com1",
    "LPT9",
    "con.txt",
    "CONIN$",
    "clock$",
    "COM¹",
    "lpt³.log",
]


def _symlink_dir(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # unprivileged Windows
        pytest.skip(f"directory symlinks unavailable: {exc}")


def _write_resumable_run(run_dir: Path, config) -> Path:
    """A run directory `_setup_resume` will accept."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_manifest(
        RunManifest(
            run_id=run_dir.name,
            created_at="2026-07-26T00:00:00Z",
            command="panelcast run",
            flags={"dataset": None},
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
        ),
        run_dir,
    )
    (run_dir / "resolved_config.yaml").write_text(
        dump_resolved_config(config), encoding="utf-8"
    )
    return run_dir


class TestValidateRunId:
    @pytest.mark.parametrize("run_id", ESCAPING_IDS + MALFORMED_IDS + RESERVED_IDS)
    def test_rejected(self, run_id):
        with pytest.raises(RunPathError):
            validate_run_id(run_id)

    @pytest.mark.parametrize(
        "run_id",
        [
            "2026-01-19_143052_123456_a3f9",
            "sel_hs_abc123_20260709T120000",
            "runA",
            "run.with.dots",
            "nullify",
            "console",
            "com10",
        ],
    )
    def test_accepted(self, run_id):
        assert validate_run_id(run_id) == run_id

    def test_generated_ids_are_accepted(self):
        for _ in range(20):
            validate_run_id(generate_run_id())

    def test_message_names_the_field(self):
        with pytest.raises(RunPathError, match="resume"):
            validate_run_id("../x", field="resume")


class TestSafeRunDir:
    def test_returns_unresolved_join(self, tmp_path):
        assert safe_run_dir(tmp_path, "runA") == tmp_path / "runA"

    def test_subdir_join(self, tmp_path):
        assert safe_run_dir(tmp_path, "runA", subdir="failed") == tmp_path / "failed" / "runA"

    @pytest.mark.parametrize("run_id", ESCAPING_IDS)
    def test_escaping_ids_rejected(self, tmp_path, run_id):
        with pytest.raises(RunPathError):
            safe_run_dir(tmp_path, run_id)

    def test_nonexistent_id_is_fine(self, tmp_path):
        # Containment is a property of the path, not of what exists on disk.
        assert safe_run_dir(tmp_path, "never-created").name == "never-created"

    def test_symlinked_run_name_escaping_root_rejected(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        base = tmp_path / "outputs"
        base.mkdir()
        _symlink_dir(base / "escape", outside)
        with pytest.raises(RunPathError):
            safe_run_dir(base, "escape")

    def test_symlink_inside_root_allowed(self, tmp_path):
        base = tmp_path / "outputs"
        (base / "real").mkdir(parents=True)
        _symlink_dir(base / "alias", base / "real")
        assert safe_run_dir(base, "alias") == base / "alias"

    def test_symlinked_failed_slot_escaping_root_rejected(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        base = tmp_path / "outputs"
        (base / "failed").mkdir(parents=True)
        _symlink_dir(base / "failed" / "escape", outside)
        with pytest.raises(RunPathError):
            safe_run_dir(base, "escape", subdir="failed")

    def test_symlinked_output_base_is_not_an_escape(self, tmp_path):
        real = tmp_path / "real_outputs"
        (real / "runA").mkdir(parents=True)
        link = tmp_path / "outputs"
        _symlink_dir(link, real)
        assert safe_run_dir(link, "runA") == link / "runA"


class TestPathIsWithin:
    def test_strict_descendant_only(self, tmp_path):
        assert path_is_within(tmp_path / "a", tmp_path)
        assert path_is_within(tmp_path / "a" / "b", tmp_path)
        assert not path_is_within(tmp_path, tmp_path)
        assert not path_is_within(tmp_path.parent, tmp_path)

    def test_resolution_error_fails_closed(self, tmp_path, monkeypatch):
        def fail(_path):
            raise OSError("symlink loop")

        monkeypatch.setattr(Path, "resolve", fail)
        assert not path_is_within(tmp_path / "a", tmp_path)


class TestMaliciousLatestPointer:
    def _base(self, tmp_path):
        base = tmp_path / "outputs"
        base.mkdir()
        return base

    @pytest.mark.parametrize("target", ["..", "../secrets", "/etc", "a/b", "..\\secrets"])
    def test_pointer_outside_root_is_ignored(self, tmp_path, target):
        base = self._base(tmp_path)
        (tmp_path / "secrets").mkdir()
        (base / "latest.json").write_text(
            json.dumps({"run_id": "x", "run_dir": target}), encoding="utf-8"
        )
        assert resolve_latest(base) is None

    def test_pointer_naming_a_symlinked_escape_is_ignored(self, tmp_path):
        base = self._base(tmp_path)
        outside = tmp_path / "outside"
        outside.mkdir()
        _symlink_dir(base / "escape", outside)
        (base / "latest.json").write_text(
            json.dumps({"run_id": "escape", "run_dir": "escape"}), encoding="utf-8"
        )
        assert resolve_latest(base) is None

    @pytest.mark.parametrize(
        "payload",
        [
            {"run_id": "runA", "run_dir": "runB"},
            {"run_id": 1, "run_dir": "1"},
            {"run_id": "1", "run_dir": 1},
        ],
    )
    def test_pointer_identity_must_be_one_bare_string(self, tmp_path, payload):
        base = self._base(tmp_path)
        (base / "runA").mkdir()
        (base / "runB").mkdir()
        (base / "1").mkdir()
        (base / "latest.json").write_text(json.dumps(payload), encoding="utf-8")
        assert resolve_latest(base) is None

    def test_valid_pointer_still_resolves(self, tmp_path):
        base = self._base(tmp_path)
        (base / "runA").mkdir()
        (base / "latest.json").write_text(
            json.dumps({"run_id": "runA", "run_dir": "runA"}), encoding="utf-8"
        )
        assert resolve_latest(base) == base / "runA"

    def test_latest_link_pointing_outside_is_ignored(self, tmp_path):
        base = self._base(tmp_path)
        outside = tmp_path / "outside"
        outside.mkdir()
        _symlink_dir(base / "latest", outside)
        assert resolve_latest(base) is None

    def test_latest_link_pointing_inside_still_resolves(self, tmp_path):
        base = self._base(tmp_path)
        (base / "runA").mkdir()
        _symlink_dir(base / "latest", base / "runA")
        assert resolve_latest(base) == base / "latest"


class TestConfigRejectsEscapingIds:
    @pytest.mark.parametrize("run_id", ESCAPING_IDS + RESERVED_IDS)
    def test_run_id(self, run_id):
        from panelcast.pipelines.orchestrator import PipelineConfig

        with pytest.raises(ValueError, match="run_id"):
            PipelineConfig(run_id=run_id)

    @pytest.mark.parametrize("resume", ESCAPING_IDS + RESERVED_IDS)
    def test_resume(self, resume):
        from panelcast.pipelines.orchestrator import PipelineConfig

        with pytest.raises(ValueError, match="resume"):
            PipelineConfig(resume=resume)

    def test_run_id_from_yaml_is_rejected(self):
        from panelcast.config.pipeline_yaml import apply_yaml_overrides
        from panelcast.pipelines.orchestrator import PipelineConfig

        kwargs = apply_yaml_overrides({}, {"run_id": "../../elsewhere"})
        with pytest.raises(ValueError, match="run_id"):
            PipelineConfig(**kwargs)


class TestResumeContainment:
    def _orchestrator(self, base, resume_id):
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        orch = PipelineOrchestrator(PipelineConfig(), output_base=base)
        # Assigned after validation on purpose: the resolver, not config
        # validation, has to be the load-bearing guard at the disk boundary.
        orch.config.resume = resume_id
        return orch

    def _write_run(self, run_dir):
        from panelcast.pipelines.orchestrator import PipelineConfig

        _write_resumable_run(run_dir, PipelineConfig())

    def test_traversal_resume_raises_pipeline_error(self, tmp_path):
        from panelcast.pipelines.orchestrator import PipelineError

        base = tmp_path / "outputs"
        base.mkdir()
        orch = self._orchestrator(base, "../elsewhere")
        with pytest.raises(PipelineError):
            orch._setup_resume()

    def test_symlinked_resume_target_raises_pipeline_error(self, tmp_path):
        from panelcast.pipelines.orchestrator import PipelineError

        base = tmp_path / "outputs"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "manifest.json").write_text("{}", encoding="utf-8")
        _symlink_dir(base / "escape", outside)
        orch = self._orchestrator(base, "escape")
        with pytest.raises(PipelineError):
            orch._setup_resume()

    def test_active_run_resumes(self, tmp_path):
        base = tmp_path / "outputs"
        self._write_run(base / "run_a")
        orch = self._orchestrator(base, "run_a")
        orch._setup_resume()
        assert orch.run_dir == base / "run_a"

    def test_failed_run_is_moved_back(self, tmp_path):
        base = tmp_path / "outputs"
        self._write_run(base / "failed" / "run_a")
        orch = self._orchestrator(base, "run_a")
        orch._setup_resume()
        assert orch.run_dir == base / "run_a"
        assert (base / "run_a" / "manifest.json").exists()
        assert not (base / "failed" / "run_a").exists()

    def test_symlinked_failed_slot_is_not_moved(self, tmp_path):
        from panelcast.pipelines.orchestrator import PipelineError

        base = tmp_path / "outputs"
        (base / "failed").mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep", encoding="utf-8")
        _symlink_dir(base / "failed" / "run_a", outside)
        orch = self._orchestrator(base, "run_a")
        with pytest.raises(PipelineError):
            orch._setup_resume()
        assert (outside / "keep.txt").exists()
        assert not (base / "run_a").exists()


class TestFailureQuarantineContainment:
    def test_symlinked_quarantine_slot_is_never_deleted(self, tmp_path):
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        base = tmp_path / "outputs"
        run_dir = base / "run_a"
        run_dir.mkdir(parents=True)
        (run_dir / "artifact.txt").write_text("x", encoding="utf-8")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep", encoding="utf-8")
        (base / "failed").mkdir()
        _symlink_dir(base / "failed" / "run_a", outside)

        orch = PipelineOrchestrator(PipelineConfig(), output_base=base)
        orch.run_dir = run_dir
        orch._handle_failure(RuntimeError("boom"), "train")

        assert (outside / "keep.txt").exists()
        assert (run_dir / "artifact.txt").exists()

    def test_quarantine_cleanup_error_does_not_escape(self, tmp_path, monkeypatch):
        from panelcast.pipelines import orchestrator as orchestrator_module
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        base = tmp_path / "outputs"
        run_dir = base / "run_a"
        run_dir.mkdir(parents=True)
        failed = base / "failed" / "run_a"
        failed.mkdir(parents=True)

        def refuse_cleanup(path):
            raise OSError("locked")

        monkeypatch.setattr(orchestrator_module.shutil, "rmtree", refuse_cleanup)
        orch = PipelineOrchestrator(PipelineConfig(), output_base=base)
        orch.run_dir = run_dir
        orch._handle_failure(RuntimeError("original"), "train")

        assert run_dir.exists()
        assert failed.exists()

    def test_normal_quarantine_still_moves(self, tmp_path):
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        base = tmp_path / "outputs"
        run_dir = base / "run_a"
        run_dir.mkdir(parents=True)
        (run_dir / "artifact.txt").write_text("x", encoding="utf-8")

        orch = PipelineOrchestrator(PipelineConfig(), output_base=base)
        orch.run_dir = run_dir
        orch._handle_failure(RuntimeError("boom"), "train")

        assert (base / "failed" / "run_a" / "artifact.txt").exists()
        assert not run_dir.exists()


class TestRunsCliContainment:
    @pytest.mark.parametrize("run_id", ESCAPING_IDS)
    def test_resolve_run_dir_rejects_escapes(self, tmp_path, run_id):
        from panelcast.cli.runs_cmd import resolve_run_dir

        (tmp_path / "outputs").mkdir()
        with pytest.raises(typer.BadParameter):
            resolve_run_dir(run_id, tmp_path / "outputs")

    def test_resolve_run_dir_finds_active_and_failed(self, tmp_path):
        from panelcast.cli.runs_cmd import resolve_run_dir

        base = tmp_path / "outputs"
        (base / "run_a").mkdir(parents=True)
        (base / "run_a" / "manifest.json").write_text("{}", encoding="utf-8")
        (base / "failed" / "run_b").mkdir(parents=True)
        (base / "failed" / "run_b" / "manifest.json").write_text("{}", encoding="utf-8")
        assert resolve_run_dir("run_a", base) == base / "run_a"
        assert resolve_run_dir("run_b", base) == base / "failed" / "run_b"

    def test_symlinked_run_is_not_resolved(self, tmp_path):
        from panelcast.cli.runs_cmd import resolve_run_dir

        base = tmp_path / "outputs"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "manifest.json").write_text("{}", encoding="utf-8")
        _symlink_dir(base / "escape", outside)
        with pytest.raises(typer.BadParameter):
            resolve_run_dir("escape", base)


class TestSweepAndBacktestIds:
    def test_sweep_dir_rejects_traversal(self, tmp_path):
        from panelcast.select.runner import SweepConfig

        with pytest.raises(RunPathError):
            SweepConfig(sweep_id="../escape", output_root=tmp_path)

    def test_sweep_dir_resolves_bare_id_once(self, tmp_path):
        from dataclasses import asdict

        from panelcast.select.runner import SweepConfig

        cfg = SweepConfig(sweep_id="s1", output_root=tmp_path)
        assert cfg.sweep_dir == tmp_path / "s1"
        assert cfg.sweep_dir is cfg.sweep_dir
        assert "_sweep_dir" not in asdict(cfg)

    def test_backtest_dir_rejects_traversal(self, tmp_path):
        from panelcast.pipelines.backtest import BacktestConfig

        with pytest.raises(RunPathError):
            BacktestConfig(backtest_id="../escape", output_root=tmp_path)

    def test_backtest_dir_resolves_bare_id_once(self, tmp_path):
        from dataclasses import asdict

        from panelcast.pipelines.backtest import BacktestConfig

        cfg = BacktestConfig(backtest_id="nightly", output_root=tmp_path)
        assert cfg.backtest_dir == tmp_path / "nightly"
        assert cfg.backtest_dir is cfg.backtest_dir
        assert "_backtest_dir" not in asdict(cfg)


class TestWindowsPathShapes:
    """Windows-only escapes are rejected everywhere, so ids stay portable."""

    @pytest.mark.parametrize(
        "bad", ["C:evil", "C:/Windows", "\\\\server\\share", "..\\up", "sub\\run"]
    )
    def test_drive_relative_unc_and_backslash_rejected(self, tmp_path, bad):
        with pytest.raises(RunPathError):
            safe_run_dir(tmp_path, bad)

    @pytest.mark.parametrize("bad", ["runA.", "runA ", "runA..", "run:stream"])
    def test_ids_windows_would_silently_alias(self, tmp_path, bad):
        # Windows strips trailing dots/spaces and treats ':' as a stream, so
        # these would name an existing run under a different-looking id.
        (tmp_path / "runA").mkdir()
        with pytest.raises(RunPathError):
            safe_run_dir(tmp_path, bad)
