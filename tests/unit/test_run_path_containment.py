"""Run/resume identifiers stay inside the output root (#365).

Every lookup, move, and delete keyed by a run identifier goes through
``panelcast.paths.safe_run_dir``; these are the escapes it has to refuse.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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

    @pytest.mark.parametrize("error", [OSError, RuntimeError])
    def test_resolution_error_fails_closed(self, tmp_path, monkeypatch, error):
        # RuntimeError too: Python 3.11 and 3.12 convert a resolve() ELOOP into
        # one, and a planted symlink loop must not raise through every caller
        # that documents a refusal instead.
        def fail(_path):
            raise error("symlink loop")

        monkeypatch.setattr(Path, "resolve", fail)
        assert not path_is_within(tmp_path / "a", tmp_path)

    def test_a_real_symlink_loop_never_raises(self, tmp_path):
        # Version-dependent verdict, single invariant: 3.11/3.12 raise
        # RuntimeError and refuse, 3.13+ resolve the loop to itself and contain
        # it. Either way nothing escapes to the caller.
        loop = tmp_path / "loop"
        try:
            loop.symlink_to(loop, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"directory symlinks unavailable: {exc}")

        assert path_is_within(loop, tmp_path) in (True, False)


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

    def test_active_run_resumes_even_if_failed_root_escapes(self, tmp_path):
        base = tmp_path / "outputs"
        self._write_run(base / "run_a")
        outside = tmp_path / "outside"
        outside.mkdir()
        _symlink_dir(base / "failed", outside)

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

    def test_active_lookup_ignores_an_escaping_failed_root(self, tmp_path):
        from panelcast.cli.runs_cmd import resolve_run_dir

        base = tmp_path / "outputs"
        (base / "run_a").mkdir(parents=True)
        (base / "run_a" / "manifest.json").write_text("{}", encoding="utf-8")
        outside = tmp_path / "outside"
        outside.mkdir()
        _symlink_dir(base / "failed", outside)

        assert resolve_run_dir("run_a", base) == base / "run_a"

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


def _confirmation_cfg(tmp_path: Path, base: Path, sweep_id: str = "s1") -> Any:
    """A sweep whose confirmation fits write under ``base``."""
    from panelcast.select.runner import SweepConfig

    return SweepConfig(
        sweep_id=sweep_id,
        output_root=tmp_path / "select",
        panelcast_bin="pc",
        pipeline_output_base=base,
    )


def _minted_run_id(config_path: Path | str) -> str:
    """The run id a select-written arm config carries."""
    import yaml

    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    run_id = payload.get("run_id")
    assert isinstance(run_id, str) and run_id, f"no top-level run_id in {config_path}"
    return run_id


def _write_arm_manifest(
    run_dir: Path, launched_at: datetime, flags: dict[str, object] | None = None
) -> None:
    """A manifest `_attribution_error` accepts for an arm launched at ``launched_at``.

    ``created_at`` sits a couple of seconds after the launch — the shape
    production writes (a naive local timestamp from the subprocess, which
    starts after the arm does), not the equality boundary. Callers pass
    ``datetime.now()``, so the stamp lands slightly ahead of wall clock;
    `_attribution_error` has no not-in-the-future clause to trip on.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    created_at = launched_at + timedelta(seconds=2)
    (run_dir / "manifest.json").write_text(
        json.dumps({"created_at": created_at.isoformat(), "flags": flags or {}}),
        encoding="utf-8",
    )


class TestSelectRunLookupContainment:
    """Select's own run lookups fail closed, whatever the caller validated first (#375).

    The ids here are internally minted, so these are not live escapes — the
    point is that containment holds on the lookup itself rather than on the
    sweep having been set up in the usual order.
    """

    @pytest.mark.parametrize("run_id", ESCAPING_IDS + MALFORMED_IDS + RESERVED_IDS)
    def test_sweep_run_dir_rejects_bad_ids(self, tmp_path, run_id):
        from panelcast.select.runner import sweep_run_dir

        with pytest.raises(RunPathError):
            sweep_run_dir(tmp_path, run_id)

    @pytest.mark.parametrize("run_id", ESCAPING_IDS + MALFORMED_IDS + RESERVED_IDS)
    def test_both_gates_refuse_the_same_known_ids(self, tmp_path, run_id):
        # `refusal_detail` decides "this run never wrote under that name" by
        # re-running validate_run_id, which is exact only while that stays the
        # whole of safe_run_dir's shape check. This pins the agreement over the
        # shapes we know about; a rule added to safe_run_dir alone introduces a
        # shape that is by definition not in these lists, which is why the
        # coupling is also written down in `refusal_detail`'s docstring.
        with pytest.raises(RunPathError):
            validate_run_id(run_id)
        with pytest.raises(RunPathError):
            safe_run_dir(tmp_path, run_id)

    @pytest.mark.parametrize("sweep_id", ["hs", "nightly-2026-07-30_offset-logit-rescreen"])
    def test_the_real_confirmation_mint_passes_the_gate(self, tmp_path, sweep_id):
        # The premise the handshake rests on: an id select mints is one the
        # gate accepts, so a refusal is never select refusing its own naming.
        # Read out of the config the production mint wrote, not re-templated.
        from panelcast.select.confirmation import run_confirmation
        from panelcast.select.runner import sweep_run_dir

        base = tmp_path / "outputs"
        base.mkdir()
        cfg = _confirmation_cfg(tmp_path, base, sweep_id=sweep_id)
        minted: list[str] = []

        def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None):
            minted.append(_minted_run_id(config_path))
            (base / minted[-1]).mkdir(parents=True)
            return 0, "ok"

        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)

        assert len(minted) == 2  # reference and winner
        for run_id in minted:
            assert validate_run_id(run_id) == run_id
            assert sweep_run_dir(base, run_id).name == run_id

    def test_the_confirmation_mint_can_overshoot_a_legal_sweep_id(self, tmp_path):
        # `SweepConfig` accepts a `sweep_id` up to the run-id limit, but the
        # confirmation mint wraps ~50 more around it, so a legal sweep_id can
        # produce an illegal run_id — the boundary #435 is about. The wrapper
        # is measured from a real mint rather than re-templated, and the long
        # id never becomes a directory component: a 250-character one under
        # tmp_path would blow Windows' MAX_PATH before proving anything.
        from panelcast.select.confirmation import run_confirmation
        from panelcast.select.runner import _claim_named_run, record_key

        base = tmp_path / "outputs"
        base.mkdir()
        minted: list[str] = []

        def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None):
            minted.append(_minted_run_id(config_path))
            (base / minted[-1]).mkdir(parents=True)
            return 0, "ok"

        run_confirmation(
            {"latent_process": "ar1"}, _confirmation_cfg(tmp_path, base), seeds=(42,),
            launch=launch,
        )
        assert minted[0].startswith("sel_s1_confirm_reference_"), minted[0]
        head, sep, tail = minted[0].partition("_s1_")
        assert sep, minted[0]

        long_id = "s" * 250
        assert _confirmation_cfg(tmp_path, base, sweep_id=long_id).sweep_id == long_id
        assert validate_run_id(long_id) == long_id
        with pytest.raises(RunPathError):
            validate_run_id(f"{head}_{long_id}_{tail}")

        # The arm mint wraps less, but not enough less: the same sweep_id
        # overshoots there too, and an arm reaching the handshake with it is
        # refused without being told to go looking for artifacts.
        arm_id = f"sel_{long_id}_{record_key('abc123', 0)}_{minted[0].rsplit('_', 1)[1]}"
        with pytest.raises(RunPathError):
            validate_run_id(arm_id)
        _, problem = _claim_named_run(arm_id, base, {}, datetime.now(), set())
        assert problem is not None and "never wrote under that name" in problem

    def test_undecidable_containment_is_not_reported_as_an_escape(self, tmp_path, monkeypatch):
        # `path_is_within` fails closed on OSError, so an output base that
        # cannot be resolved at all raises the same RunPathError a genuine
        # escape does. Nothing left the root, and the message must not say it
        # did.
        from panelcast.select.runner import _claim_named_run

        base = tmp_path / "outputs"
        base.mkdir()

        def fail(_self, strict=False):
            raise OSError("cwd is gone")

        monkeypatch.setattr(Path, "resolve", fail)
        run_dir, problem = _claim_named_run(
            "sel_s1_x_20260730T120000123456", base, {}, datetime.now(), set()
        )

        assert run_dir is None
        assert problem is not None
        assert "could not be decided" in problem
        # Including the exception tail, which carries `path_is_within`'s
        # fail-closed default wording — the claim this branch exists to deny.
        assert "resolves outside" not in problem
        assert "artifacts may exist" not in problem
        assert "arm run_id" in problem
        assert "cwd is gone" in problem  # the errno the arm has no traceback for
        # This base is absolute, so the message must not blame a relative one.
        assert "is relative" not in problem
        assert "could not be resolved" in problem

    def test_a_confirmation_refused_before_launching_reports_the_real_cause(
        self, tmp_path, monkeypatch
    ):
        # The undecidable state is process-wide, so on the confirmation path it
        # is the *pre-launch* resolve that meets it first — with the relative
        # output base that is the only configuration which can reach it.
        from panelcast.select.confirmation import run_confirmation

        monkeypatch.chdir(tmp_path)
        base = Path("outputs")
        (tmp_path / "outputs").mkdir()
        cfg = _confirmation_cfg(tmp_path, base)
        # Load-bearing: warms the cached sweep_dir, which resolves, so the
        # patch below cannot make the sweep's own directory unreachable.
        assert cfg.sweep_dir == tmp_path / "select" / "s1"
        launches: list[Path] = []

        def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None):
            launches.append(config_path)
            return 0, "ok"

        def fail(_self, strict=False):
            raise OSError("cwd is gone")

        monkeypatch.setattr(Path, "resolve", fail)
        result = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)

        error = result.seeds[0].error
        assert error is not None
        assert launches == []
        assert "before launching" in error
        assert "could not be decided" in error
        assert "resolves outside" not in error
        assert "artifacts may exist" not in error
        # The lexical spelling is the only one available once the cwd read
        # that would absolutize it is what failed; the errno is the evidence.
        assert "outputs could not be resolved" in error
        assert "cwd is gone" in error

    def test_a_pre_launch_breadcrumb_never_claims_artifacts_exist(self, tmp_path):
        # Defensive branch: reaching it needs something already at a name minted
        # microseconds earlier, so drive `refusal_detail` directly rather than
        # pretend a sweep can get there.
        from panelcast.select.runner import refusal_detail

        base = tmp_path / "outputs"
        base.mkdir()
        exc = RunPathError("Invalid confirmation run_id: 'x' resolves outside the output root")

        before = refusal_detail(
            base, "sel_s1_x_20260730T120000123456", exc,
            field="confirmation run_id", after_fit=False,
        )
        after = refusal_detail(
            base, "sel_s1_x_20260730T120000123456", exc,
            field="confirmation run_id", after_fit=True,
        )

        assert "nothing had been written yet" in before
        assert "artifacts may exist" not in before
        assert "artifacts may exist outside the output base" in after
        assert str(base / "sel_s1_x_20260730T120000123456") in before

    def test_an_undecidable_refusal_builds_no_breadcrumb(self, tmp_path, monkeypatch):
        # Pins the ordering `refusal_detail`'s docstring argues for: the
        # unresolvable check runs before the breadcrumb is built. The path that
        # failed is named; the run dir is not.
        from panelcast.select.runner import refusal_detail

        monkeypatch.chdir(tmp_path)
        run_id = "sel_s1_x_20260730T120000123456"
        exc = RunPathError(f"Invalid arm run_id: {run_id!r} resolves outside the output root")

        def fail(_self, strict=False):
            raise OSError("cwd is gone")

        monkeypatch.setattr(Path, "resolve", fail)
        detail = refusal_detail(Path("outputs"), run_id, exc, field="arm run_id", after_fit=True)

        assert "could not be decided" in detail
        assert "arm run_id" in detail
        # The base is what failed here, so that is what gets named — no
        # breadcrumb for the run dir, which is the point of the ordering.
        assert "outputs could not be resolved" in detail
        # The id is valid — it passed the shape gate — so nothing convicts it.
        assert "Invalid" not in detail
        assert run_id not in detail.split("containment for arm run_id")[0]
        assert str(Path("outputs") / run_id) not in detail

    def test_a_run_name_pointing_at_the_root_is_not_an_escape(self, tmp_path):
        # `path_is_within` wants a strict descendant, so a name symlinked at
        # the root is refused — correctly, it is not a run directory — but
        # nothing left the root and the message must not say it did.
        from panelcast.select.runner import _claim_named_run

        base = tmp_path / "outputs"
        base.mkdir()
        run_id = "sel_s1_root_20260730T120000123456"
        _symlink_dir(base / run_id, base)

        _, problem = _claim_named_run(run_id, base, {}, datetime.now(), set())

        assert problem is not None
        assert "resolves to the output root" in problem
        assert "artifacts may exist" not in problem
        assert " -> " not in problem
        # Including the exception tail, whose fail-closed default says the id
        # left the root — this is the one refusal where it wrote into it.
        assert "resolves outside" not in problem
        assert "landed in the root itself" in problem
        assert str(base / run_id) in problem
        assert str(base) in problem

    def test_a_root_pointing_name_is_named_absolutely(self, tmp_path, monkeypatch):
        # This is the refusal that asks someone to remove a link and sweep a
        # root, so both paths have to be followable from wherever they read it.
        from panelcast.select.runner import _claim_named_run

        monkeypatch.chdir(tmp_path)
        base = Path("outputs")
        base.mkdir()
        run_id = "sel_s1_root_20260730T120000123456"
        # Absolute target: a relative one would resolve against the link's own
        # directory and name a child of the root, not the root.
        _symlink_dir(base / run_id, base.absolute())

        _, problem = _claim_named_run(run_id, base, {}, datetime.now(), set())

        assert problem is not None and "resolves to the output root" in problem
        named = problem.split("the refused name ", 1)[1].split(" resolves to", 1)[0]
        root = problem.split("the output root ", 1)[1].split(" itself", 1)[0]
        assert Path(named).is_absolute()
        assert Path(root).is_absolute()
        assert Path(named).parent == Path(root)

    def test_a_root_pointing_name_refused_before_launching_claims_no_write(self, tmp_path):
        # The root branch sits above the after_fit split, so it has to honour
        # it: a pre-launch refusal means nothing ran through that name.
        from panelcast.select.runner import refusal_detail

        base = tmp_path / "outputs"
        base.mkdir()
        run_id = "sel_s1_root_20260730T120000123456"
        _symlink_dir(base / run_id, base)

        detail = refusal_detail(
            base,
            run_id,
            RunPathError("Invalid confirmation run_id: resolves outside the output root"),
            field="confirmation run_id",
            after_fit=False,
        )

        assert "resolves to the output root" in detail
        assert "nothing ran, so nothing was written through it" in detail
        assert "landed in the root" not in detail

    def test_an_unresolvable_run_name_is_undecidable_too(self, tmp_path, monkeypatch):
        # `path_is_within` resolves both operands, so a run name that will not
        # resolve — a symlink loop on the Pythons that raise for one — must not
        # fall through to the escape wording just because the base resolves.
        from panelcast.select.runner import refusal_detail

        base = tmp_path / "outputs"
        base.mkdir()
        run_id = "sel_s1_x_20260730T120000123456"
        real = Path.resolve

        def fail_for_the_run_name(self, strict=False):
            if self.name == run_id:
                raise RuntimeError(f"Symlink loop from {str(self)!r}")
            return real(self)

        monkeypatch.setattr(Path, "resolve", fail_for_the_run_name)
        detail = refusal_detail(
            base,
            run_id,
            RunPathError("Invalid arm run_id: resolves outside the output root"),
            field="arm run_id",
            after_fit=True,
        )

        assert "could not be decided" in detail
        assert "artifacts may exist" not in detail
        assert " -> " not in detail
        assert "Symlink loop" in detail
        # It is the name that failed, so that is the path the message names —
        # the base resolved fine and is not blamed for it.
        assert f"{base / run_id} could not be resolved" in detail
        assert f"{base} could not be resolved" not in detail

    def test_the_arm_mint_shape_passes_the_gate(self, tmp_path):
        # The arm mint has no cheap production seam (it happens inside
        # `run_sweep`), so this pins its shape with the real `record_key` —
        # the rung suffix is the segment most likely to grow a bad character.
        from panelcast.select.runner import record_key, sweep_run_dir

        stamp = f"{datetime.now():%Y%m%dT%H%M%S%f}"
        for rung in (0, 3):
            run_id = f"sel_hs_{record_key('abc123', rung)}_{stamp}"
            assert validate_run_id(run_id) == run_id
            assert sweep_run_dir(tmp_path, run_id).name == run_id

    def test_sweep_run_dir_returns_an_absolute_path(self, tmp_path, monkeypatch):
        from panelcast.select.runner import sweep_run_dir

        monkeypatch.chdir(tmp_path)
        base = Path("outputs")
        (base / "sel_s1_abc123@r1_20260709T120000123456").mkdir(parents=True)
        resolved = sweep_run_dir(base, "sel_s1_abc123@r1_20260709T120000123456")
        assert resolved.is_absolute()
        assert resolved == (tmp_path / "outputs" / "sel_s1_abc123@r1_20260709T120000123456")
        # The documented headline property: the run dir need not exist, and
        # neither need the output base a confirmation-only entry point names.
        assert sweep_run_dir(base, "sel_s1_never_created_20260709T120000123456").is_absolute()
        assert sweep_run_dir(Path("fresh") / "outputs", "sel_s1_x_20260709T120000123456") == (
            tmp_path / "fresh" / "outputs" / "sel_s1_x_20260709T120000123456"
        )

    def test_claim_named_run_accepts_a_minted_id(self, tmp_path):
        from panelcast.select.runner import _claim_named_run

        base = tmp_path / "outputs"
        launched_at = datetime.now()
        run_id = "sel_s1_abc123_20260709T120000123456"
        _write_arm_manifest(base / run_id, launched_at)
        claimed: set[str] = set()

        run_dir, problem = _claim_named_run(run_id, base, {}, launched_at, claimed)

        assert problem is None
        assert run_dir == (base / run_id).resolve()
        assert claimed == {str((base / run_id).resolve())}

    @pytest.mark.parametrize("run_id", ESCAPING_IDS + MALFORMED_IDS + RESERVED_IDS)
    def test_claim_named_run_refuses_bad_ids(self, tmp_path, run_id):
        # Where an id names a target at all — `outside` for the escapes,
        # `latest` for the reserved names — that target is a fully valid,
        # claimable run, so nothing but the id gate stands between the lookup
        # and attributing it to this arm.
        from panelcast.select.runner import _claim_named_run

        base = tmp_path / "outputs"
        base.mkdir()
        launched_at = datetime.now()
        _write_arm_manifest(tmp_path / "outside", launched_at)
        _write_arm_manifest(base / "latest", launched_at)
        claimed: set[str] = set()

        run_dir, problem = _claim_named_run(run_id, base, {}, launched_at, claimed)

        assert run_dir is None
        # Only the RunPathError branch names the field, so an ordinary
        # "was never created" miss cannot make this assertion pass.
        assert problem is not None and "arm run_id" in problem
        # There is no pre-launch arm resolve, so every containment refusal here
        # is the post-fit case, and the message has to say so. None of these
        # ids is well formed, so the orchestrator would have refused the same
        # shape before creating anything — including "latest", which names a
        # real directory here that this run nonetheless never wrote. The
        # message must not send anyone looking.
        assert "after the arm ran" in problem
        assert "never wrote under that name" in problem
        assert "artifacts may exist" not in problem
        assert claimed == set()

    def test_claim_named_run_refuses_a_symlinked_escape(self, tmp_path):
        from panelcast.select.runner import _claim_named_run

        base = tmp_path / "outputs"
        base.mkdir()
        outside = tmp_path / "outside"
        launched_at = datetime.now()
        _write_arm_manifest(outside, launched_at)
        _symlink_dir(base / "sel_s1_escape_20260709T120000123456", outside)
        claimed: set[str] = set()

        run_dir, problem = _claim_named_run(
            "sel_s1_escape_20260709T120000123456", base, {}, launched_at, claimed
        )

        assert run_dir is None
        assert problem is not None and "arm run_id" in problem
        # There is no pre-launch arm resolve, so every containment refusal here
        # is the post-fit case, and the message has to say so — and follow the
        # link, since its target is where the arm's artifacts actually landed.
        # (A link that *loops* rather than escaping is a different case, and a
        # version-dependent one: 3.11/3.12 raise RuntimeError, which
        # `path_is_within` now catches and refuses, while 3.13+ resolve it to
        # itself — still inside the root, so contained and not refused.)
        assert "after the arm ran" in problem
        assert "artifacts may exist outside the output base" in problem
        assert " -> " in problem
        named = problem.split(" -> ", 1)[1].split(" is left in place")[0]
        assert Path(named).is_absolute()
        assert Path(named).resolve() == outside.resolve()
        assert claimed == set()

    def test_claim_named_run_never_creates_the_run_dir(self, tmp_path):
        from panelcast.select.runner import _claim_named_run

        base = tmp_path / "outputs"
        base.mkdir()
        run_dir, problem = _claim_named_run(
            "sel_s1_missing_20260709T120000123456", base, {}, datetime.now(), set()
        )
        assert run_dir is None
        assert problem is not None and "never created" in problem
        assert list(base.iterdir()) == []

    def test_confirmation_lookup_fails_closed_on_a_mutated_sweep_id(self, tmp_path):
        # `SweepConfig` validates `sweep_id` at construction, so reach the
        # unvalidated state the way a caller that skipped setup order would:
        # mutate it after the sweep dir is already cached.
        from panelcast.select.confirmation import run_confirmation
        from panelcast.select.runner import SweepConfig

        base = tmp_path / "outputs"
        base.mkdir()
        (tmp_path / "outside").mkdir()
        # Built inline rather than through `_confirmation_cfg` because the
        # sweep_dir read below has to happen before the mutation.
        cfg = SweepConfig(
            sweep_id="s1",
            output_root=tmp_path / "select",
            panelcast_bin="pc",
            pipeline_output_base=base,
        )
        # Load-bearing: reading sweep_dir warms the cached property, so the
        # mutation below cannot retroactively move the sweep's own directory.
        assert cfg.sweep_dir == tmp_path / "select" / "s1"
        cfg.sweep_id = "../outside"
        launches: list[Path] = []

        def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None):
            launches.append(config_path)
            return 0, "ok"

        result = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)

        assert not result.confirmed
        assert result.seeds[0].error is not None
        assert "confirmation run_id" in result.seeds[0].error
        assert result.seeds[0].reference_run is None
        # The id is decidable up front, so no fit is paid for before refusing.
        assert launches == []
        assert "before launching" in result.seeds[0].error
        # The shape branch is reachable from confirmation too, not just arms.
        assert "never wrote under that name" in result.seeds[0].error

    def _escaping_confirmation(self, tmp_path, relative: bool = False, base: Path | None = None):
        """Run a confirmation whose fit plants an escaping link at its run name."""
        from panelcast.select.confirmation import run_confirmation

        base = tmp_path / "outputs" if base is None else base
        base.mkdir(exist_ok=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep", encoding="utf-8")
        links: list[Path] = []

        def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None):
            link = base / _minted_run_id(config_path)
            _symlink_dir(link, Path("..") / "outside" if relative else outside)
            links.append(link)
            return 0, "ok"

        result = run_confirmation(
            {"latent_process": "ar1"}, _confirmation_cfg(tmp_path, base), seeds=(42,),
            launch=launch,
        )
        return result, links, outside

    def test_confirmation_lookup_refuses_a_symlinked_escape(self, tmp_path):
        # The up-front resolve is lexical (nothing exists yet), so the symlink
        # half of containment has to bite on the post-fit lookup instead.
        result, links, outside = self._escaping_confirmation(tmp_path)

        assert not result.confirmed
        assert result.seeds[0].error is not None
        assert "confirmation run_id" in result.seeds[0].error
        assert result.seeds[0].reference_run is None
        # The fit ran, so this is the post-fit branch. The lookup is read-only:
        # the escaping name survives, named in the error as the breadcrumb to
        # wherever the fit's artifacts actually landed (#413).
        assert "after its fit" in result.seeds[0].error
        assert str(links[0]) in result.seeds[0].error
        assert links[0].is_symlink()
        assert (outside / "keep.txt").exists()

    def test_the_breadcrumb_names_a_relative_link_absolutely(self, tmp_path):
        # readlink() returns the link's contents verbatim, which for a relative
        # link means nothing without knowing where the link itself sits. The
        # join stays un-normalized on purpose: a lexical normpath would lie
        # whenever the output base is itself a symlink.
        result, _, outside = self._escaping_confirmation(tmp_path, relative=True)

        error = result.seeds[0].error
        assert error is not None and " -> " in error
        named = error.split(" -> ", 1)[1].split(" is left in place")[0]
        assert Path(named).is_absolute()
        assert Path(named).resolve() == outside.resolve()

    def test_the_breadcrumb_is_absolute_from_a_relative_output_base(
        self, tmp_path, monkeypatch
    ):
        # `pipeline_output_base` defaults to the relative `outputs`, so without
        # the absolute() the whole breadcrumb — link and target — would only
        # mean anything to a reader standing where the process stood.
        monkeypatch.chdir(tmp_path)
        result, _, outside = self._escaping_confirmation(
            tmp_path, relative=True, base=Path("outputs")
        )

        error = result.seeds[0].error
        assert error is not None and " -> " in error
        pointer = error.split("anything at ", 1)[1].split(" -> ", 1)[0]
        named = error.split(" -> ", 1)[1].split(" is left in place")[0]
        assert Path(pointer).is_absolute()
        assert Path(named).is_absolute()
        assert Path(named).resolve() == outside.resolve()

    def test_a_failed_absolute_does_not_eat_the_refusal(self, tmp_path, monkeypatch):
        # absolute() reads the cwd on a relative output base, so it is inside
        # the same guard: a failure there costs the absolute spelling, not the
        # refusal. (A genuinely removed cwd never reaches here — `path_is_within`
        # turns the OSError into the RunPathError this is formatting.)
        def refuse_absolute(self: Path) -> Path:
            raise OSError("cwd is gone")

        monkeypatch.setattr(Path, "absolute", refuse_absolute)
        result, links, _ = self._escaping_confirmation(tmp_path)

        error = result.seeds[0].error
        assert error is not None
        assert "confirmation run_id" in error
        assert "after its fit" in error
        assert str(links[0]) in error
        assert " -> " not in error

    def test_an_unreadable_link_keeps_the_absolute_spelling(self, tmp_path, monkeypatch):
        # The cross the other two tests never make: relative base × failing hop.
        # `refused` is assigned before the hop precisely so the absolute form
        # already earned survives a readlink that fails after it.
        def refuse_readlink(self: Path) -> Path:
            raise OSError("gone")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(Path, "readlink", refuse_readlink)
        result, _, _ = self._escaping_confirmation(tmp_path, base=Path("outputs"))

        error = result.seeds[0].error
        assert error is not None and " -> " not in error
        pointer = error.split("anything at ", 1)[1].split(" is left in place")[0]
        assert Path(pointer).is_absolute()

    def test_an_unreadable_link_does_not_eat_the_refusal(self, tmp_path, monkeypatch):
        # Formatting the breadcrumb must never replace the containment error.
        def refuse_readlink(self: Path) -> Path:
            raise OSError("gone")

        monkeypatch.setattr(Path, "readlink", refuse_readlink)
        result, links, _ = self._escaping_confirmation(tmp_path)

        error = result.seeds[0].error
        assert error is not None
        assert "confirmation run_id" in error
        assert "after its fit" in error
        assert str(links[0]) in error  # the name survives; only the hop is lost
        assert " -> " not in error

    def test_confirmation_lookup_tolerates_an_unwritten_output_base(self, tmp_path):
        # A confirmation-only entry point can be the first thing to touch its
        # output base, so resolving before the first fit must not require it.
        from panelcast.select.confirmation import run_confirmation

        base = tmp_path / "fresh_outputs"

        def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None):
            (base / _minted_run_id(config_path)).mkdir(parents=True)
            return 0, "ok"

        result = run_confirmation(
            {"latent_process": "ar1"}, _confirmation_cfg(tmp_path, base), seeds=(42,),
            launch=launch,
        )

        # Both fits resolve and are recorded, which is the property under test.
        # The seed still errors afterwards, because these stub run dirs carry
        # no log-likelihood to pair — that failure must not be a containment one.
        assert result.seeds[0].reference_run is not None
        assert result.seeds[0].winner_run is not None
        assert "confirmation run_id" not in (result.seeds[0].error or "")
        assert "refused" not in (result.seeds[0].error or "")

    def test_a_refused_run_dir_is_never_deleted(self, tmp_path, monkeypatch):
        # A run name can stop being contained without being a symlink itself
        # (a containment change in a parent produces the same refusal), so
        # drive the post-fit refusal directly: whatever is at that name — real
        # directory or link — the read-only lookup leaves it alone.
        from panelcast.select import confirmation as confirmation_module
        from panelcast.select.confirmation import run_confirmation

        base = tmp_path / "outputs"
        base.mkdir()
        real = confirmation_module.sweep_run_dir
        calls: list[str] = []

        def only_after_fit(output_base: Path, run_id: str, *, field: str = "run_id") -> Path:
            calls.append(run_id)
            if len(calls) == 2:  # the reference fit's post-fit resolve
                raise RunPathError(f"Invalid {field}: simulated post-fit rejection")
            return real(output_base, run_id, field=field)

        monkeypatch.setattr(confirmation_module, "sweep_run_dir", only_after_fit)

        def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None):
            (base / _minted_run_id(config_path) / "evaluation").mkdir(parents=True)
            return 0, "ok"

        result = run_confirmation(
            {"latent_process": "ar1"}, _confirmation_cfg(tmp_path, base), seeds=(42,),
            launch=launch,
        )

        assert result.seeds[0].error is not None
        assert "after its fit" in result.seeds[0].error
        # Not a link, so the breadcrumb is the name itself, with no hop.
        assert str((base / calls[0]).absolute()) in result.seeds[0].error
        assert " -> " not in result.seeds[0].error
        assert (base / calls[0] / "evaluation").is_dir()
