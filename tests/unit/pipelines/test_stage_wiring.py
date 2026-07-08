"""Stage-wise wiring across separate orchestrator invocations.

Covers the audit fixes for the run lifecycle:
- consumer-only invocations (``panelcast stage evaluate``) read upstream
  products from the most recent successful run that produced them, while
  writes go to the current run dir;
- a producer present in the same stage list wins over latest-run resolution;
- dry runs record nothing and never take the latest pointer;
- run directories are created exclusively (no same-second sharing).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from panelcast.paths import ArtifactPaths, resolve_latest
from panelcast.pipelines.errors import PipelineError
from panelcast.pipelines.manifest import EnvironmentInfo
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator
from panelcast.pipelines.stages import PipelineStage
from panelcast.utils.git_state import GitState


@pytest.fixture
def mock_env():
    with (
        patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
        patch(
            "panelcast.pipelines.orchestrator.verify_environment",
            return_value=MagicMock(is_reproducible=True, pixi_lock_hash="abc123", warnings=[]),
        ),
    ):
        yield


@pytest.fixture
def isolated_outputs(tmp_path, monkeypatch, mock_env):
    """Hermetic cwd with a raw input file; returns the output base."""
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw" / "raw.csv"
    raw.parent.mkdir(parents=True)
    raw.write_text("a,b\n1,2\n", encoding="utf-8")
    return tmp_path / "outputs"


def _fake_stages(
    paths: ArtifactPaths,
    executed: list[str],
    reads: dict[str, str],
) -> list[PipelineStage]:
    """Marker-file stages mirroring the real producer/consumer graph.

    Producers write their run id into their product root; consumers record
    the run id they found in their input product (``reads``) and write their
    own product.
    """

    def producer(name: str, root_attr: str, filename: str):
        def run_fn(ctx):
            executed.append(name)
            root = getattr(ctx.paths, root_attr)
            root.mkdir(parents=True, exist_ok=True)
            (root / filename).write_text(ctx.manifest.run_id, encoding="utf-8")

        return run_fn

    def consumer(name: str, read_attr: str, read_file: str, write_attr: str, write_file: str):
        def run_fn(ctx):
            executed.append(name)
            reads[name] = (getattr(ctx.paths, read_attr) / read_file).read_text(encoding="utf-8")
            root = getattr(ctx.paths, write_attr)
            root.mkdir(parents=True, exist_ok=True)
            (root / write_file).write_text(ctx.manifest.run_id, encoding="utf-8")

        return run_fn

    return [
        PipelineStage(
            name="data",
            description="fake data stage",
            run_fn=producer("data", "processed", "marker.parquet"),
            input_paths=[Path("data/raw/raw.csv")],
            output_paths=[Path("data/processed/marker.parquet")],
        ),
        PipelineStage(
            name="train",
            description="fake train stage",
            run_fn=producer("train", "models", "model.txt"),
            input_paths=[Path("data/processed/marker.parquet")],
            output_paths=[paths.models / "model.txt"],
            depends_on=["data"],
        ),
        PipelineStage(
            name="evaluate",
            description="fake evaluate stage",
            run_fn=consumer("evaluate", "models", "model.txt", "evaluation", "metrics.json"),
            input_paths=[paths.models / "model.txt"],
            output_paths=[paths.evaluation / "metrics.json"],
            depends_on=["train"],
        ),
        PipelineStage(
            name="predict",
            description="fake predict stage",
            run_fn=consumer("predict", "models", "model.txt", "predictions", "summary.json"),
            input_paths=[paths.models / "model.txt"],
            output_paths=[paths.predictions / "summary.json"],
            depends_on=["evaluate"],
        ),
        PipelineStage(
            name="report",
            description="fake report stage",
            run_fn=_report_run_fn(executed, reads),
            input_paths=[
                paths.evaluation / "metrics.json",
                paths.predictions / "summary.json",
            ],
            output_paths=[paths.reports / "MODEL_CARD.md"],
            depends_on=["predict"],
        ),
    ]


def _report_run_fn(executed: list[str], reads: dict[str, str]):
    """Report reads BOTH evaluation and predictions, like the real stage."""

    def run_fn(ctx):
        executed.append("report")
        reads["report_evaluation"] = (ctx.paths.evaluation / "metrics.json").read_text(
            encoding="utf-8"
        )
        reads["report_predictions"] = (ctx.paths.predictions / "summary.json").read_text(
            encoding="utf-8"
        )
        ctx.paths.reports.mkdir(parents=True, exist_ok=True)
        (ctx.paths.reports / "MODEL_CARD.md").write_text(ctx.manifest.run_id, encoding="utf-8")

    return run_fn


def _run_pipeline(
    output_base: Path,
    run_id: str,
    stages: list[str] | None = None,
    executed: list[str] | None = None,
    reads: dict[str, str] | None = None,
    **config_kwargs,
) -> tuple[int, PipelineOrchestrator]:
    executed = executed if executed is not None else []
    reads = reads if reads is not None else {}
    config = PipelineConfig(enforce_lockfile=False, stages=stages, **config_kwargs)
    orchestrator = PipelineOrchestrator(config, output_base=output_base)

    def fake_order(requested=None, min_ratings=10, descriptor=None, descriptor_path=None, paths=None):
        all_stages = _fake_stages(paths or ArtifactPaths.flat(), executed, reads)
        if requested is None:
            return all_stages
        return [s for s in all_stages if s.name in set(requested)]

    with (
        patch("panelcast.pipelines.orchestrator.get_execution_order", side_effect=fake_order),
        patch("panelcast.pipelines.orchestrator.generate_run_id", return_value=run_id),
    ):
        return orchestrator.run(), orchestrator


class TestConsumerStageResolution:
    def test_stage_train_then_stage_evaluate_as_separate_invocations(self, isolated_outputs):
        """The documented stage-wise workflow: evaluate in its own invocation
        finds the model trained by the previous invocation."""
        out = isolated_outputs

        code, _ = _run_pipeline(out, "runA", stages=["data", "train"])
        assert code == 0
        assert (out / "runA" / "models" / "model.txt").read_text(encoding="utf-8") == "runA"

        reads: dict[str, str] = {}
        code, _ = _run_pipeline(out, "runB", stages=["evaluate"], reads=reads)
        assert code == 0
        # evaluate read runA's model and wrote into its own run dir.
        assert reads["evaluate"] == "runA"
        assert (out / "runB" / "evaluation" / "metrics.json").read_text(encoding="utf-8") == "runB"
        assert not (out / "runB" / "models").exists()

    def test_producer_in_stage_list_wins_over_latest_run(self, isolated_outputs):
        """`run --stages evaluate,report`: report reads evaluate's fresh
        output from the current run, while absent producers' products
        (models, predictions) come from the runs that made them."""
        out = isolated_outputs

        assert _run_pipeline(out, "runA", stages=["data", "train"])[0] == 0
        assert _run_pipeline(out, "runB", stages=["evaluate", "predict"])[0] == 0

        reads: dict[str, str] = {}
        code, _ = _run_pipeline(out, "runC", stages=["evaluate", "report"], reads=reads)
        assert code == 0
        assert reads["evaluate"] == "runA"  # models still from the training run
        assert reads["report_evaluation"] == "runC"  # THIS run's evaluate wins
        assert reads["report_predictions"] == "runB"  # latest run that predicted

    def test_resolution_skips_runs_missing_the_product(self, isolated_outputs):
        """The latest run may not contain every product: an evaluate-only run
        holds no models, so the next evaluate resolves models further back."""
        out = isolated_outputs

        assert _run_pipeline(out, "runA", stages=["data", "train"])[0] == 0
        assert _run_pipeline(out, "runB", stages=["evaluate"])[0] == 0
        assert resolve_latest(out) == out / "runB"  # latest has no models dir

        reads: dict[str, str] = {}
        code, _ = _run_pipeline(out, "runC", stages=["evaluate"], reads=reads)
        assert code == 0
        assert reads["evaluate"] == "runA"

    def test_consumer_without_prior_run_fails_helpfully(self, isolated_outputs):
        out = isolated_outputs

        code, orchestrator = _run_pipeline(out, "runX", stages=["evaluate"])
        assert code != 0
        assert orchestrator.manifest is not None
        assert "models" in orchestrator.manifest.error
        assert "train" in orchestrator.manifest.error

    def test_dry_run_previews_consumer_plan_without_sources(self, isolated_outputs):
        """A dry run must still show the plan when no prior run exists."""
        out = isolated_outputs

        code, _ = _run_pipeline(out, "runDry", stages=["evaluate"], dry_run=True)
        assert code == 0

    def test_dry_run_is_not_a_resolution_source(self, isolated_outputs):
        out = isolated_outputs

        code, _ = _run_pipeline(out, "runDry", stages=["data", "train"], dry_run=True)
        assert code == 0

        code, orchestrator = _run_pipeline(out, "runB", stages=["evaluate"])
        assert code != 0
        assert "models" in orchestrator.manifest.error

    def test_full_run_keeps_everything_in_its_own_dir(self, isolated_outputs):
        out = isolated_outputs

        reads: dict[str, str] = {}
        code, _ = _run_pipeline(out, "runFull", stages=None, reads=reads)
        assert code == 0
        assert reads["evaluate"] == "runFull"
        assert reads["report_evaluation"] == "runFull"
        assert reads["report_predictions"] == "runFull"
        for product in ("models", "evaluation", "predictions", "reports"):
            assert (out / "runFull" / product).is_dir()


class TestResolveArtifactPathsUnit:
    """_resolve_artifact_paths against hand-written prior runs."""

    @staticmethod
    def _write_run(
        output_base: Path,
        run_id: str,
        stages_completed: list[str],
        products: tuple[str, ...] = (),
        success: bool = True,
        dry_run: bool = False,
    ) -> Path:
        run_dir = output_base / run_id
        run_dir.mkdir(parents=True)
        for product in products:
            (run_dir / product).mkdir()
        manifest = {
            "run_id": run_id,
            "created_at": "2026-01-01T00:00:00",
            "command": "panelcast run",
            "flags": {"dry_run": dry_run},
            "seed": 42,
            "git": {"commit": "abc", "branch": "main", "dirty": False, "untracked_count": 0},
            "environment": {
                "python_version": "3.11",
                "jax_version": "0.4",
                "numpyro_version": None,
                "arviz_version": None,
                "platform": "test",
                "pixi_lock_hash": None,
            },
            "input_hashes": {},
            "stage_hashes": {},
            "stages_completed": stages_completed,
            "stages_skipped": [],
            "outputs": {},
            "success": success,
            "error": None,
            "duration_seconds": 0.0,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return run_dir

    def _orchestrator(self, output_base: Path, stages: list[str] | None) -> PipelineOrchestrator:
        orch = PipelineOrchestrator(PipelineConfig(stages=stages), output_base=output_base)
        orch.run_dir = output_base / "9999-current"
        orch.run_dir.mkdir(parents=True, exist_ok=True)
        return orch

    def test_full_run_is_pure_run_scoped(self, tmp_path):
        orch = self._orchestrator(tmp_path, stages=None)
        assert orch._artifact_paths() == ArtifactPaths.for_run(orch.run_dir)

    def test_consumer_only_redirects_read_root(self, tmp_path):
        train_run = self._write_run(tmp_path, "0001-train", ["data", "train"], ("models",))
        orch = self._orchestrator(tmp_path, stages=["evaluate"])
        paths = orch._artifact_paths()
        assert paths.models == train_run / "models"
        assert paths.evaluation == orch.run_dir / "evaluation"

    def test_failed_unsuccessful_and_dry_runs_are_skipped(self, tmp_path):
        good = self._write_run(tmp_path, "0001-good", ["train"], ("models",))
        self._write_run(tmp_path, "0002-failed", ["train"], ("models",), success=False)
        self._write_run(tmp_path, "0003-dry", ["train"], ("models",), dry_run=True)
        # A run whose manifest lists train but whose models dir is gone.
        pruned = self._write_run(tmp_path, "0004-pruned", ["train"])
        assert not (pruned / "models").exists()

        orch = self._orchestrator(tmp_path, stages=["evaluate"])
        assert orch._artifact_paths().models == good / "models"

    def test_no_source_raises_pipeline_error(self, tmp_path):
        orch = self._orchestrator(tmp_path, stages=["evaluate"])
        with pytest.raises(PipelineError, match="models"):
            orch._resolve_artifact_paths()

    def test_stage_list_without_product_readers_needs_no_source(self, tmp_path):
        orch = self._orchestrator(tmp_path, stages=["data", "splits"])
        assert orch._artifact_paths() == ArtifactPaths.for_run(orch.run_dir)

    def test_sensitivity_reads_models_but_writes_reports_here(self, tmp_path):
        train_run = self._write_run(tmp_path, "0001-train", ["train"], ("models",))
        orch = self._orchestrator(tmp_path, stages=["sensitivity"])
        paths = orch._artifact_paths()
        assert paths.models == train_run / "models"
        assert paths.reports == orch.run_dir / "reports"


class TestDryRunLifecycle:
    def test_dry_run_takes_no_latest_pointer(self, isolated_outputs):
        out = isolated_outputs

        code, orchestrator = _run_pipeline(out, "runDry", dry_run=True)
        assert code == 0
        assert not (out / "latest.json").exists()
        assert not (out / "latest").exists()
        assert resolve_latest(out) is None
        assert orchestrator.manifest.stages_completed == []
        assert orchestrator.manifest.stage_hashes == {}

    def test_skip_existing_after_dry_run_still_executes(self, isolated_outputs):
        out = isolated_outputs

        assert _run_pipeline(out, "runDry", dry_run=True)[0] == 0

        executed: list[str] = []
        code, orchestrator = _run_pipeline(out, "runB", executed=executed, skip_existing=True)
        assert code == 0
        assert "data" in executed  # not skipped off the dry run's manifest
        assert orchestrator.manifest.stages_skipped == []

    def test_resolve_latest_ignores_stale_dry_run_pointer(self, tmp_path):
        """A latest.json written by an older checkout may target a dry run;
        resolve_latest must not serve that artifact-less dir."""
        run_dir = tmp_path / "2026-01-01_000000"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"flags": {"dry_run": True}}), encoding="utf-8"
        )
        (tmp_path / "latest.json").write_text(
            json.dumps({"run_id": run_dir.name, "run_dir": run_dir.name}), encoding="utf-8"
        )
        assert resolve_latest(tmp_path) is None


class TestExclusiveRunDirs:
    def _setup_run(self, orchestrator: PipelineOrchestrator) -> None:
        env = EnvironmentInfo(
            python_version="3.11",
            jax_version="0.4",
            numpyro_version=None,
            arviz_version=None,
            platform="test",
            pixi_lock_hash=None,
        )
        git = GitState(commit="abc", branch="main", dirty=False, untracked_count=0)
        with (
            patch("panelcast.pipelines.orchestrator.capture_environment", return_value=env),
            patch("panelcast.pipelines.orchestrator.capture_git_state", return_value=git),
        ):
            orchestrator._setup_run()

    def test_setup_run_retries_on_existing_dir(self, tmp_path):
        (tmp_path / "dup").mkdir(parents=True)
        orch = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path)
        with patch(
            "panelcast.pipelines.orchestrator.generate_run_id",
            side_effect=["dup", "dup", "fresh"],
        ):
            self._setup_run(orch)
        assert orch.run_dir == tmp_path / "fresh"
        assert orch.manifest.run_id == "fresh"

    def test_setup_run_gives_up_after_retries(self, tmp_path):
        (tmp_path / "dup").mkdir(parents=True)
        orch = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path)
        with (
            patch("panelcast.pipelines.orchestrator.generate_run_id", return_value="dup"),
            pytest.raises(PipelineError, match="unique run directory"),
        ):
            self._setup_run(orch)
