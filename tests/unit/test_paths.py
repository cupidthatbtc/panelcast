"""Tests for the ArtifactPaths module."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from panelcast.paths import ArtifactPaths
from panelcast.pipelines.stages import StageContext


class TestFlatLayout:
    def test_flat_matches_legacy_literals(self):
        paths = ArtifactPaths.flat()
        assert paths.processed == Path("data/processed")
        assert paths.splits == Path("data/splits")
        assert paths.features == Path("data/features")
        assert paths.models == Path("models")
        assert paths.evaluation == Path("outputs/evaluation")
        assert paths.predictions == Path("outputs/predictions")
        assert paths.reports == Path("reports")


class TestForRunLayout:
    def test_mutable_products_scope_under_run_dir(self):
        run_dir = Path("outputs/2026-01-01_000000")
        paths = ArtifactPaths.for_run(run_dir)
        assert paths.models == run_dir / "models"
        assert paths.evaluation == run_dir / "evaluation"
        assert paths.predictions == run_dir / "predictions"
        assert paths.reports == run_dir / "reports"

    def test_data_roots_stay_flat(self):
        flat = ArtifactPaths.flat()
        paths = ArtifactPaths.for_run(Path("outputs/run"))
        assert paths.processed == flat.processed
        assert paths.splits == flat.splits
        assert paths.features == flat.features


class TestFrozen:
    def test_field_assignment_raises(self):
        paths = ArtifactPaths.flat()
        with pytest.raises(FrozenInstanceError):
            paths.models = Path("elsewhere")


class TestContextIntegration:
    def test_stage_context_defaults_to_flat(self):
        ctx = StageContext(
            run_dir=Path("outputs/run"),
            seed=42,
            strict=False,
            verbose=False,
            manifest=None,
        )
        assert ctx.paths == ArtifactPaths.flat()

    def test_from_ctx_returns_carried_paths(self):
        run_paths = ArtifactPaths.for_run(Path("outputs/run"))
        ctx = SimpleNamespace(paths=run_paths)
        assert ArtifactPaths.from_ctx(ctx) is run_paths

    def test_from_ctx_falls_back_to_flat_when_absent(self):
        assert ArtifactPaths.from_ctx(SimpleNamespace()) == ArtifactPaths.flat()
