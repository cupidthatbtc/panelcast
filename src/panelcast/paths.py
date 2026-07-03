"""Artifact root directories for pipeline stages.

Data roots (processed/splits/features) are a deterministic cross-run cache
and stay flat in both layouts; mutable products (models/evaluation/
predictions/reports) can be scoped under a run directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    """Root directories every stage writer/consumer resolves paths against."""

    processed: Path
    splits: Path
    features: Path
    models: Path
    evaluation: Path
    predictions: Path
    reports: Path

    @classmethod
    def flat(cls) -> ArtifactPaths:
        """The legacy flat repository layout."""
        return cls(
            processed=Path("data/processed"),
            splits=Path("data/splits"),
            features=Path("data/features"),
            models=Path("models"),
            evaluation=Path("outputs/evaluation"),
            predictions=Path("outputs/predictions"),
            reports=Path("reports"),
        )

    @classmethod
    def for_run(cls, run_dir: Path) -> ArtifactPaths:
        """Run-scoped layout: mutable products live under ``run_dir``."""
        return cls(
            processed=Path("data/processed"),
            splits=Path("data/splits"),
            features=Path("data/features"),
            models=run_dir / "models",
            evaluation=run_dir / "evaluation",
            predictions=run_dir / "predictions",
            reports=run_dir / "reports",
        )

    @classmethod
    def from_ctx(cls, ctx: object) -> ArtifactPaths:
        """Paths carried by a stage context; flat layout when absent.

        The isinstance check keeps bare test contexts (SimpleNamespace,
        MagicMock) on the legacy flat layout.
        """
        paths = getattr(ctx, "paths", None)
        return paths if isinstance(paths, cls) else cls.flat()
