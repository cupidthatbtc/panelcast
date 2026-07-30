"""Where each run-scoped product is read from and written to.

Writes always land in the current run directory; a ``--stages`` selection that
omits a product's writer redirects that read root to the most recent successful
run which produced it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace as dataclass_replace
from pathlib import Path

import structlog

from panelcast.paths import ArtifactPaths, path_is_within
from panelcast.pipelines.errors import PipelineError
from panelcast.pipelines.manifest import load_run_manifest

log = structlog.get_logger()


def resolve_artifact_paths(
    *,
    run_dir: Path,
    output_base: Path,
    stages: list[str] | None,
    dry_run: bool,
    product_writers: dict[str, tuple[str, ...]],
    product_readers: dict[str, tuple[str, ...]],
    find_source: Callable[[str, tuple[str, ...]], Path | None],
) -> ArtifactPaths:
    """Run-scoped roots, with read roots redirected for consumer-only runs.

    A ``--stages`` selection that excludes a product's writer would
    otherwise look for that product in the just-created (empty) run dir.
    Each such root that a selected stage reads resolves to the most recent
    successful run that produced it; a producer present in the stage list
    wins over latest-run resolution, so ``--stages evaluate,report`` reads
    evaluate's fresh output. Writes always target the current run dir.
    """
    current = ArtifactPaths.for_run(run_dir)
    if stages is None:
        return current  # full run: every product is produced here
    selected = set(stages)
    overrides: dict[str, Path] = {}
    for product, writers in product_writers.items():
        if selected.intersection(writers):
            continue
        readers = selected.intersection(product_readers.get(product, ()))
        if not readers:
            continue
        source = find_source(product, writers)
        if source is None:
            if dry_run:
                # A dry run only previews the plan; a missing source is
                # worth a warning, not a failure.
                log.warning(
                    "artifact_root_unresolved",
                    product=product,
                    readers=sorted(readers),
                )
                continue
            raise PipelineError(
                f"Stage(s) {sorted(readers)} read '{product}' artifacts, but this "
                f"invocation does not run {list(writers)} and no previous "
                f"successful run under {output_base} contains '{product}'. "
                f"Run `panelcast stage {writers[0]}` (or a full `panelcast run`) "
                "first.",
                stage="setup",
            )
        overrides[product] = source / product
        log.info(
            "artifact_root_from_previous_run",
            product=product,
            source_run=source.name,
            readers=sorted(readers),
        )
    return dataclass_replace(current, **overrides) if overrides else current


def find_run_with_product(
    output_base: Path,
    current_run: Path | None,
    product: str,
    writers: tuple[str, ...],
) -> Path | None:
    """Most recent successful non-dry run whose dir contains ``product``."""
    try:
        candidates = sorted(
            (p for p in output_base.iterdir() if p.is_dir()), reverse=True
        )
    except OSError:
        return None
    for run_dir in candidates:
        if run_dir == current_run or run_dir.name in ("latest", "failed"):
            continue
        if not path_is_within(run_dir, output_base):
            continue
        try:
            manifest = load_run_manifest(run_dir / "manifest.json")
        except Exception:
            continue
        if not manifest.success or manifest.flags.get("dry_run"):
            continue
        if not any(w in manifest.stages_completed for w in writers):
            continue
        if (run_dir / product).is_dir():
            return run_dir
    return None
