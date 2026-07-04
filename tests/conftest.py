"""Pytest configuration and fixtures.

The sys.path manipulation below enables running tests directly via
`pixi run test` without requiring `pip install -e .`. This is simpler
for a research project where the package isn't distributed.

For production projects, prefer using `pip install -e .` and removing
the path manipulation.
"""

import dataclasses
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panelcast.paths import ArtifactPaths  # noqa: E402  (needs SRC on sys.path)

# Every cwd-relative artifact root a pipeline stage writes by default. A test
# that runs a stage without isolating cwd/output_dir corrupts the repo's real
# working tree (this is #118). Derived from ArtifactPaths.flat() so any root
# added there is guarded automatically.
_ARTIFACT_ROOTS = ArtifactPaths.flat()
_GUARDED_DIRS = tuple(
    ROOT / getattr(_ARTIFACT_ROOTS, f.name) for f in dataclasses.fields(_ARTIFACT_ROOTS)
)


def _snapshot_artifacts() -> dict[str, tuple[int, int]]:
    """Map 'relpath-from-root' -> (mtime_ns, size) for the repo's real artifact dirs."""
    snap: dict[str, tuple[int, int]] = {}
    for base in _GUARDED_DIRS:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                st = path.stat()
                snap[str(path.relative_to(ROOT))] = (st.st_mtime_ns, st.st_size)
    return snap


@pytest.fixture(autouse=True)
def _guard_repo_artifact_dirs(request):
    """Fail the offending test if it mutates the repo's real artifact dirs.

    Regression guard for #118: a prepare/data run with the default (relative)
    ``output_dir`` silently overwrote the working data with ~50-row fixtures and
    broke a live sweep. The guarded roots are every cwd-relative dir in
    ``ArtifactPaths.flat()`` (data/{processed,splits,features}, models,
    outputs/{evaluation,predictions}, reports), so any stage — prepare, train,
    evaluate, predict, report — that escapes isolation trips this. Tests must
    isolate via ``output_dir=tmp_path`` or ``monkeypatch.chdir(tmp_path)``; the
    absolute paths here mean correctly-isolated writes under a ``tmp`` cwd don't.
    """
    before = _snapshot_artifacts()
    yield
    after = _snapshot_artifacts()
    if before != after:
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(n for n in set(before) & set(after) if before[n] != after[n])
        raise AssertionError(
            f"{request.node.nodeid} wrote to the repo's real artifact dirs "
            f"(added={added}, removed={removed}, changed={changed}). A pipeline "
            f"stage escaped isolation — pass output_dir=tmp_path or "
            f"monkeypatch.chdir(tmp_path). See issue #118."
        )
