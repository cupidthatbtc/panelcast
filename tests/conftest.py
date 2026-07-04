"""Pytest configuration and fixtures.

The sys.path manipulation below enables running tests directly via
`pixi run test` without requiring `pip install -e .`. This is simpler
for a research project where the package isn't distributed.

For production projects, prefer using `pip install -e .` and removing
the path manipulation.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The pipeline stages write to these repo-relative dirs by default; a test that
# runs a stage without isolating cwd/output_dir corrupts the real working data.
_GUARDED_DATA_DIRS = (
    ROOT / "data" / "processed",
    ROOT / "data" / "splits",
    ROOT / "data" / "features",
)


def _snapshot_data() -> dict[str, tuple[int, int]]:
    """Map 'dir/relpath' -> (mtime_ns, size) for the repo's real pipeline data."""
    snap: dict[str, tuple[int, int]] = {}
    for base in _GUARDED_DATA_DIRS:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                st = path.stat()
                snap[f"{base.name}/{path.relative_to(base)}"] = (st.st_mtime_ns, st.st_size)
    return snap


@pytest.fixture(autouse=True)
def _guard_repo_data_dirs(request):
    """Fail the offending test if it mutates the repo's real data/{processed,splits,features}.

    Regression guard for #118: a prepare/data run with the default (relative)
    ``output_dir`` silently overwrote the working data with ~50-row fixtures and
    broke a live sweep. Tests that run a pipeline stage must isolate via
    ``output_dir=tmp_path`` or ``monkeypatch.chdir(tmp_path)``; the absolute paths
    here mean correctly-isolated writes under a ``tmp`` cwd don't trip it.
    """
    before = _snapshot_data()
    yield
    after = _snapshot_data()
    if before != after:
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(n for n in set(before) & set(after) if before[n] != after[n])
        raise AssertionError(
            f"{request.node.nodeid} wrote to the repo's real data/ dirs "
            f"(added={added}, removed={removed}, changed={changed}). A pipeline "
            f"stage escaped isolation — pass output_dir=tmp_path or "
            f"monkeypatch.chdir(tmp_path). See issue #118."
        )
