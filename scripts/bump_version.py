"""Bump the release version across every hand-synced metadata file.

Usage: pixi run bump 0.10.0

pyproject.toml is the source of truth; pixi.toml, CITATION.cff, and the
MODEL_CARD.md header are the static copies no code can read (see
CONTRIBUTING "Releasing"). tests/unit/test_release_metadata.py remains the
CI backstop if any file is edited by hand instead.
"""

from __future__ import annotations

import datetime
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _sub(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if n != 1:
        sys.exit(f"error: {path.name}: pattern {pattern!r} not found")
    path.write_text(new_text, encoding="utf-8")
    print(f"  {path.name}: {replacement}")


def bump(root: Path, version: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+([.-].+)?", version):
        sys.exit(f"error: {version!r} does not look like a version")
    with open(root / "pyproject.toml", "rb") as f:
        current = tomllib.load(f)["project"]["version"]
    today = datetime.date.today().isoformat()

    _sub(root / "pyproject.toml", rf'^version = "{re.escape(current)}"$', f'version = "{version}"')
    _sub(root / "pixi.toml", r'^version = ".*"$', f'version = "{version}"')
    _sub(root / "CITATION.cff", r"^version: \S+$", f"version: {version}")
    _sub(root / "CITATION.cff", r'^date-released: ".*"$', f'date-released: "{today}"')
    _sub(root / "MODEL_CARD.md", r"^- \*\*Version:\*\* \S+$", f"- **Version:** {version}")
    _sub(root / "MODEL_CARD.md", r"^- \*\*Last updated:\*\* \S+$", f"- **Last updated:** {today}")

    print(f"bumped {current} -> {version}")
    print("still manual: the CHANGELOG entry; requirements.lock if pixi.lock changed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: bump_version.py NEW_VERSION")
    bump(REPO, sys.argv[1])
