"""Guard for the hand-synced release metadata.

pyproject.toml is the version's source of truth; CONTRIBUTING's release
procedure hand-syncs pixi.toml, CITATION.cff, and the MODEL_CARD.md header.
requirements.lock likewise attests the SHA256 of pixi.lock. This test fails
the build when any of them drifts.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _pyproject_version() -> str:
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_pixi_toml_version_matches_pyproject():
    with open(REPO / "pixi.toml", "rb") as f:
        pixi_version = tomllib.load(f)["workspace"]["version"]
    assert pixi_version == _pyproject_version()


def test_citation_cff_version_matches_pyproject():
    text = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    m = re.search(r"^version: (\S+)$", text, flags=re.MULTILINE)
    assert m, "CITATION.cff has no version field"
    assert m.group(1) == _pyproject_version()


def test_model_card_version_matches_pyproject():
    text = (REPO / "MODEL_CARD.md").read_text(encoding="utf-8")
    m = re.search(r"^- \*\*Version:\*\* (\S+)$", text, flags=re.MULTILINE)
    assert m, "MODEL_CARD.md has no Version header line"
    assert m.group(1) == _pyproject_version()


def test_requirements_lock_attests_actual_pixi_lock_digest():
    # Normalize CRLF so Windows and Linux checkouts hash identically.
    content = (REPO / "pixi.lock").read_bytes().replace(b"\r\n", b"\n")
    actual = hashlib.sha256(content).hexdigest()
    text = (REPO / "requirements.lock").read_text(encoding="utf-8")
    m = re.search(r"SHA256\(pixi\.lock\)=([0-9a-f]{64})", text)
    assert m, "requirements.lock has no SHA256(pixi.lock) attestation"
    assert m.group(1) == actual, (
        "requirements.lock attests a stale pixi.lock digest; update the "
        f"SHA256(pixi.lock)= line to {actual}"
    )
