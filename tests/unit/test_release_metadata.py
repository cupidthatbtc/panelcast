"""Guard for the hand-synced release metadata.

pyproject.toml is the version's source of truth; CONTRIBUTING's release
procedure hand-syncs pixi.toml, CITATION.cff, the MODEL_CARD.md header, and
the CHANGELOG.md entry. requirements.lock likewise attests the SHA256 of
pixi.lock. This test fails the build when any of them drifts.
"""

from __future__ import annotations

import hashlib
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _pyproject_version() -> str:
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _single_match(path: str, pattern: str, field: str) -> str:
    text = (REPO / path).read_text(encoding="utf-8")
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    assert len(matches) == 1, f"{path} must have exactly one {field} field"
    return matches[0]


def _newest_changelog_release() -> tuple[str, str]:
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## \[([^\]]+)\](.*)$", text, flags=re.MULTILINE)
    newest = next((heading for heading in headings if heading[0].lower() != "unreleased"), None)
    assert newest, "CHANGELOG.md has no release heading"
    version, suffix = newest
    assert re.fullmatch(r"\d+\.\d+\.\d+[^\]]*", version), (
        f"CHANGELOG.md's newest release heading {version!r} is not a version"
    )
    date_match = re.fullmatch(r" — (\d{4}-\d{2}-\d{2})", suffix)
    assert date_match, "CHANGELOG.md's newest release heading has no ISO release date"
    return version, date_match.group(1)


def test_pixi_toml_version_matches_pyproject():
    with open(REPO / "pixi.toml", "rb") as f:
        pixi_version = tomllib.load(f)["workspace"]["version"]
    assert pixi_version == _pyproject_version()


def test_citation_cff_version_matches_pyproject():
    version = _single_match("CITATION.cff", r"^version: (\S+)$", "version")
    assert version == _pyproject_version()


def test_model_card_version_matches_pyproject():
    version = _single_match(
        "MODEL_CARD.md", r"^- \*\*Version:\*\* (\S+)$", "Version header"
    )
    assert version == _pyproject_version()


def test_changelog_version_matches_pyproject():
    newest, _ = _newest_changelog_release()
    version = _pyproject_version()
    assert newest == version, (
        f"CHANGELOG.md's newest release heading is {newest}, but "
        f"pyproject.toml declares {version}; add the release's changelog entry"
    )


def test_release_dates_match():
    _, changelog_date = _newest_changelog_release()
    citation_date = _single_match(
        "CITATION.cff", r'^date-released: "(\d{4}-\d{2}-\d{2})"$', "date-released"
    )
    model_card_date = _single_match(
        "MODEL_CARD.md",
        r"^- \*\*Last updated:\*\* (\d{4}-\d{2}-\d{2})$",
        "Last updated header",
    )
    assert changelog_date == citation_date == model_card_date, (
        "release dates disagree: "
        f"CHANGELOG={changelog_date}, CITATION={citation_date}, MODEL_CARD={model_card_date}"
    )


def test_single_match_rejects_duplicate_release_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    (tmp_path / "CITATION.cff").write_text(
        "version: 0.23.0\nversion: 0.23.0\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="exactly one version"):
        _single_match("CITATION.cff", r"^version: (\S+)$", "version")


def test_changelog_rejects_malformed_newest_release_date(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        "## [0.23.0] - 2026-07-29\n\n## [0.22.1] — 2026-07-27\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="no ISO release date"):
        _newest_changelog_release()


def test_release_date_guard_rejects_disagreement(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        "## [0.23.0] — 2026-07-29\n", encoding="utf-8"
    )
    (tmp_path / "CITATION.cff").write_text(
        'date-released: "2026-07-30"\n', encoding="utf-8"
    )
    (tmp_path / "MODEL_CARD.md").write_text(
        "- **Last updated:** 2026-07-29\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="release dates disagree"):
        test_release_dates_match()


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
