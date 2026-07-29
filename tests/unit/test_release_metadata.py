"""Guard for the hand-synced release metadata.

pyproject.toml is the version's source of truth; CONTRIBUTING's release
procedure hand-syncs pixi.toml, CITATION.cff, the MODEL_CARD.md header, and
the CHANGELOG.md entry. requirements.lock likewise attests the SHA256 of
pixi.lock. This test fails the build when any of them drifts.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_CITATION_VERSION_PATTERN = r"^version:[^\S\n]*(\S*)[^\S\n]*$"
_MODEL_CARD_VERSION_PATTERN = r"^- \*\*Version:\*\*[^\S\n]*(\S*)[^\S\n]*$"
# Date fields may carry trailing annotations; version fields may not.
_CITATION_DATE_PATTERN = r"^date-released:[^\S\n]*(\S*)"
_MODEL_CARD_DATE_PATTERN = r"^- \*\*Last updated:\*\*[^\S\n]*(\S*)"


def _pyproject_version() -> str:
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _single_match(
    path: str, pattern: str, field: str, *, repo: Path | None = None
) -> str:
    root = REPO if repo is None else repo
    text = (root / path).read_text(encoding="utf-8")
    compiled = re.compile(pattern, flags=re.MULTILINE)
    assert compiled.groups == 1, (
        f"{path} {field} pattern must capture exactly one value"
    )
    matches = compiled.findall(text)
    assert matches, f"{path} has no {field} field"
    assert len(matches) == 1, f"{path} has {len(matches)} {field} fields; expected one"
    return matches[0]


def _newest_changelog_release(*, repo: Path | None = None) -> tuple[str, str]:
    root = REPO if repo is None else repo
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## \[([^\]]+)\](.*)$", text, flags=re.MULTILINE)
    newest = next((heading for heading in headings if heading[0].lower() != "unreleased"), None)
    assert newest, "CHANGELOG.md has no release heading"
    version, suffix = newest
    release_versions = [value for value, _ in headings if value.lower() != "unreleased"]
    count = release_versions.count(version)
    assert count == 1, f"CHANGELOG.md has {count} headings for release {version}"
    assert re.fullmatch(r"\d+\.\d+\.\d+[^\]]*", version), (
        f"CHANGELOG.md's newest release heading {version!r} is not a version"
    )
    date_match = re.fullmatch(
        r"\s*(?:-|\N{EN DASH}|\N{EM DASH})\s*(\d{4}-\d{2}-\d{2})\s*",
        suffix,
    )
    assert date_match, (
        f"CHANGELOG.md's newest release heading suffix {suffix!r} has no ISO release date"
    )
    return version, date_match.group(1)


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def test_pixi_toml_version_matches_pyproject():
    with open(REPO / "pixi.toml", "rb") as f:
        pixi_version = tomllib.load(f)["workspace"]["version"]
    assert pixi_version == _pyproject_version()


def _assert_version_matches(
    path: str, pattern: str, field: str, *, repo: Path | None = None
) -> None:
    version = _single_match(path, pattern, field, repo=repo)
    # Fixtures vary one metadata surface while the real project version stays authoritative.
    expected = _pyproject_version()
    assert _unquote(version) == expected, (
        f"{path} {field} {version!r} does not match pyproject {expected}"
    )


def test_citation_cff_version_matches_pyproject():
    _assert_version_matches("CITATION.cff", _CITATION_VERSION_PATTERN, "version")


def test_model_card_version_matches_pyproject():
    _assert_version_matches("MODEL_CARD.md", _MODEL_CARD_VERSION_PATTERN, "Version")


def test_changelog_version_matches_pyproject():
    newest, _ = _newest_changelog_release()
    version = _pyproject_version()
    assert newest == version, (
        f"CHANGELOG.md's newest release heading is {newest}, but "
        f"pyproject.toml declares {version}; add the release's changelog entry"
    )


def _iso_date(raw: str, field: str) -> date:
    value = _unquote(raw)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", value), (
        f"{field} value {raw!r} is not an ISO date"
    )
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        message = f"{field} value {raw!r} is not a valid date"
        raise AssertionError(message) from exc


def _release_dates(*, repo: Path | None = None) -> tuple[date, date, date]:
    _, changelog_raw = _newest_changelog_release(repo=repo)
    citation_raw = _single_match(
        "CITATION.cff", _CITATION_DATE_PATTERN, "date-released", repo=repo
    )
    model_card_raw = _single_match(
        "MODEL_CARD.md",
        _MODEL_CARD_DATE_PATTERN,
        "Last updated",
        repo=repo,
    )
    return (
        _iso_date(changelog_raw, "CHANGELOG release date"),
        _iso_date(citation_raw, "CITATION.cff date-released"),
        _iso_date(model_card_raw, "MODEL_CARD Last updated"),
    )


def _assert_release_date_consistency(
    *, repo: Path | None = None, today: date | None = None
) -> None:
    changelog_date, citation_date, model_card_date = _release_dates(repo=repo)
    assert changelog_date == citation_date, (
        "release dates disagree: "
        f"CHANGELOG={changelog_date}, CITATION={citation_date}"
    )
    latest_allowed = (date.today() if today is None else today) + timedelta(days=1)
    assert changelog_date <= latest_allowed, (
        "release date is implausibly in the future: "
        f"release={changelog_date}, latest allowed={latest_allowed}"
    )
    # The model-card field is a document edit date, so it may be newer than release.
    assert model_card_date >= changelog_date, (
        "MODEL_CARD Last updated predates the release: "
        f"MODEL_CARD={model_card_date}, release={changelog_date}"
    )
    assert model_card_date <= latest_allowed, (
        "MODEL_CARD Last updated is implausibly in the future: "
        f"MODEL_CARD={model_card_date}, latest allowed={latest_allowed}"
    )


def test_release_date_consistency():
    _assert_release_date_consistency()


def _write_date_fixture(
    repo: Path,
    *,
    citation: str = "date-released: 2026-07-29\n",
    model_card: str = "- **Last updated:** 2026-07-29\n",
    changelog: str = "## [0.23.0] — 2026-07-29\n",
) -> None:
    (repo / "CITATION.cff").write_text(citation, encoding="utf-8")
    (repo / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    (repo / "CHANGELOG.md").write_text(changelog, encoding="utf-8")


def test_single_match_rejects_duplicate_release_fields(tmp_path):
    (tmp_path / "CITATION.cff").write_text(
        "version: 0.23.0\nversion: 0.23.0\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="has 2 version fields"):
        _single_match(
            "CITATION.cff", _CITATION_VERSION_PATTERN, "version", repo=tmp_path
        )


def test_single_match_rejects_missing_release_field(tmp_path):
    (tmp_path / "CITATION.cff").write_text("title: panelcast\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="has no version field"):
        _single_match(
            "CITATION.cff", _CITATION_VERSION_PATTERN, "version", repo=tmp_path
        )


def test_empty_version_reports_the_file_and_field(tmp_path):
    (tmp_path / "CITATION.cff").write_text("version:\n", encoding="utf-8")
    with pytest.raises(AssertionError, match=r"CITATION.cff version ''"):
        _assert_version_matches(
            "CITATION.cff", _CITATION_VERSION_PATTERN, "version", repo=tmp_path
        )


def test_empty_model_card_version_reports_the_file_and_field(tmp_path):
    (tmp_path / "MODEL_CARD.md").write_text("- **Version:**\n", encoding="utf-8")
    with pytest.raises(AssertionError, match=r"MODEL_CARD.md Version ''"):
        _assert_version_matches(
            "MODEL_CARD.md", _MODEL_CARD_VERSION_PATTERN, "Version", repo=tmp_path
        )


def test_duplicate_with_first_empty_model_card_version_is_rejected(tmp_path):
    (tmp_path / "MODEL_CARD.md").write_text(
        "- **Version:**\n- **Version:** 0.23.0\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="has 2 Version fields"):
        _single_match(
            "MODEL_CARD.md", _MODEL_CARD_VERSION_PATTERN, "Version", repo=tmp_path
        )


def test_version_field_rejects_trailing_content(tmp_path):
    (tmp_path / "CITATION.cff").write_text(
        "version: 0.23.0 leftover\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="has no version field"):
        _single_match(
            "CITATION.cff", _CITATION_VERSION_PATTERN, "version", repo=tmp_path
        )


def test_single_match_requires_exactly_one_capture_group(tmp_path):
    (tmp_path / "CITATION.cff").write_text("version: 0.23.0\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="capture exactly one value"):
        _single_match(
            "CITATION.cff",
            r"^(version):[^\S\n]*(\S*)",
            "version",
            repo=tmp_path,
        )


def test_empty_date_field_fails_clearly(tmp_path):
    _write_date_fixture(tmp_path, citation="date-released:\nauthors:\n")
    with pytest.raises(AssertionError, match=r"value '' is not an ISO date"):
        _assert_release_date_consistency(repo=tmp_path, today=date(2026, 7, 29))


def test_duplicate_with_first_empty_date_field_is_rejected(tmp_path):
    _write_date_fixture(
        tmp_path,
        citation="date-released:\ndate-released: 2026-07-29\n",
    )
    with pytest.raises(AssertionError, match="has 2 date-released fields"):
        _assert_release_date_consistency(repo=tmp_path, today=date(2026, 7, 29))


def test_empty_model_card_date_fails_clearly(tmp_path):
    _write_date_fixture(
        tmp_path,
        model_card="- **Last updated:**\n## Limitations\n",
    )
    with pytest.raises(
        AssertionError, match=r"MODEL_CARD Last updated value '' is not an ISO date"
    ):
        _assert_release_date_consistency(repo=tmp_path, today=date(2026, 7, 29))


def test_duplicate_with_first_empty_model_card_date_is_rejected(tmp_path):
    _write_date_fixture(
        tmp_path,
        model_card=(
            "- **Last updated:**\n"
            "- **Last updated:** 2026-07-29\n"
        ),
    )
    with pytest.raises(AssertionError, match="has 2 Last updated fields"):
        _assert_release_date_consistency(repo=tmp_path, today=date(2026, 7, 29))


def test_iso_date_rejects_invalid_calendar_date():
    with pytest.raises(AssertionError, match="is not a valid date"):
        _iso_date("2026-02-30", "release date")


def test_changelog_rejects_malformed_newest_release_date(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text(
        "## [0.23.0] : 2026-07-29\n\n## [0.22.1] — 2026-07-27\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match=r"suffix .* has no ISO release date"):
        _newest_changelog_release(repo=tmp_path)


def test_newest_changelog_release_skips_unreleased(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\n## [0.23.0] - 2026-07-29\n", encoding="utf-8"
    )
    assert _newest_changelog_release(repo=tmp_path) == ("0.23.0", "2026-07-29")


def test_changelog_rejects_duplicate_current_release(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text(
        "## [0.23.0] — 2026-07-29\n\n## [0.23.0] — 2026-07-28\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match=r"has 2 headings for release 0.23.0"):
        _newest_changelog_release(repo=tmp_path)


def test_release_date_guard_accepts_newer_model_card_and_unquoted_cff(tmp_path):
    _write_date_fixture(
        tmp_path, model_card="- **Last updated:** 2026-07-30 (v0.23.0)\n"
    )
    _assert_release_date_consistency(repo=tmp_path, today=date(2026, 7, 29))


def test_release_date_guard_rejects_stale_model_card(tmp_path):
    _write_date_fixture(
        tmp_path, model_card="- **Last updated:** 2026-07-28\n"
    )
    with pytest.raises(AssertionError, match="predates the release"):
        _assert_release_date_consistency(repo=tmp_path, today=date(2026, 7, 29))


def test_release_date_guard_rejects_future_model_card(tmp_path):
    _write_date_fixture(
        tmp_path, model_card="- **Last updated:** 2027-09-02\n"
    )
    with pytest.raises(AssertionError, match="MODEL_CARD Last updated is implausibly"):
        _assert_release_date_consistency(repo=tmp_path, today=date(2026, 7, 29))


def test_release_date_guard_rejects_future_release(tmp_path):
    _write_date_fixture(
        tmp_path,
        changelog="## [0.24.0] — 2027-07-29\n",
        citation="date-released: 2027-07-29\n",
        model_card="- **Last updated:** 2027-07-29\n",
    )
    with pytest.raises(AssertionError, match="release date is implausibly in the future"):
        _assert_release_date_consistency(repo=tmp_path, today=date(2026, 7, 29))


def test_release_date_guard_rejects_disagreement(tmp_path):
    _write_date_fixture(
        tmp_path, citation='date-released: "2026-07-30"\n'
    )
    with pytest.raises(AssertionError, match="release dates disagree"):
        _assert_release_date_consistency(repo=tmp_path, today=date(2026, 7, 29))


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
