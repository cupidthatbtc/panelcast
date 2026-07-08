"""The bump script must rewrite every file test_release_metadata pins."""

from __future__ import annotations

import datetime
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "bump_version", Path(__file__).resolve().parents[2] / "scripts" / "bump_version.py"
)
bump_version = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bump_version)


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.9.0"\n')
    (tmp_path / "pixi.toml").write_text('[workspace]\nname = "x"\nversion = "0.9.0"\n')
    (tmp_path / "CITATION.cff").write_text('version: 0.9.0\ndate-released: "2026-01-01"\n')
    (tmp_path / "MODEL_CARD.md").write_text(
        "- **Version:** 0.9.0\n- **Last updated:** 2026-01-01\n"
    )
    return tmp_path


def test_bump_rewrites_all_pinned_files(repo):
    bump_version.bump(repo, "0.10.0")
    today = datetime.date.today().isoformat()
    assert 'version = "0.10.0"' in (repo / "pyproject.toml").read_text()
    assert 'version = "0.10.0"' in (repo / "pixi.toml").read_text()
    citation = (repo / "CITATION.cff").read_text()
    assert "version: 0.10.0" in citation
    assert f'date-released: "{today}"' in citation
    card = (repo / "MODEL_CARD.md").read_text()
    assert "- **Version:** 0.10.0" in card
    assert f"- **Last updated:** {today}" in card


def test_bump_rejects_malformed_version(repo):
    with pytest.raises(SystemExit):
        bump_version.bump(repo, "not-a-version")


def test_bump_fails_when_a_pin_is_missing(repo):
    (repo / "MODEL_CARD.md").write_text("no version header here\n")
    with pytest.raises(SystemExit):
        bump_version.bump(repo, "0.10.0")
