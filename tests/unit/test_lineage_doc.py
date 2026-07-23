"""Lineage-page link/claim checks (#301)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

LINEAGE = (REPO_ROOT / "docs" / "LINEAGE.md").read_text("utf-8")


def test_lineage_names_the_predecessor_and_dates():
    assert "aoty_pred_pub" in LINEAGE
    assert "2026-06-20" in LINEAGE  # migration boundary
    assert "2026-01-01" in LINEAGE  # predecessor start


def test_lineage_documents_the_joss_floor():
    assert "2026-12-21" in LINEAGE
    assert "January" in LINEAGE and "2027" in LINEAGE


def test_readme_and_contributing_link_the_lineage_page():
    assert "docs/LINEAGE.md" in (REPO_ROOT / "README.md").read_text("utf-8")
    assert "docs/LINEAGE.md" in (REPO_ROOT / "CONTRIBUTING.md").read_text("utf-8")


def test_lineage_internal_links_resolve():
    for target in ("docs/PORTING.md", "CHANGELOG.md", "MODEL_CARD.md"):
        assert (REPO_ROOT / target).exists(), target
