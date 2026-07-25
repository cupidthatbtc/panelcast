"""The replication acceptance recipe's diff and manifest logic (#273)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "replication_acceptance.py"
_spec = importlib.util.spec_from_file_location("replication_acceptance", _SCRIPT)
recipe = importlib.util.module_from_spec(_spec)
sys.modules["replication_acceptance"] = recipe
_spec.loader.exec_module(recipe)


def _verdict(name: str, achieved: str = "match", verdict: str = "PASS", **extra) -> dict:
    return {
        "name": name,
        "quantity": "group_mean_trend",
        "expected": "increasing",
        "observed": "0.5 [0.4, 0.6]",
        "achieved": achieved,
        "target": "match",
        "verdict": verdict,
        **extra,
    }


class TestDiff:
    def test_matching_conclusions_are_clean(self):
        actual = [_verdict("a"), _verdict("b", "qualitative", "DIVERGENCE")]
        # Observed numbers wiggle run to run; only graded conclusions count.
        expected = [
            _verdict("a", observed="0.7 [0.6, 0.8]"),
            _verdict("b", "qualitative", "DIVERGENCE"),
        ]
        assert recipe.diff_verdicts(actual, expected) == []

    def test_grade_change_reported(self):
        lines = recipe.diff_verdicts(
            [_verdict("a", "qualitative", "DIVERGENCE")], [_verdict("a")]
        )
        assert len(lines) == 1
        assert "expected ('match', 'PASS')" in lines[0]

    def test_missing_and_unexpected_claims_reported(self):
        lines = recipe.diff_verdicts([_verdict("new")], [_verdict("old")])
        assert any("missing claim 'old'" in line for line in lines)
        assert any("unexpected claim 'new'" in line for line in lines)

    def test_duplicate_claim_names_rejected(self):
        with pytest.raises(SystemExit, match="duplicate"):
            recipe.diff_verdicts([_verdict("a"), _verdict("a")], [_verdict("a")])


class TestManifest:
    def test_valid_manifest_loads(self, tmp_path):
        path = tmp_path / "m.yaml"
        path.write_text(
            "domains:\n"
            "  - {name: x, dataset: d.yaml, claims: c.yaml, expected: e.json}\n",
            encoding="utf-8",
        )
        domains = recipe.load_manifest(path)
        assert domains[0]["name"] == "x"

    def test_empty_manifest_rejected(self, tmp_path):
        path = tmp_path / "m.yaml"
        path.write_text("domains: []\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="no domains"):
            recipe.load_manifest(path)

    def test_incomplete_entry_rejected(self, tmp_path):
        path = tmp_path / "m.yaml"
        path.write_text("domains:\n  - {name: x, dataset: d.yaml}\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="non-empty string"):
            recipe.load_manifest(path)

    def test_null_and_non_string_fields_rejected(self, tmp_path):
        path = tmp_path / "m.yaml"
        path.write_text(
            "domains:\n"
            "  - {name: x, dataset: null, claims: 3, expected: e.json}\n",
            encoding="utf-8",
        )
        with pytest.raises(SystemExit, match="non-empty string"):
            recipe.load_manifest(path)

    def test_scalar_domains_rejected(self, tmp_path):
        path = tmp_path / "m.yaml"
        path.write_text("domains: 1\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="must be a list"):
            recipe.load_manifest(path)

    def test_non_mapping_shapes_rejected(self, tmp_path):
        top_list = tmp_path / "list.yaml"
        top_list.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="must be a mapping"):
            recipe.load_manifest(top_list)
        scalar_entry = tmp_path / "scalar.yaml"
        scalar_entry.write_text("domains:\n  - just_a_string\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="entries must be mappings"):
            recipe.load_manifest(scalar_entry)

    def test_missing_domain_paths_fail_actionably(self):
        with pytest.raises(SystemExit, match="does not exist"):
            recipe.check_domain_paths(
                {"name": "x", "dataset": "nope/d.yaml", "claims": "nope/c.yaml"}
            )

    def test_committed_manifest_parses(self):
        domains = recipe.load_manifest(
            Path(recipe.REPO_ROOT) / "configs" / "replication_acceptance.yaml"
        )
        assert [d["name"] for d in domains] == ["astro", "baseball", "chess"]
