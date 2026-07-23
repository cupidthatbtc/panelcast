"""Terminology-ratchet integrity checks (#303)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "terminology_ratchet", REPO_ROOT / "scripts" / "terminology_ratchet.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sanctioned_files_exist():
    ratchet = _load()
    missing = [rel for rel in ratchet.SANCTIONED if not (ratchet.SRC / rel).exists()]
    assert not missing, f"SANCTIONED entries with no file (remove them): {missing}"


def test_baseline_paths_exist():
    ratchet = _load()
    baseline = ratchet.load_baseline()
    missing = [rel for rel in baseline if not (ratchet.SRC / rel).exists()]
    assert not missing, f"baseline entries with no file (re-bank): {missing}"


def test_current_counts_match_baseline():
    # The committed baseline must be exact right now; drift is caught here in
    # the fast suite as well as by the dedicated CI step.
    ratchet = _load()
    assert ratchet.check(ratchet.collect_counts()) == 0


def test_pattern_catches_derived_identifiers():
    ratchet = _load()
    for token in ("n_artists", "album_seq", "AlbumTypeBlock", "mu_artist", "max_albums"):
        assert ratchet._TERM_RE.search(token), token
    assert not ratchet._TERM_RE.search("entity_idx event_seq")
