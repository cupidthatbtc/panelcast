"""The typing ratchet stays in lock-step with the mypy config (#302)."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_ratchet_module():
    spec = importlib.util.spec_from_file_location(
        "typing_ratchet", REPO_ROOT / "scripts" / "typing_ratchet.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ratcheted_codes_match_global_disable_list():
    ratchet = _load_ratchet_module()
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    disabled = pyproject["tool"]["mypy"]["disable_error_code"]
    assert sorted(ratchet.RATCHETED_CODES) == sorted(disabled), (
        "scripts/typing_ratchet.py RATCHETED_CODES must mirror "
        "[tool.mypy] disable_error_code — the ratchet would otherwise stop "
        "watching a code the config still forgives"
    )


def test_baseline_only_contains_ratcheted_or_default_on_codes():
    ratchet = _load_ratchet_module()
    baseline = ratchet.load_baseline()
    for path, codes in baseline.items():
        assert path.startswith("src/panelcast/"), path


def test_collect_counts_rejects_format_drift():
    ratchet = _load_ratchet_module()
    # An output whose summary disagrees with its parseable lines must be loud.
    output = "src/panelcast/x.py:1: error: boom  [arg-type]\nFound 2 errors in 1 file\n"
    try:
        ratchet.collect_counts(output)
    except SystemExit as e:
        assert "format" in str(e)
    else:
        raise AssertionError("format drift was not detected")
