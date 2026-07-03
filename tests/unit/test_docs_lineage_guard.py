"""Guard for docs/DATA_LINEAGE.md's verification checklist.

The lineage doc ends with a checklist of every file path and code symbol it
references. This test parses that checklist and asserts each entry still
exists in src/, so a refactor that moves or renames code cannot silently
strand the doc again (as the cli.py -> cli/ package split once did).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "DATA_LINEAGE.md"
SRC = REPO / "src" / "panelcast"

FILE_ENTRY = re.compile(r"^- \[ \] `(src/panelcast/[\w./]+\.py)`$")
# e.g. "- [ ] `fit_model()` in `fit.py`" or "- [ ] `PriorConfig` dataclass in `priors.py`"
SYMBOL_ENTRY = re.compile(r"^- \[ \] `(\w+)(?:\(\))?`.* in `([\w.]+\.py)`$")


def _checklist_lines() -> list[str]:
    text = DOC.read_text(encoding="utf-8")
    _, _, checklist = text.partition("# VERIFICATION CHECKLIST")
    assert checklist, "docs/DATA_LINEAGE.md lost its VERIFICATION CHECKLIST section"
    # The "No Old Artifact References" tail lists things that must NOT exist.
    checklist, _, _ = checklist.partition("## No Old Artifact References")
    return checklist.splitlines()


def _parse() -> tuple[list[str], list[tuple[str, str]]]:
    paths: list[str] = []
    symbols: list[tuple[str, str]] = []
    for line in _checklist_lines():
        if m := FILE_ENTRY.match(line):
            paths.append(m.group(1))
        elif m := SYMBOL_ENTRY.match(line):
            symbols.append((m.group(1), m.group(2)))
    return paths, symbols


def test_checklist_parses_nontrivially():
    paths, symbols = _parse()
    assert len(paths) >= 40, f"only {len(paths)} file entries parsed - checklist moved?"
    assert len(symbols) >= 40, f"only {len(symbols)} symbol entries parsed - checklist moved?"


def test_checklist_file_paths_exist():
    paths, _ = _parse()
    missing = [p for p in paths if not (REPO / p).is_file()]
    assert not missing, (
        f"docs/DATA_LINEAGE.md references files that no longer exist: {missing}. "
        "Update the doc (body and checklist) to the new locations."
    )


def test_checklist_symbols_exist_in_named_modules():
    _, symbols = _parse()
    stranded = []
    for symbol, module in symbols:
        word = re.compile(rf"\b{re.escape(symbol)}\b")
        candidates = sorted(SRC.rglob(module))
        if not any(word.search(f.read_text(encoding="utf-8")) for f in candidates):
            stranded.append(f"{symbol} in {module}")
    assert not stranded, (
        f"docs/DATA_LINEAGE.md references symbols missing from their named modules: {stranded}. "
        "Update the doc to match the refactored code."
    )
