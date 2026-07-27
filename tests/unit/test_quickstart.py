from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _project() -> dict:
    with open(REPO / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]


def _install_cell_source() -> str:
    notebook = json.loads(
        (REPO / "examples" / "quickstart.ipynb").read_text(encoding="utf-8")
    )
    cell = next(
        cell for cell in notebook["cells"] if "%pip install" in "".join(cell["source"])
    )
    return "".join(cell["source"])


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def test_quickstart_jax_range_covers_project_floor():
    project_jax = next(
        item for item in _project()["dependencies"] if item.startswith("jax>=")
    )
    project_floor_match = re.search(r"jax>=([0-9.]+)", project_jax)
    assert project_floor_match

    notebook_range = re.search(
        r'"jax>=([0-9.]+),<([0-9.]+)"', _install_cell_source()
    )
    assert notebook_range, "quickstart must bound JAX to a compatible release line"

    floor = _version_tuple(project_floor_match.group(1))
    lower = _version_tuple(notebook_range.group(1))
    upper = _version_tuple(notebook_range.group(2))
    assert lower <= floor < upper, (
        f"quickstart JAX range {lower}–{upper} excludes project floor {floor}"
    )


def test_quickstart_panelcast_pin_matches_project_version():
    match = re.search(r'"panelcast==([0-9.]+)"', _install_cell_source())
    assert match, "quickstart must install an immutable panelcast release"
    assert match.group(1) == _project()["version"]
