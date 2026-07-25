"""Domain packs — drop-a-folder-and-run external domains (#276).

A pack bundles everything a paper replication needs into one runnable
folder: the dataset descriptor (the model contract), an optional build step
that reshapes the raw deposit into the tidy panel (the one irreducibly
per-paper file), provenance, and optionally the paper's machine-checkable
claims (#272). ``pack.yaml`` is the discovery/glue manifest.

    <pack-name>/
      pack.yaml            # manifest (schema below)
      descriptor.yaml      # panelcast dataset descriptor
      fit.yaml             # optional pipeline config (priors, gates)
      build.py             # raw deposit -> tidy panel CSV (optional)
      claims.yaml          # optional (#272)
      data/                # raw + built panel (gitignored)
      notes/               # provenance, results
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field

log = structlog.get_logger()


class ExpectedPanel(BaseModel):
    """Post-build sanity gate on the tidy panel."""

    model_config = ConfigDict(extra="forbid")

    rows: int = Field(ge=1)
    entities: int = Field(ge=1)


class PackData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    license: str | None = None
    fetch: str | None = None
    manual: str | None = None
    checksums: str | None = None
    expected_panel: ExpectedPanel | None = None


class PackPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation: str
    doi: str | None = None


class PackManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    paper: PackPaper
    data: PackData = Field(default_factory=PackData)
    descriptor: str = "descriptor.yaml"
    fit: str | None = None
    build: str | None = None
    claims: str | None = None
    # Optional PipelineConfig overrides; validated against real field names
    # at load so a typo dies at read time, not hours into a fit.
    run: dict[str, object] = Field(default_factory=dict)


def load_pack(pack_dir: Path | str) -> tuple[PackManifest, Path]:
    """Read and validate a pack; returns (manifest, resolved pack dir).

    Unknown manifest keys are fatal; missing provenance (data.source /
    data.license) warns — provenance hygiene, not a hard gate.
    """
    pack_dir = Path(pack_dir).resolve()
    manifest_path = pack_dir / "pack.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{pack_dir} has no pack.yaml — not a domain pack.")
    with open(manifest_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path}: pack.yaml must be a mapping.")
    manifest = PackManifest(**data)

    descriptor_path = pack_dir / manifest.descriptor
    if not descriptor_path.exists():
        raise FileNotFoundError(f"{manifest.name}: descriptor {descriptor_path} is missing.")
    for field in ("fit", "build", "claims"):
        rel = getattr(manifest, field)
        if rel is not None and not (pack_dir / rel).exists():
            raise FileNotFoundError(f"{manifest.name}: {field} file {pack_dir / rel} is missing.")
    if manifest.run:
        from panelcast.pipelines.orchestrator import PipelineConfig

        known = {f.name for f in dataclasses.fields(PipelineConfig)}
        unknown = sorted(set(manifest.run) - known)
        if unknown:
            raise ValueError(
                f"{manifest.name}: run overrides {unknown} are not pipeline "
                "config fields."
            )
    if manifest.data.source is None or manifest.data.license is None:
        log.warning(
            "pack_provenance_incomplete",
            pack=manifest.name,
            missing=[k for k in ("source", "license") if getattr(manifest.data, k) is None],
        )
    return manifest, pack_dir


def ensure_panel(manifest: PackManifest, pack_dir: Path) -> None:
    """Run the pack's build step if the tidy panel is missing, then gate it.

    The descriptor's resolved raw path IS the built panel; a missing panel
    with no build step is an actionable error (fetch/manual instructions).
    """
    from panelcast.config.descriptor import load_descriptor

    descriptor = load_descriptor(str(pack_dir / manifest.descriptor))
    panel_path = descriptor.resolve_raw_path()
    if not panel_path.exists():
        if manifest.build is None:
            hint = manifest.data.fetch or manifest.data.manual or "see the pack README"
            raise FileNotFoundError(
                f"{manifest.name}: panel {panel_path} is missing and the pack "
                f"declares no build step ({hint})."
            )
        log.info("pack_building_panel", pack=manifest.name, build=manifest.build)
        result = subprocess.run(
            [sys.executable, str(pack_dir / manifest.build)], cwd=pack_dir, check=False
        )
        # Re-resolve: pre-build, a relative raw_path_default falls through to
        # a bare CWD-relative path (the descriptor-dir fallback only fires
        # for existing files), so the pre-build resolution goes stale the
        # moment build.py writes under the pack dir.
        panel_path = descriptor.resolve_raw_path()
        if result.returncode != 0 or not panel_path.exists():
            raise RuntimeError(
                f"{manifest.name}: build step failed (exit {result.returncode}) "
                f"or did not produce {panel_path}."
            )
    expected = manifest.data.expected_panel
    if expected is not None:
        import pandas as pd

        panel = pd.read_csv(panel_path, encoding=descriptor.encoding)
        # raw_column_map is raw-display -> canonical; the panel carries raw
        # headers, so reverse-look the entity column up.
        reverse = {v: k for k, v in descriptor.raw_column_map.items()}
        entity_col = reverse.get(descriptor.entity_col, descriptor.entity_col)
        if entity_col not in panel.columns:
            raise RuntimeError(
                f"{manifest.name}: built panel has no entity column "
                f"'{entity_col}' — columns: {list(panel.columns)}."
            )
        rows, entities = len(panel), panel[entity_col].nunique()
        if rows != expected.rows or entities != expected.entities:
            raise RuntimeError(
                f"{manifest.name}: built panel has {rows} rows / {entities} "
                f"entities, expected {expected.rows} / {expected.entities} — "
                "the deposit or build step changed."
            )


_TEMPLATE_MANIFEST = """\
name: {name}
paper:
  citation: "AUTHOR (YEAR), VENUE"
  # doi: 10.xxxx/xxxxx
data:
  source: "where the raw deposit comes from"
  license: "the deposit's license"
  # fetch: scripts/fetch.sh
  # expected_panel: {{ rows: 1000, entities: 100 }}
descriptor: descriptor.yaml
build: build.py
# fit: fit.yaml
# claims: claims.yaml
# run:
#   min_ratings: 1
"""

_TEMPLATE_DESCRIPTOR = """\
# panelcast dataset descriptor — only the non-default fields.
# See docs/NEW_DOMAIN_PLAYBOOK.md for every field.
name: {name}
"""

_TEMPLATE_BUILD = '''\
"""Reshape the raw deposit into the tidy one-row-per-event panel CSV.

The one irreducibly per-paper step: every raw format differs, so this
cannot live in YAML. Write the panel where descriptor.yaml expects it.
"""

raise SystemExit("fill in build.py for this pack")
'''

_TEMPLATE_GITIGNORE = "data/\noutputs/\n"


def scaffold_pack(name: str, parent: Path | str = ".") -> Path:
    """Create a skeleton pack so contributors start from a valid layout."""
    import re

    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is None:
        # The manifest pattern, enforced up front: the scaffold must produce
        # a pack load_pack accepts, not one it rejects on first run.
        raise ValueError(
            f"pack name '{name}' must be lowercase kebab/snake case "
            "(pattern [a-z0-9][a-z0-9_-]*)."
        )
    pack_dir = Path(parent) / name
    if pack_dir.exists():
        raise FileExistsError(f"{pack_dir} already exists.")
    (pack_dir / "notes").mkdir(parents=True)
    (pack_dir / "data").mkdir()
    (pack_dir / "pack.yaml").write_text(_TEMPLATE_MANIFEST.format(name=name), encoding="utf-8")
    (pack_dir / "descriptor.yaml").write_text(
        _TEMPLATE_DESCRIPTOR.format(name=name), encoding="utf-8"
    )
    (pack_dir / "build.py").write_text(_TEMPLATE_BUILD, encoding="utf-8")
    (pack_dir / ".gitignore").write_text(_TEMPLATE_GITIGNORE, encoding="utf-8")
    (pack_dir / "README.md").write_text(
        f"# {name}\n\nFill pack.yaml, descriptor.yaml, and build.py; then\n"
        f"`panelcast replicate {name}`.\n",
        encoding="utf-8",
    )
    return pack_dir
