"""Read `pixi.lock` into the package records the audit and SBOM tools share.

pixi records conda packages by URL only, so channel, subdir, name, version, and
build all come from the artifact URL. The `purls` field is the one authoritative
bridge from a conda package to its PyPI identity; native libraries carry no purl
at all, which is what forces the two-tier treatment in `dependency_audit.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "pixi.lock"

CONDA_SUFFIXES = (".conda", ".tar.bz2")


@dataclass(frozen=True)
class LockedPackage:
    kind: str  # "conda" or "pypi"
    name: str
    version: str
    url: str
    sha256: str | None
    pypi_name: str | None  # set when the package is, or maps to, a PyPI distribution
    channel: str | None = None
    subdir: str | None = None
    build: str | None = None

    @property
    def purl(self) -> str:
        if self.kind == "pypi":
            return f"pkg:pypi/{self.name}@{self.version}"
        qualifiers = {"channel": self.channel, "subdir": self.subdir, "build": self.build}
        suffix = "&".join(f"{key}={value}" for key, value in qualifiers.items() if value)
        return f"pkg:conda/{self.name}@{self.version}" + (f"?{suffix}" if suffix else "")


@dataclass(frozen=True)
class Lock:
    packages: tuple[LockedPackage, ...]
    # environment -> platform -> package URLs
    environments: dict[str, dict[str, tuple[str, ...]]]
    digest: str

    def by_url(self) -> dict[str, LockedPackage]:
        return {package.url: package for package in self.packages}

    def platform_packages(self) -> dict[tuple[str, str], tuple[LockedPackage, ...]]:
        index = self.by_url()
        return {
            (environment, platform): tuple(index[url] for url in urls if url in index)
            for environment, platforms in self.environments.items()
            for platform, urls in platforms.items()
        }


def _conda_fields(url: str) -> tuple[str, str, str, str | None, str | None]:
    """(name, version, build, channel, subdir) from a conda artifact URL."""
    parts = url.split("/")
    filename = parts[-1]
    subdir = parts[-2] if len(parts) >= 2 else None
    channel = parts[-3] if len(parts) >= 3 else None
    for suffix in CONDA_SUFFIXES:
        if filename.endswith(suffix):
            filename = filename[: -len(suffix)]
            break
    # Names carry hyphens of their own (aws-c-auth-0.9.3-hef928c7_0), so the only
    # stable split is the last two fields: version and build string.
    fields = filename.rsplit("-", 2)
    if len(fields) != 3:
        raise ValueError(f"unrecognised conda artifact name: {url}")
    return fields[0], fields[1], fields[2], channel, subdir


def _normalise(name: str) -> str:
    return name.lower().replace("_", "-")


def _mapped_pypi_name(entry: dict) -> str | None:
    for purl in entry.get("purls") or []:
        if str(purl).startswith("pkg:pypi/"):
            return _normalise(str(purl)[len("pkg:pypi/") :].split("?", 1)[0].split("@", 1)[0])
    return None


def parse(path: Path | str = DEFAULT_LOCK) -> Lock:
    path = Path(path)
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    document = yaml.safe_load(raw.decode("utf-8"))

    packages: list[LockedPackage] = []
    for entry in document.get("packages") or []:
        if "pypi" in entry:
            name = _normalise(str(entry["name"]))
            packages.append(
                LockedPackage(
                    kind="pypi",
                    name=name,
                    version=str(entry["version"]),
                    url=str(entry["pypi"]),
                    sha256=entry.get("sha256"),
                    pypi_name=name,
                )
            )
        elif "conda" in entry:
            url = str(entry["conda"])
            name, version, build, channel, subdir = _conda_fields(url)
            packages.append(
                LockedPackage(
                    kind="conda",
                    name=name,
                    version=version,
                    url=url,
                    sha256=entry.get("sha256"),
                    pypi_name=_mapped_pypi_name(entry),
                    channel=channel,
                    subdir=subdir,
                    build=build,
                )
            )

    environments: dict[str, dict[str, tuple[str, ...]]] = {}
    for environment, spec in (document.get("environments") or {}).items():
        platforms: dict[str, tuple[str, ...]] = {}
        for platform, entries in (spec.get("packages") or {}).items():
            platforms[platform] = tuple(
                str(entry[kind]) for entry in entries for kind in ("conda", "pypi") if kind in entry
            )
        environments[environment] = platforms

    return Lock(
        packages=tuple(packages),
        environments=environments,
        digest=hashlib.sha256(raw).hexdigest(),
    )
