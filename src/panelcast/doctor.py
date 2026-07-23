"""`panelcast doctor`: read-only environment and reproducibility preflight (#162).

Every check wraps an existing, tested function; the doctor composes them into
one PASS/WARN/FAIL screen so setup problems surface in seconds instead of
mid-run — and "my env drifted" is instantly distinguishable from "the code
broke". Strictly read-only; a probe failure becomes a FAIL row, never a crash.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str
    hint: str | None = None


def _check(name: str, fn) -> CheckResult:
    try:
        return fn()
    except Exception as exc:  # a broken probe is a finding, not a crash
        return CheckResult(name, "FAIL", f"probe failed: {exc}")


def _lockfile() -> CheckResult:
    from panelcast.utils.environment import verify_environment

    status = verify_environment()
    if status.pixi_lock_hash:
        return CheckResult("pixi.lock", "PASS", f"hash {status.pixi_lock_hash[:12]}")
    return CheckResult(
        "pixi.lock", "WARN", "no lockfile found", "run from the repo root or `pixi install`"
    )


def _versions() -> CheckResult:
    from panelcast.pipelines.manifest import capture_environment

    env = capture_environment()
    detail = (
        f"python {env.python_version}, jax {env.jax_version}"
        + (f"/{env.jaxlib_version}" if env.jaxlib_version else "")
        + f", numpyro {env.numpyro_version or 'MISSING'}, arviz {env.arviz_version or 'MISSING'}"
    )
    if env.numpyro_version is None or env.arviz_version is None:
        return CheckResult("versions", "FAIL", detail, "pixi install")
    return CheckResult("versions", "PASS", f"{detail}; fingerprint {env.fingerprint}")


def _accelerator() -> CheckResult:
    import jax

    backend = jax.default_backend()
    devices = ", ".join(d.device_kind for d in jax.devices())
    if backend == "gpu":
        from panelcast.gpu_memory.query import query_gpu_memory

        info = query_gpu_memory()
        detail = f"gpu: {devices} ({info.free_gb:.1f}/{info.total_gb:.1f} GB free)"
        return CheckResult("accelerator", "PASS", detail)
    jax_platforms = os.environ.get("JAX_PLATFORMS")
    note = f" (JAX_PLATFORMS={jax_platforms})" if jax_platforms else ""
    return CheckResult("accelerator", "PASS", f"cpu-only: {devices}{note}")


def _compile_cache() -> CheckResult:
    if os.environ.get("PANELCAST_JAX_CACHE") == "0":
        return CheckResult("compile cache", "WARN", "disabled via PANELCAST_JAX_CACHE=0")
    env_dir = os.environ.get("PANELCAST_JAX_CACHE_DIR")
    cache_dir = Path(env_dir) if env_dir else Path.home() / ".cache" / "panelcast" / "jax"
    probe = cache_dir / ".doctor_probe"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return CheckResult("compile cache", "FAIL", f"{cache_dir} not writable: {exc}")
    n_entries = sum(1 for _ in cache_dir.iterdir())
    return CheckResult("compile cache", "PASS", f"{cache_dir} writable, {n_entries} entries")


def _git() -> CheckResult:
    from panelcast.utils.git_state import capture_git_state

    state = capture_git_state()
    detail = f"{state.commit[:7]} on {state.branch}"
    if state.dirty or state.untracked_count:
        return CheckResult(
            "git",
            "WARN",
            f"{detail}, dirty={state.dirty}, untracked={state.untracked_count}",
            "publication runs should record a clean tree",
        )
    return CheckResult("git", "PASS", detail)


def _dataset(dataset: str | None) -> CheckResult:
    from panelcast.config.descriptor import load_descriptor

    descriptor = load_descriptor(dataset)
    csv_path = descriptor.resolve_raw_path()
    if not csv_path.exists():
        return CheckResult(
            "dataset",
            "FAIL",
            f"{descriptor.name}: raw CSV not found at {csv_path}",
            f"set {descriptor.raw_path_env} or place the file at the descriptor default",
        )
    return CheckResult(
        "dataset", "PASS", f"{descriptor.name}: {csv_path} (hash {descriptor.descriptor_hash()})"
    )


def _data_stamps() -> CheckResult:
    from panelcast.pipelines.stamps import DATA_STAGE_ROOTS, read_stamp

    lines = []
    for stage_name, root in DATA_STAGE_ROOTS.items():
        stamp = read_stamp(root)
        if stamp is None:
            lines.append(f"{stage_name}: unstamped")
        else:
            lines.append(f"{stage_name}: run {stamp.get('run_id', '?')}")
    if all("unstamped" in line for line in lines):
        return CheckResult(
            "data stamps", "WARN", "; ".join(lines), "no prepared data yet — a run builds it"
        )
    return CheckResult("data stamps", "PASS", "; ".join(lines))


def _calibration_store() -> CheckResult:
    from panelcast.gpu_memory.calibration_store import default_store_path, load_records

    records = load_records()
    if not records:
        return CheckResult(
            "calibration",
            "WARN",
            "no fit history — memory/runtime predictions use cold-start anchors (RTX 5090)",
            "predictions sharpen automatically after the first GPU fits",
        )
    return CheckResult(
        "calibration", "PASS", f"{len(records)} fit records at {default_store_path()}"
    )


def _disk() -> CheckResult:
    usage = shutil.disk_usage(Path.cwd())
    free_gb = usage.free / 1024**3
    if free_gb < 5:
        return CheckResult(
            "disk", "FAIL", f"{free_gb:.1f} GB free under {Path.cwd()}", "free disk space"
        )
    if free_gb < 20:
        return CheckResult("disk", "WARN", f"{free_gb:.1f} GB free (publication runs need room)")
    return CheckResult("disk", "PASS", f"{free_gb:.0f} GB free")


def run_doctor(dataset: str | None = None) -> list[CheckResult]:
    """All checks, in dependency-ish order; never raises."""
    return [
        _check("pixi.lock", _lockfile),
        _check("versions", _versions),
        _check("accelerator", _accelerator),
        _check("compile cache", _compile_cache),
        _check("git", _git),
        _check("dataset", lambda: _dataset(dataset)),
        _check("data stamps", _data_stamps),
        _check("calibration", _calibration_store),
        _check("disk", _disk),
    ]
