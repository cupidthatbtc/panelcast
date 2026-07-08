"""Rolling-origin backtest: headline metrics as a distribution, not a point.

Runs the full leakage-safe stage chain (splits -> features -> train ->
evaluate) once per origin k = 0..K-1, where origin k holds out each entity's
(last-k)-th event and drops everything after it. Each origin is a normal
run directory launched as a subprocess (the pattern proven in
``select/runner.py``); a JSON ledger under ``outputs/backtest/<id>/`` makes
a killed backtest resume at the next unfinished origin.

Populations are not identical across origins — deeper origins shrink the
eligible entity set — so the aggregate table reports n_test and n_entities
per origin and cross-origin variation must be read with that in mind.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger()

ORIGIN_TIMEOUT_RETURNCODE = -9999

# metrics.json (schema_version 2) fields aggregated across origins.
_AGGREGATED_METRICS = (
    ("mae", ("point_metrics", "mae")),
    ("rmse", ("point_metrics", "rmse")),
    ("r2", ("point_metrics", "r2")),
    ("crps", ("crps", "mean_crps")),
    ("coverage_0.80", ("calibration", "coverages", "0.80", "empirical")),
    ("coverage_0.95", ("calibration", "coverages", "0.95", "empirical")),
    ("wis", ("calibration", "wis")),
    ("elpd_per_obs", ("info_criteria", "elpd_per_obs")),
)


@dataclass
class OriginRecord:
    origin: int
    status: str = "pending"  # pending | completed | failed | timeout
    run_dir: str | None = None
    split_content_hash: str | None = None
    n_test: int | None = None
    n_entities: int | None = None
    metrics: dict[str, Any] | None = None
    wall_clock_seconds: float | None = None
    error: str | None = None


class BacktestLedger:
    """Checkpointed origin records; identity by origin index enables resume."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[int, OriginRecord] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("origins", []):
                record = OriginRecord(**entry)
                self.records[record.origin] = record

    def upsert(self, record: OriginRecord) -> None:
        self.records[record.origin] = record
        self.checkpoint()

    def checkpoint(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records = sorted(self.records.values(), key=lambda r: r.origin)
        payload = {"origins": [asdict(r) for r in records]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def completed_origins(self) -> set[int]:
        return {k for k, r in self.records.items() if r.status == "completed"}


@dataclass
class BacktestConfig:
    origins: int = 3
    backtest_id: str = "default"
    output_root: Path = Path("outputs/backtest")
    dataset: str | None = None
    num_chains: int | None = None
    num_samples: int | None = None
    num_warmup: int | None = None
    origin_timeout_seconds: float | None = None
    panelcast_bin: str | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)

    @property
    def backtest_dir(self) -> Path:
        return self.output_root / self.backtest_id


def _dig(payload: dict, path: tuple[str, ...]) -> float | None:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return float(node) if isinstance(node, (int, float)) else None


def _write_origin_config(cfg: BacktestConfig, origin: int, path: Path) -> None:
    payload: dict[str, Any] = {
        **cfg.extra_config,
        "origin_offset": origin,
        "stages": ["splits", "features", "train", "evaluate"],
    }
    if cfg.dataset is not None:
        payload["dataset"] = cfg.dataset
    for key, value in (
        ("num_chains", cfg.num_chains),
        ("num_samples", cfg.num_samples),
        ("num_warmup", cfg.num_warmup),
    ):
        if value is not None:
            payload[key] = value
    import yaml  # type: ignore[import-untyped]

    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _launch_origin(
    config_path: Path, panelcast_bin: str, timeout_seconds: float | None
) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [panelcast_bin, "run", "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return (
            ORIGIN_TIMEOUT_RETURNCODE,
            f"origin exceeded timeout of {timeout_seconds}s and was killed",
        )
    tail = (proc.stdout + "\n" + proc.stderr)[-4000:]
    return proc.returncode, tail


def _attributed_run_dir(launched_at: datetime, claimed: set[str]) -> tuple[Path | None, str | None]:
    """Resolve the just-finished run and verify it is this origin's own output."""
    from panelcast.paths import resolve_latest

    run_dir = resolve_latest()
    if run_dir is None:
        return None, "no latest run pointer after origin finished"
    run_dir = Path(run_dir).resolve()
    if str(run_dir) in claimed:
        return None, f"attribution failed: {run_dir} already belongs to another origin"
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        created = datetime.fromisoformat(str(manifest.get("created_at")))
    except (OSError, ValueError, TypeError):
        return None, f"attribution failed: unreadable manifest at {run_dir}"
    if created.tzinfo is not None:
        created = created.astimezone().replace(tzinfo=None)
    if created < launched_at - timedelta(seconds=1):
        return None, (
            f"attribution failed: run {run_dir} predates this origin's launch "
            "(stale or foreign latest pointer)"
        )
    claimed.add(str(run_dir))
    return run_dir, None


def _harvest_origin(record: OriginRecord, run_dir: Path) -> None:
    """Pull metrics, split hash, and population sizes out of the origin's run dir."""
    metrics_path = run_dir / "evaluation" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    record.metrics = {name: _dig(payload, path) for name, path in _AGGREGATED_METRICS}
    record.n_test = payload.get("n_test")

    split_manifest = run_dir / "data" / "splits" / "within_entity_temporal" / "manifest.json"
    if not split_manifest.exists():
        # Run-scoped layouts keep splits under the run dir; fall back to flat.
        split_manifest = Path("data/splits/within_entity_temporal/manifest.json")
    if split_manifest.exists():
        sm = json.loads(split_manifest.read_text(encoding="utf-8"))
        record.split_content_hash = sm.get("content_hash") or (sm.get("splits") or {}).get(
            "content_hash"
        )
        test_stats = (sm.get("splits") or {}).get("test") or {}
        record.n_entities = test_stats.get("unique_artists")


def aggregate_backtest(records: list[OriginRecord]) -> dict:
    """Mean, SE across origins, and min/max for every aggregated metric."""
    completed = [r for r in records if r.status == "completed" and r.metrics]
    metrics_agg: dict[str, dict] = {}
    for name, _ in _AGGREGATED_METRICS:
        values = [r.metrics[name] for r in completed if r.metrics.get(name) is not None]
        if not values:
            metrics_agg[name] = None
            continue
        arr = np.asarray(values, dtype=float)
        metrics_agg[name] = {
            "mean": float(arr.mean()),
            "se": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else None,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_origins": len(arr),
            "per_origin": {str(r.origin): r.metrics.get(name) for r in completed},
        }
    return {
        "n_origins_requested": len(records),
        "n_origins_completed": len(completed),
        "origins": [
            {
                "origin": r.origin,
                "status": r.status,
                "n_test": r.n_test,
                "n_entities": r.n_entities,
                "run_dir": r.run_dir,
                "split_content_hash": r.split_content_hash,
            }
            for r in sorted(records, key=lambda r: r.origin)
        ],
        "metrics": metrics_agg,
        "note": (
            "Origins hold out progressively earlier events; deeper origins "
            "shrink the eligible entity set, so per-origin populations are "
            "not identical and cross-origin variation includes that shift."
        ),
    }


def render_backtest_markdown(aggregate: dict) -> str:
    lines = [
        "# Rolling-origin backtest",
        "",
        f"Origins completed: {aggregate['n_origins_completed']}"
        f"/{aggregate['n_origins_requested']}",
        "",
        "| Origin | status | n_test | n_entities |",
        "| --- | --- | --- | --- |",
    ]
    for o in aggregate["origins"]:
        lines.append(f"| {o['origin']} | {o['status']} | {o['n_test']} | {o['n_entities']} |")
    lines += ["", "| Metric | mean | SE | min | max |", "| --- | --- | --- | --- | --- |"]
    for name, block in aggregate["metrics"].items():
        if block is None:
            lines.append(f"| {name} | — | — | — | — |")
            continue
        se = f"{block['se']:.4f}" if block["se"] is not None else "—"
        lines.append(
            f"| {name} | {block['mean']:.4f} | {se} "
            f"| {block['min']:.4f} | {block['max']:.4f} |"
        )
    lines += ["", aggregate["note"], ""]
    return "\n".join(lines)


def run_backtest(cfg: BacktestConfig) -> dict:
    """Execute (or resume) the backtest and write aggregate artifacts."""
    from panelcast.select.runner import _default_panelcast_bin

    backtest_dir = cfg.backtest_dir
    backtest_dir.mkdir(parents=True, exist_ok=True)
    ledger = BacktestLedger(backtest_dir / "ledger.json")
    panelcast_bin = cfg.panelcast_bin or _default_panelcast_bin()
    claimed = {
        r.run_dir for r in ledger.records.values() if r.run_dir and r.status == "completed"
    }

    for origin in range(cfg.origins):
        existing = ledger.records.get(origin)
        if existing is not None and existing.status == "completed":
            log.info("backtest_origin_skipped", origin=origin, reason="already completed")
            continue
        record = OriginRecord(origin=origin, status="pending")
        ledger.upsert(record)

        config_path = backtest_dir / f"origin_{origin}.yaml"
        _write_origin_config(cfg, origin, config_path)
        log.info("backtest_origin_start", origin=origin, config=str(config_path))
        launched_at = datetime.now()
        started = time.monotonic()
        returncode, tail = _launch_origin(
            config_path, panelcast_bin, cfg.origin_timeout_seconds
        )
        record.wall_clock_seconds = round(time.monotonic() - started, 1)

        if returncode == ORIGIN_TIMEOUT_RETURNCODE:
            record.status = "timeout"
            record.error = tail
            ledger.upsert(record)
            continue
        if returncode != 0:
            record.status = "failed"
            record.error = tail[-1500:]
            ledger.upsert(record)
            continue

        run_dir, problem = _attributed_run_dir(launched_at, claimed)
        if problem is not None:
            record.status = "failed"
            record.error = problem
            ledger.upsert(record)
            continue
        record.run_dir = str(run_dir)
        try:
            _harvest_origin(record, run_dir)
            record.status = "completed"
        except (OSError, ValueError, KeyError) as e:
            record.status = "failed"
            record.error = f"harvest failed: {type(e).__name__}: {e}"
        ledger.upsert(record)
        log.info(
            "backtest_origin_done",
            origin=origin,
            status=record.status,
            wall_clock_seconds=record.wall_clock_seconds,
        )

    aggregate = aggregate_backtest(list(ledger.records.values()))
    (backtest_dir / "backtest_metrics.json").write_text(
        json.dumps(aggregate, indent=2, default=str), encoding="utf-8"
    )
    (backtest_dir / "backtest_report.md").write_text(
        render_backtest_markdown(aggregate), encoding="utf-8"
    )
    log.info(
        "backtest_complete",
        completed=aggregate["n_origins_completed"],
        requested=aggregate["n_origins_requested"],
        dir=str(backtest_dir),
    )
    return aggregate
