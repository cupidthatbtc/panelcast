"""Staged, budget-aware, resumable sweep runner for `panelcast select` (#100, A2).

Every arm is one orchestrated pipeline run (run-scoped products, per-point
log-likelihood persisted). Execution is STRICTLY SERIAL: ``data/features`` is a
flat cross-run cache, so arms that vary feature-affecting knobs rebuild
splits+features before fitting, and the stage stamps fail fast if anything else
touches the repo mid-sweep. The repo belongs to the sweep for its duration.

Stages: (1) one-factor-at-a-time from the shipped defaults; (2) compose
stage-1 winners and probe their pairwise interactions; (3) an optional random
sample of untried combinations. Cheap data diagnostics REORDER the search —
they never prune. Stage order is the budget-truncation order.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random as _random
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.select.space import (
    KNOBS,
    arm_conflicts,
    default_arm,
    enumerate_space,
    knob_is_active,
)

log = structlog.get_logger()

_FEATURE_STAGES = ["splits", "features", "train", "evaluate"]
_MODEL_STAGES = ["train", "evaluate"]

# Diagnostics only reorder: each key maps to the knobs it argues for trying
# earlier. Transform × family is also the known-dangerous stage-2 interaction.
_DIAGNOSTIC_PRIORITIES = {
    "target_skewed": ("target_transform", "likelihood_family"),
    "integer_heaped": ("discretize_observation",),
    "sparse_histories": ("errors_in_variables", "entity_group_pooling"),
    "obs_count_spread": ("heteroscedastic_entity_obs", "learn_n_exponent"),
}


@dataclass
class SweepConfig:
    """Everything one sweep needs; serialized next to the ledger."""

    sweep_id: str
    dataset: str | None = None
    output_root: Path = Path("outputs/select")
    reference_first: bool = True
    max_fits: int | None = None
    budget_hours: float | None = None
    include_stage2: bool = True
    stage3_fits: int = 0
    winner_z: float = 2.0
    num_chains: int | None = None
    num_samples: int | None = None
    num_warmup: int | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)
    panelcast_bin: str | None = None
    arm_timeout_seconds: float | None = None

    @property
    def sweep_dir(self) -> Path:
        return self.output_root / self.sweep_id


@dataclass
class ArmRecord:
    """Ledger entry for one arm; the knob-dict hash is its identity."""

    arm_id: str
    knobs: dict[str, Any]
    stage: int
    status: str = "pending"  # pending | completed | failed | excluded
    run_dir: str | None = None
    wall_clock_seconds: float | None = None
    error: str | None = None
    score: dict[str, Any] | None = None
    note: str | None = None


def arm_id(knobs: dict[str, Any]) -> str:
    """Stable identity of an arm: hash of the FULL merged knob dict."""
    merged = {**default_arm(), **knobs}
    payload = json.dumps(merged, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def complete_arm(
    arm: dict[str, Any],
    descriptor: DatasetDescriptor,
    available_columns: frozenset[str] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Add the minimal companion change a structurally conflicted arm needs.

    OFAT wants each option tried alone, but some options cannot exist alone:
    a bounded family or the discretization toggle requires the identity
    transform. The completion is recorded on the arm so the report shows the
    option was tried with its structural companion, not silently dropped.
    Returns (arm, note) — or (arm, None) untouched; arms that stay conflicted
    after completion are the caller's to mark excluded.
    """
    conflicts = arm_conflicts(arm, descriptor, available_columns)
    if not conflicts:
        return arm, None
    needs_identity = any("target_transform='identity'" in c for c in conflicts)
    if needs_identity and "target_transform" not in arm:
        completed = {**arm, "target_transform": "identity"}
        if not arm_conflicts(completed, descriptor, available_columns):
            return completed, "completed with target_transform=identity (structural)"
    return arm, None


def ofat_arms(
    descriptor: DatasetDescriptor,
    available_columns: frozenset[str] | None = None,
) -> list[tuple[dict[str, Any], str | None]]:
    """Stage 1: every knob varied alone from the shipped-defaults base.

    Returns (arm, note) pairs; arms that remain structurally impossible after
    completion are excluded upstream by ``arm_conflicts`` and never returned.
    """
    base = default_arm()
    space = enumerate_space(descriptor, available_columns)
    arms: list[tuple[dict[str, Any], str | None]] = []
    for knob in KNOBS:
        if not knob_is_active(knob, base):
            # Inert alone (n_exponent_prior without learn_n_exponent): vary it
            # jointly with its enabler so the option is still tried.
            enabler = {"learn_n_exponent": True} if knob.name == "n_exponent_prior" else {}
            if not enabler:
                log.warning("inert_knob_without_enabler_mapping", knob=knob.name)
            for value in space[knob.name]:
                if value == base[knob.name] or not enabler:
                    continue
                arm = {**enabler, knob.name: value}
                if not arm_conflicts(arm, descriptor, available_columns):
                    arms.append((arm, f"paired with {next(iter(enabler))}=True (inert alone)"))
            continue
        for value in space[knob.name]:
            if value == base[knob.name]:
                continue
            arm, note = complete_arm({knob.name: value}, descriptor, available_columns)
            if arm_conflicts(arm, descriptor, available_columns):
                continue
            arms.append((arm, note))
    return arms


def diagnose_data(train_df, descriptor: DatasetDescriptor) -> dict[str, bool]:
    """Cheap data signals that reorder the search (never prune)."""
    y = train_df[descriptor.target_col].dropna()
    skew = 0.0
    if len(y) > 2 and float(y.std()) > 0:
        z = (y - float(y.mean())) / float(y.std())
        skew = float((z**3).mean())
    integer_fraction = float((y == y.round()).mean()) if len(y) else 0.0
    events_per_entity = train_df.groupby(descriptor.entity_col).size()
    n_obs_col = descriptor.n_obs_col
    spread = 0.0
    if n_obs_col in train_df.columns:
        counts = train_df[n_obs_col].dropna()
        if len(counts) and float(counts.mean()) > 0:
            spread = float(counts.std() / counts.mean())
    return {
        "target_skewed": abs(skew) > 1.0,
        "integer_heaped": integer_fraction > 0.9,
        "sparse_histories": float(events_per_entity.median()) < 3.0,
        "obs_count_spread": spread > 1.0,
        "skewness": skew,
    }


def reorder_arms(
    arms: list[tuple[dict[str, Any], str | None]],
    diagnostics: dict[str, Any],
) -> list[tuple[dict[str, Any], str | None]]:
    """Diagnosed-relevant knobs float to the front; relative order otherwise kept."""
    prioritized: list[str] = []
    for signal, knob_names in _DIAGNOSTIC_PRIORITIES.items():
        if diagnostics.get(signal):
            prioritized.extend(knob_names)

    def rank(pair: tuple[dict[str, Any], str | None]) -> int:
        arm = pair[0]
        for i, name in enumerate(prioritized):
            if name in arm:
                return i
        return len(prioritized)

    return sorted(arms, key=rank)


def stage2_arms(
    winners: list[dict[str, Any]],
    descriptor: DatasetDescriptor,
    available_columns: frozenset[str] | None = None,
    seen: set[str] | None = None,
) -> list[tuple[dict[str, Any], str | None]]:
    """Compose stage-1 winners and probe their pairwise interactions."""
    seen = set() if seen is None else seen
    arms: list[tuple[dict[str, Any], str | None]] = []

    def _add(arm: dict[str, Any], note: str) -> None:
        arm, completion = complete_arm(arm, descriptor, available_columns)
        if arm_conflicts(arm, descriptor, available_columns):
            return
        aid = arm_id(arm)
        if aid in seen:
            return
        seen.add(aid)
        arms.append((arm, f"{note}; {completion}" if completion else note))

    if len(winners) > 1:
        combined: dict[str, Any] = {}
        for w in winners:
            combined.update(w)
        _add(combined, "all stage-1 winners composed")
    for a, b in itertools.combinations(winners, 2):
        _add({**a, **b}, "pairwise winner interaction")
    return arms


def stage3_arms(
    descriptor: DatasetDescriptor,
    n_arms: int,
    seed_material: str,
    available_columns: frozenset[str] | None = None,
    seen: set[str] | None = None,
) -> list[tuple[dict[str, Any], str | None]]:
    """Random sample of untried combinations — the interaction backstop."""
    seen = set() if seen is None else seen
    space = enumerate_space(descriptor, available_columns)
    base = default_arm()
    rng = _random.Random(seed_material)
    arms: list[tuple[dict[str, Any], str | None]] = []
    attempts = 0
    while len(arms) < n_arms and attempts < n_arms * 50:
        attempts += 1
        arm = {}
        for knob in KNOBS:
            values = space[knob.name]
            if len(values) > 1 and rng.random() < 0.5:
                value = rng.choice([v for v in values if v != base[knob.name]])
                arm[knob.name] = value
        if not arm:
            continue
        arm, completion = complete_arm(arm, descriptor, available_columns)
        if arm_conflicts(arm, descriptor, available_columns):
            continue
        aid = arm_id(arm)
        if aid in seen:
            continue
        seen.add(aid)
        arms.append((arm, completion or "stage-3 random sample"))
    return arms


def feature_signature(merged: dict[str, Any]) -> str:
    """The slice of an arm that determines the flat feature cache's content."""
    keys = sorted(k.name for k in KNOBS if k.affects_features)
    return json.dumps({k: merged[k] for k in keys}, sort_keys=True, default=str)


class SweepLedger:
    """Checkpointed arm records; identity by knob-dict hash enables --resume."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, ArmRecord] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("arms", []):
                record = ArmRecord(**entry)
                self.records[record.arm_id] = record

    def upsert(self, record: ArmRecord) -> None:
        self.records[record.arm_id] = record
        self.checkpoint()

    def checkpoint(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"arms": [asdict(r) for r in self.records.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def completed_ids(self) -> set[str]:
        return {aid for aid, r in self.records.items() if r.status == "completed"}

    def fits_done(self) -> int:
        return sum(1 for r in self.records.values() if r.status in ("completed", "failed"))

    def hours_spent(self) -> float:
        return sum(r.wall_clock_seconds or 0.0 for r in self.records.values()) / 3600.0


def _default_panelcast_bin() -> str:
    sibling = Path(sys.executable).with_name("panelcast")
    return str(sibling) if sibling.exists() else "panelcast"


def _write_arm_config(
    cfg: SweepConfig, merged: dict[str, Any], stages: list[str], path: Path
) -> None:
    payload: dict[str, Any] = {**cfg.extra_config, **merged, "stages": stages}
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


def launch_arm(
    config_path: Path, panelcast_bin: str, timeout_seconds: float | None = None
) -> tuple[int, str]:
    """Run one arm as a subprocess; returns (returncode, combined output tail).

    A fit that exceeds ``timeout_seconds`` is killed (subprocess.run reaps the
    child before raising) and reported as a failure, so one pathological arm
    can't stall the whole serial sweep.
    """
    import os

    env = {**os.environ, "PANELCAST_SAVE_LOG_LIKELIHOOD": "1"}
    try:
        proc = subprocess.run(
            [panelcast_bin, "run", "--config", str(config_path)],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return -9, f"arm exceeded timeout of {timeout_seconds}s and was killed"
    tail = (proc.stdout + "\n" + proc.stderr)[-4000:]
    return proc.returncode, tail


def run_sweep(
    cfg: SweepConfig,
    descriptor: DatasetDescriptor,
    train_df=None,
    available_columns: frozenset[str] | None = None,
    launch: Callable[..., tuple[int, str]] | None = None,
    scorer: Callable[[Path, Path | None], dict[str, Any]] | None = None,
) -> SweepLedger:
    """Execute the staged sweep serially; every arm checkpoints the ledger.

    ``launch`` and ``scorer`` are injectable for tests; the defaults run the
    real pipeline subprocess and (when the scoring module is present) the
    paired-ELPD scorer against the reference arm's snapshot.
    """
    from panelcast.paths import resolve_latest

    launch = launch or launch_arm
    panelcast_bin = cfg.panelcast_bin or _default_panelcast_bin()
    ledger = SweepLedger(cfg.sweep_dir / "ledger.json")
    cfg.sweep_dir.mkdir(parents=True, exist_ok=True)
    (cfg.sweep_dir / "sweep_config.json").write_text(
        json.dumps({**asdict(cfg), "output_root": str(cfg.output_root)}, indent=2, default=str),
        encoding="utf-8",
    )

    base = default_arm()
    diagnostics: dict[str, Any] = {}
    if train_df is not None:
        diagnostics = diagnose_data(train_df, descriptor)
        (cfg.sweep_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )

    stage1 = reorder_arms(ofat_arms(descriptor, available_columns), diagnostics)
    plan: list[tuple[int, dict[str, Any], str | None]] = []
    if cfg.reference_first:
        plan.append((1, {}, "reference (shipped defaults)"))
    plan.extend((1, arm, note) for arm, note in stage1)

    reference_run: Path | None = None
    cache_signature: str | None = None

    def _budget_exhausted() -> str | None:
        if cfg.max_fits is not None and ledger.fits_done() >= cfg.max_fits:
            return f"max_fits={cfg.max_fits} reached"
        if cfg.budget_hours is not None and ledger.hours_spent() >= cfg.budget_hours:
            return f"budget_hours={cfg.budget_hours} reached"
        return None

    def _execute(stage: int, arm: dict[str, Any], note: str | None) -> None:
        nonlocal reference_run, cache_signature
        aid = arm_id(arm)
        existing = ledger.records.get(aid)
        if existing and existing.status == "completed":
            if not arm and existing.run_dir:
                reference_run = Path(existing.run_dir)
            log.info("arm_skipped_resume", arm_id=aid)
            return
        merged = {**base, **arm}
        signature = feature_signature(merged)
        stages = _FEATURE_STAGES if signature != cache_signature else _MODEL_STAGES
        record = ArmRecord(arm_id=aid, knobs=arm, stage=stage, note=note)
        config_path = cfg.sweep_dir / f"arm_{aid}.yaml"
        _write_arm_config(cfg, merged, stages, config_path)
        log.info("arm_start", arm_id=aid, stage=stage, knobs=arm, stages=stages)
        started = time.monotonic()
        code, tail = launch(config_path, panelcast_bin, cfg.arm_timeout_seconds)
        record.wall_clock_seconds = time.monotonic() - started
        if code != 0:
            record.status = "failed"
            record.error = tail[-1500:]
            # The failed run may have half-rebuilt the flat caches; force the
            # next arm to rebuild rather than trust an unknown state.
            cache_signature = None
            ledger.upsert(record)
            log.warning("arm_failed", arm_id=aid, returncode=code)
            return
        cache_signature = signature
        run_dir = resolve_latest()
        record.run_dir = str(run_dir) if run_dir else None
        if not arm and run_dir is not None:
            reference_run = run_dir
        if scorer is not None and run_dir is not None:
            try:
                record.score = scorer(run_dir, reference_run)
            except Exception as exc:  # scoring must never kill the sweep
                record.note = f"{record.note or ''}; scoring failed: {exc}".strip("; ")
        record.status = "completed"
        ledger.upsert(record)

    for stage, arm, note in plan:
        reason = _budget_exhausted()
        if reason:
            log.warning("sweep_truncated", stage=stage, reason=reason)
            return ledger
        _execute(stage, arm, note)

    if cfg.include_stage2:
        winners = [
            r.knobs
            for r in ledger.records.values()
            if r.stage == 1
            and r.knobs
            and r.status == "completed"
            and (r.score or {}).get("z", float("-inf")) >= cfg.winner_z
        ]
        seen = set(ledger.records)
        for arm, note in stage2_arms(winners, descriptor, available_columns, seen):
            reason = _budget_exhausted()
            if reason:
                log.warning("sweep_truncated", stage=2, reason=reason)
                return ledger
            _execute(2, arm, note)

    if cfg.stage3_fits > 0:
        seen = set(ledger.records)
        for arm, note in stage3_arms(
            descriptor, cfg.stage3_fits, cfg.sweep_id, available_columns, seen
        ):
            reason = _budget_exhausted()
            if reason:
                log.warning("sweep_truncated", stage=3, reason=reason)
                return ledger
            _execute(3, arm, note)

    return ledger
