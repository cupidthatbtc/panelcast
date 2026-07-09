"""Staged, budget-aware, resumable sweep runner for `panelcast select` (#100, A2).

Every arm is one orchestrated pipeline run (run-scoped products, per-point
log-likelihood persisted). Execution is STRICTLY SERIAL: ``data/features`` is a
flat cross-run cache, so arms that vary feature-affecting knobs rebuild
splits+features before fitting, and the stage stamps fail fast if anything else
touches the repo mid-sweep. The repo belongs to the sweep for its duration.

Stages: (1) one-factor-at-a-time from the shipped defaults; (2) compose
stage-1 winners and probe their pairwise interactions; (3) an optional random
sample of untried combinations. Cheap data diagnostics REORDER the search —
they never prune. Stage order is the budget-priority order; an arm whose
predicted cost exceeds the remaining budget is recorded as ``skipped_budget``
(retryable under a bigger budget) rather than truncating the stage.

Stage 1 optionally runs as a successive-halving rung ladder (#164): every OFAT
arm screens at a cheap pre-registered sampler scale, the top keep-fraction by
paired-ELPD z (plus near-threshold margin rescues) promotes to the next rung,
and only final-rung fits feed the report. True ASHA — early-stopping a fit on
intermediate ELPD — is not possible here: fits run to completion inside their
subprocess and no intermediate pointwise log-likelihood exists, so rung =
sampler scale is the version of early stopping this architecture supports.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random as _random
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
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

# An arm killed for exceeding its per-arm timeout returns this sentinel instead
# of a real exit code, so resume can tell a budget-kill (terminal — never retry)
# apart from an ordinary crash. Deliberately out of the range of real signals.
ARM_TIMEOUT_RETURNCODE = -1000

# Stage 2 composes the stage-1 winner set (1 + C(n, 2) arms), so the winner set
# is capped to keep the fit count within the consented plan's stage-2 ceiling.
STAGE2_MAX_WINNERS = 3

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
    # Successive-halving ladder for stage 1 (#164): a pre-registered list of
    # sampler-scale dicts (num_chains/num_samples/num_warmup + keep_fraction on
    # every rung but the last). None = single-scale legacy sweep. True ASHA
    # (early-stopping a fit on intermediate ELPD) is out of reach here — fits
    # run to completion inside their subprocess and no intermediate pointwise
    # log-likelihood exists; rung = sampler scale is the version of early
    # stopping this architecture supports.
    rungs: list[dict[str, Any]] | None = None
    # Rescue margin: arms with z >= winner_z - screen_margin promote regardless
    # of keep_fraction (pre-registered in configs/select.yaml rules).
    screen_margin: float = 0.5
    # Where arm subprocesses write their runs (#167): each arm's run dir is
    # named up front (run_id in its config), so attribution never depends on
    # the mutable `outputs/latest.json` pointer — the concurrency prerequisite.
    pipeline_output_base: Path = Path("outputs")
    extra_config: dict[str, Any] = field(default_factory=dict)
    panelcast_bin: str | None = None
    # Per-arm kill threshold: seconds, or "auto" to size each arm's timeout from
    # its predicted runtime (a fixed timeout structurally kills every arm that
    # retains the ~10x-cost offset_logit transform, #138). The multiplier
    # absorbs non-train stages and prediction error; the floor keeps sparse
    # history from producing hair-trigger kills.
    arm_timeout_seconds: float | str | None = None
    arm_timeout_multiplier: float = 3.0
    arm_timeout_floor_seconds: float = 1800.0
    # Warmup transfer (#178): arms reuse the reference fit's adapted mass matrix
    # (exact latent-signature match only) at a reduced warmup. Screening-grade —
    # confirmation always runs cold, so no promoted champion's evidence depends
    # on transferred adaptation.
    warmup_transfer: bool = False
    warmup_transfer_num_warmup: int = 200

    def __post_init__(self) -> None:
        if self.rungs and not self.reference_first:
            raise ValueError(
                "a rung ladder requires reference_first=True: rung-0 arms score "
                "against the rung-0 reference, so without it nothing promotes and "
                "the ladder silently collapses."
            )

    @property
    def sweep_dir(self) -> Path:
        return self.output_root / self.sweep_id


@dataclass
class ArmRecord:
    """Ledger entry for one arm; the knob-dict hash is its identity."""

    arm_id: str
    knobs: dict[str, Any]
    stage: int
    status: str = "pending"  # pending | completed | failed | timeout | skipped_budget | excluded
    run_dir: str | None = None
    wall_clock_seconds: float | None = None
    error: str | None = None
    score: dict[str, Any] | None = None
    note: str | None = None
    # Optional with defaults so 0.7.x-era ledgers (without these keys) still load.
    predicted_seconds: float | None = None
    timeout_seconds_used: float | None = None
    warm_started: bool | None = None
    # Ladder rung the record was fit at (#164); v1 ledgers load as rung 0.
    rung: int = 0


def arm_id(knobs: dict[str, Any]) -> str:
    """Stable identity of an arm: hash of the FULL merged knob dict."""
    merged = {**default_arm(), **knobs}
    payload = json.dumps(merged, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def record_key(aid: str, rung: int) -> str:
    """Ledger key: the knob hash, rung-suffixed above rung 0 so a refit at a
    higher rung never collides with its screening record (v1 keys unchanged)."""
    return f"{aid}@r{rung}" if rung else aid


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
    cost: Callable[[dict[str, Any]], float] | None = None,
) -> list[tuple[dict[str, Any], str | None]]:
    """Diagnosed-relevant knobs float to the front; predicted cost breaks ties.

    Diagnostics dominate (they reorder, never prune); within each priority
    group cheaper arms run first, so a budget-capped sweep completes as many
    arms as the budget allows instead of dying on the first expensive one.
    """
    prioritized: list[str] = []
    for signal, knob_names in _DIAGNOSTIC_PRIORITIES.items():
        if diagnostics.get(signal):
            prioritized.extend(knob_names)

    def rank(pair: tuple[dict[str, Any], str | None]) -> tuple[int, float]:
        arm = pair[0]
        priority = len(prioritized)
        for i, name in enumerate(prioritized):
            if name in arm:
                priority = i
                break
        return priority, cost(arm) if cost is not None else 0.0

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


def stage2_winners(
    records: Iterable[ArmRecord],
    winner_z: float,
    cap: int = STAGE2_MAX_WINNERS,
    rung: int = 0,
) -> list[dict[str, Any]]:
    """Stage-1 winners for stage 2: top-``cap`` by z, tie-broken by arm id.

    A None z (unscored — missing/failed reference snapshot) is never a winner.
    The cap keeps the composed+pairwise stage-2 count within the plan's
    ``1 + C(cap, 2)`` ceiling. Only records fit at ``rung`` (the ladder's final
    rung) count — screening-scale z's never feed stage 2.
    """
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for r in records:
        if r.stage != 1 or not r.knobs or r.status != "completed" or r.rung != rung:
            continue
        z = (r.score or {}).get("z")
        if z is not None and z >= winner_z:
            scored.append((float(z), r.arm_id, r.knobs))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [knobs for _, _, knobs in scored[:cap]]


def rung_survivors(
    records: Iterable[ArmRecord],
    rung: int,
    keep_fraction: float,
    promote_z: float,
    screen_margin: float,
) -> list[tuple[dict[str, Any], str | None]]:
    """Arms promoted from a screening rung (#164), pre-registered rule:
    the top ``keep_fraction`` of scored arms by z, PLUS any arm whose z lands
    within ``screen_margin`` of ``promote_z`` — screening noise must never
    eliminate a near-threshold arm. Unscored arms carry no evidence and are
    never promoted (the stage2_winners convention)."""
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for r in records:
        if r.stage != 1 or not r.knobs or r.status != "completed" or r.rung != rung:
            continue
        z = (r.score or {}).get("z")
        if z is not None:
            scored.append((float(z), r.arm_id, r.knobs))
    if not scored:
        return []
    scored.sort(key=lambda t: (-t[0], t[1]))
    n_keep = max(1, math.ceil(keep_fraction * len(scored)))
    rescue_bar = promote_z - screen_margin
    survivors = []
    for i, (z, _aid, knobs) in enumerate(scored):
        if i < n_keep:
            survivors.append((knobs, f"promoted from rung {rung} (z={z:+.2f})"))
        elif z >= rescue_bar:
            survivors.append(
                (knobs, f"margin-rescued from rung {rung} (z={z:+.2f} ≥ {rescue_bar:+.2f})")
            )
    return survivors


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
    """Checkpointed arm records; identity by knob-dict hash enables --resume.

    Payload v2 (#164) adds a version field and rung-suffixed keys; v1 ledgers
    (no version, no ``rung`` on records) load as rung-0 records with their
    original keys, so old sweeps still resume.
    """

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, ArmRecord] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("arms", []):
                record = ArmRecord(**entry)
                self.records[record_key(record.arm_id, record.rung)] = record

    def upsert(self, record: ArmRecord) -> None:
        self.records[record_key(record.arm_id, record.rung)] = record
        self.checkpoint()

    def checkpoint(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 2, "arms": [asdict(r) for r in self.records.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def completed_ids(self) -> set[str]:
        # Bare arm ids, not the rung-suffixed record keys.
        return {r.arm_id for r in self.records.values() if r.status == "completed"}

    def fits_done(self) -> int:
        return sum(
            1 for r in self.records.values() if r.status in ("completed", "failed", "timeout")
        )

    def hours_spent(self) -> float:
        return sum(r.wall_clock_seconds or 0.0 for r in self.records.values()) / 3600.0


def _default_panelcast_bin() -> str:
    sibling = Path(sys.executable).with_name("panelcast")
    return str(sibling) if sibling.exists() else "panelcast"


def _write_arm_config(
    cfg: SweepConfig,
    merged: dict[str, Any],
    stages: list[str],
    path: Path,
    extra: dict[str, Any] | None = None,
    sampler_overrides: dict[str, int] | None = None,
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
    # Rung scale (the confirmation._write_config pattern); warmup-transfer's
    # reduced num_warmup in `extra` still wins, as before.
    payload.update(sampler_overrides or {})
    payload.update(extra or {})
    import yaml  # type: ignore[import-untyped]

    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _warmup_transfer_extra(cfg: SweepConfig, arm: dict[str, Any], record: ArmRecord) -> dict:
    """Per-arm warmup-transfer config keys: reference exports, later arms import.

    The import runs at the reduced warmup; a signature mismatch inside the fit
    misses cleanly to a cold fit at that same reduced warmup, which the
    divergence/Rhat gates then judge — never silently biased.
    """
    if not cfg.warmup_transfer:
        return {}
    export_path = cfg.sweep_dir / "warmup_reference.pkl"
    if not arm:
        return {"warmup_export_path": str(export_path)}
    if export_path.exists():
        record.warm_started = True
        return {
            "warmup_import_path": str(export_path),
            "num_warmup": cfg.warmup_transfer_num_warmup,
        }
    return {}


def _predict_arm_seconds(
    cfg: SweepConfig,
    merged: dict[str, Any],
    dims: dict[str, int] | None,
    sampler: dict[str, Any] | None = None,
) -> Any:
    """RuntimePrediction for one arm at the scale its subprocess will run, or None.

    ``sampler`` carries a rung's overrides so cheap-rung arms are predicted
    (and therefore auto-timed-out) at their own scale, not the tier's.
    """
    if dims is None:
        return None
    from panelcast.gpu_memory.runtime_predictor import predict_fit_seconds
    from panelcast.pipelines.orchestrator import PipelineConfig

    defaults = PipelineConfig()
    s = sampler or {}
    return predict_fit_seconds(
        s.get("num_chains") or cfg.num_chains or defaults.num_chains,
        s.get("num_samples") or cfg.num_samples or defaults.num_samples,
        s.get("num_warmup") or cfg.num_warmup or defaults.num_warmup,
        int(dims.get("n_observations") or 0),
        transform=merged.get("target_transform") or defaults.target_transform,
    )


def resolve_arm_timeout(
    cfg: SweepConfig,
    merged: dict[str, Any],
    dims: dict[str, int] | None = None,
    sampler: dict[str, Any] | None = None,
) -> tuple[float | None, Any]:
    """One arm's kill threshold; returns (timeout_seconds, RuntimePrediction | None).

    An explicit numeric ``arm_timeout_seconds`` passes through untouched
    (reproducibility: a number the user typed always wins). ``"auto"`` predicts
    the arm's own runtime — transform-aware, at the sampler scale the arm
    subprocess will actually run (a rung's ``sampler`` overrides included) —
    and uses max(floor, multiplier * predicted). Without data dims to predict
    from, auto falls back to the floor.
    """
    if cfg.arm_timeout_seconds is None:
        return None, None
    if cfg.arm_timeout_seconds != "auto":
        return float(cfg.arm_timeout_seconds), None
    prediction = _predict_arm_seconds(cfg, merged, dims, sampler)
    if prediction is None:
        return cfg.arm_timeout_floor_seconds, None
    timeout = max(cfg.arm_timeout_floor_seconds, cfg.arm_timeout_multiplier * prediction.seconds)
    return timeout, prediction


def launch_arm(
    config_path: Path,
    panelcast_bin: str,
    timeout_seconds: float | None = None,
    env_overrides: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run one arm as a subprocess; returns (returncode, combined output tail).

    A fit that exceeds ``timeout_seconds`` is killed (subprocess.run reaps the
    child before raising) and reported as a failure, so one pathological arm
    can't stall the whole serial sweep. ``env_overrides`` lets a concurrent
    caller bound each child's JAX pool (XLA_PYTHON_CLIENT_MEM_FRACTION, #167);
    None keeps the legacy child environment exactly.
    """
    import os

    env = {**os.environ, "PANELCAST_SAVE_LOG_LIKELIHOOD": "1", "PANELCAST_SAVE_PREDICTIVE": "1"}
    env.update(env_overrides or {})
    try:
        proc = subprocess.run(
            [panelcast_bin, "run", "--config", str(config_path)],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return ARM_TIMEOUT_RETURNCODE, f"arm exceeded timeout of {timeout_seconds}s and was killed"
    tail = (proc.stdout + "\n" + proc.stderr)[-4000:]
    return proc.returncode, tail


def _attribution_error(
    run_dir: Path | None,
    merged: dict[str, Any],
    launched_at: datetime,
    claimed_runs: set[str],
) -> str | None:
    """Why the resolved run cannot be this arm's own output (None = attribution holds).

    ``outputs/latest.json`` is a mutable pointer shared with any concurrent
    ``panelcast run``, and a failed pointer write silently leaves the previous
    run resolved. Before scoring, check the run's manifest: it must be created
    after this arm launched, record this arm's knob values (when it records
    them), and not already belong to another arm of this sweep.
    """
    if run_dir is None:
        return None
    if str(run_dir) in claimed_runs:
        return f"attribution failed: {run_dir} already belongs to another arm of this sweep"
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"attribution failed: no readable manifest at {run_dir}"
    try:
        created = datetime.fromisoformat(str(manifest.get("created_at")))
    except (TypeError, ValueError):
        return f"attribution failed: unreadable created_at in {run_dir}/manifest.json"
    if created.tzinfo is not None:
        created = created.astimezone().replace(tzinfo=None)
    if created < launched_at - timedelta(seconds=1):
        return (
            f"attribution failed: run {run_dir} started at {created.isoformat()}, "
            "before this arm launched (stale or foreign latest pointer)"
        )
    flags = manifest.get("flags")
    if isinstance(flags, dict):
        mismatched = sorted(k for k, v in merged.items() if k in flags and flags[k] != v)
        if mismatched:
            return "attribution failed: run config disagrees with the arm on " + ", ".join(
                mismatched
            )
    return None


def _claim_named_run(
    run_id: str,
    output_base: Path,
    merged: dict[str, Any],
    launched_at: datetime,
    claimed_runs: set[str],
) -> tuple[Path | None, str | None]:
    """Claim the run dir this arm NAMED up front (#167 handshake).

    The arm's config carries its run_id, so the run dir is known before the
    subprocess starts — no dependence on the mutable ``outputs/latest.json``
    pointer that races under concurrency. The manifest sanity checks
    (creation time, knob agreement, prior claim) still apply.
    """
    run_dir = (output_base / run_id).resolve()
    if not run_dir.exists():
        return None, f"handshake failed: expected run dir {run_dir} was never created"
    problem = _attribution_error(run_dir, merged, launched_at, claimed_runs)
    if problem is None:
        claimed_runs.add(str(run_dir))
    return run_dir, problem


def _apply_arm_timeout(
    cfg: SweepConfig,
    merged: dict[str, Any],
    dims: dict[str, int] | None,
    record: ArmRecord,
    sampler: dict[str, Any] | None = None,
) -> float | None:
    """Resolve one arm's timeout onto its record; auto predictions are logged."""
    timeout_seconds, prediction = resolve_arm_timeout(cfg, merged, dims, sampler)
    record.timeout_seconds_used = timeout_seconds
    if prediction is not None:
        record.predicted_seconds = round(prediction.seconds, 1)
        log.info(
            "arm_timeout_auto",
            arm_id=record.arm_id,
            predicted_seconds=record.predicted_seconds,
            timeout_seconds=timeout_seconds,
            source=prediction.source,
        )
    return timeout_seconds


def _cost_fn(
    cfg: SweepConfig,
    base: dict[str, Any],
    dims: dict[str, int] | None,
    sampler: dict[str, Any] | None = None,
) -> Callable[[dict[str, Any]], float] | None:
    """Predicted-seconds tiebreak for ``reorder_arms``; None without data dims."""
    if dims is None:
        return None

    def cost(arm: dict[str, Any]) -> float:
        prediction = _predict_arm_seconds(cfg, {**base, **arm}, dims, sampler)
        return prediction.seconds if prediction is not None else 0.0

    return cost


def _budget_skip_reason(cfg: SweepConfig, ledger: SweepLedger, prediction: Any) -> str | None:
    """Why this arm cannot fit the remaining budget (None = launch it)."""
    if cfg.budget_hours is None:
        return None
    remaining = cfg.budget_hours - ledger.hours_spent()
    predicted_hours = prediction.seconds / 3600.0 if prediction is not None else None
    if remaining > 0 and (predicted_hours is None or predicted_hours <= remaining):
        return None
    if remaining <= 0:
        return f"budget exhausted: {ledger.hours_spent():.2f}h spent of {cfg.budget_hours:g}h"
    return f"predicted {predicted_hours:.2f}h exceeds remaining budget {remaining:.2f}h"


def _maybe_skip_for_budget(
    cfg: SweepConfig,
    ledger: SweepLedger,
    dims: dict[str, int] | None,
    stage: int,
    arm: dict[str, Any],
    aid: str,
    note: str | None,
    rung: int = 0,
    sampler: dict[str, Any] | None = None,
) -> bool:
    """Record a retryable ``skipped_budget`` when the arm can't fit the remaining budget.

    Never clobbers a real failure record — a failed arm skipped for budget keeps
    its error; either way both statuses are non-terminal on resume.
    """
    if cfg.budget_hours is None:
        return False
    prediction = _predict_arm_seconds(cfg, {**default_arm(), **arm}, dims, sampler)
    reason = _budget_skip_reason(cfg, ledger, prediction)
    if reason is None:
        return False
    log.warning("arm_skipped_budget", arm_id=aid, stage=stage, reason=reason)
    existing = ledger.records.get(record_key(aid, rung))
    if existing is None or existing.status in ("pending", "skipped_budget"):
        ledger.upsert(
            ArmRecord(
                arm_id=aid, knobs=arm, stage=stage, status="skipped_budget", note=note,
                error=reason, rung=rung,
                predicted_seconds=round(prediction.seconds, 1) if prediction is not None else None,
            )
        )
    return True


def _record_launch_failure(record: ArmRecord, code: int, tail: str) -> None:
    if code == ARM_TIMEOUT_RETURNCODE:
        record.status = "timeout"
        log.warning("arm_timeout", arm_id=record.arm_id, returncode=code)
    else:
        record.status = "failed"
        log.warning("arm_failed", arm_id=record.arm_id, returncode=code)
    record.error = tail[-1500:]


def _run_stage1_ladder(
    cfg: SweepConfig,
    ledger: SweepLedger,
    stage1: list[tuple[dict[str, Any], str | None]],
    rungs: list[dict[str, Any] | None],
    execute: Callable[..., None],
    max_fits_reached: Callable[[], str | None],
) -> bool:
    """Stage 1 across the ladder rungs; False when truncated by max_fits."""
    final_rung = len(rungs) - 1
    survivors: list[tuple[dict[str, Any], str | None]] = list(stage1)
    for rung_idx in range(len(rungs)):
        plan: list[tuple[dict[str, Any], str | None]] = []
        if cfg.reference_first or rung_idx > 0:
            label = (
                "reference (shipped defaults)"
                if rung_idx == 0
                else f"reference refit at rung {rung_idx}"
            )
            plan.append(({}, label))
        plan.extend(survivors)
        for arm, note in plan:
            reason = max_fits_reached()
            if reason:
                log.warning("sweep_truncated", stage=1, rung=rung_idx, reason=reason)
                return False
            execute(1, arm, note, rung_idx)
        if rung_idx < final_rung:
            keep = float((rungs[rung_idx] or {}).get("keep_fraction") or 1.0)
            survivors = rung_survivors(
                ledger.records.values(), rung_idx, keep, cfg.winner_z, cfg.screen_margin
            )
            log.info(
                "rung_promotion",
                rung=rung_idx,
                n_promoted=len(survivors),
                keep_fraction=keep,
                screen_margin=cfg.screen_margin,
            )
            if not survivors:
                log.warning("rung_ladder_empty", rung=rung_idx)
                break
    return True


def run_sweep(
    cfg: SweepConfig,
    descriptor: DatasetDescriptor,
    train_df=None,
    available_columns: frozenset[str] | None = None,
    launch: Callable[..., tuple[int, str]] | None = None,
    scorer: Callable[[Path, Path | None], dict[str, Any]] | None = None,
    dims: dict[str, int] | None = None,
) -> SweepLedger:
    """Execute the staged sweep serially; every arm checkpoints the ledger.

    ``launch`` and ``scorer`` are injectable for tests; the defaults run the
    real pipeline subprocess and (when the scoring module is present) the
    paired-ELPD scorer against the reference arm's snapshot. ``dims`` are the
    data dimensions the orchestrator already resolved from the prepared
    feature matrix; the "auto" arm timeout predicts from them per arm.
    """
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

    # The stage-1 ladder (#164): rung 0 screens every OFAT arm cheap, later
    # rungs refit the pre-registered survivors (+ the reference, so pairing
    # always compares same-scale fits). [None] = single-scale legacy.
    rungs: list[dict[str, Any] | None] = list(cfg.rungs) if cfg.rungs else [None]
    final_rung = len(rungs) - 1

    def _sampler(rung_idx: int) -> dict[str, Any] | None:
        r = rungs[rung_idx]
        if r is None:
            return None
        return {k: r[k] for k in ("num_chains", "num_samples", "num_warmup") if k in r}

    stage1 = reorder_arms(
        ofat_arms(descriptor, available_columns),
        diagnostics,
        _cost_fn(cfg, base, dims, _sampler(0)),
    )

    reference_runs: dict[int, Path] = {}
    cache_signature: str | None = None
    claimed_runs = {str(r.run_dir) for r in ledger.records.values() if r.run_dir}

    def _max_fits_reached() -> str | None:
        if cfg.max_fits is not None and ledger.fits_done() >= cfg.max_fits:
            return f"max_fits={cfg.max_fits} reached"
        return None

    def _execute(stage: int, arm: dict[str, Any], note: str | None, rung: int) -> None:
        nonlocal cache_signature
        aid = arm_id(arm)
        existing = ledger.records.get(record_key(aid, rung))
        if existing and existing.status in ("completed", "timeout"):
            # A timed-out arm is terminal too: skip it instead of re-running an arm
            # that will just time out again. It carries no run_dir, so the reference
            # bookkeeping below passes it by; the status field keeps the skip visible.
            if not arm and existing.run_dir:
                reference_runs[rung] = Path(existing.run_dir)
            log.info("arm_skipped_resume", arm_id=aid, rung=rung, status=existing.status)
            return
        merged = {**base, **arm}
        sampler = _sampler(rung)
        if _maybe_skip_for_budget(cfg, ledger, dims, stage, arm, aid, note, rung, sampler):
            return
        signature = feature_signature(merged)
        stages = _FEATURE_STAGES if signature != cache_signature else _MODEL_STAGES
        record = ArmRecord(arm_id=aid, knobs=arm, stage=stage, note=note, rung=rung)
        config_path = cfg.sweep_dir / f"arm_{record_key(aid, rung)}.yaml"
        launched_at = datetime.now()
        # Name the run up front (#167): unique per launch attempt, so a
        # re-run of a failed arm never collides with its earlier dir.
        run_id = f"sel_{cfg.sweep_id}_{record_key(aid, rung)}_{launched_at:%Y%m%dT%H%M%S%f}"
        _write_arm_config(
            cfg, merged, stages, config_path,
            extra={**_warmup_transfer_extra(cfg, arm, record), "run_id": run_id},
            sampler_overrides=sampler,
        )
        timeout_seconds = _apply_arm_timeout(cfg, merged, dims, record, sampler)
        log.info("arm_start", arm_id=aid, stage=stage, rung=rung, knobs=arm, stages=stages)
        started = time.monotonic()
        code, tail = launch(config_path, panelcast_bin, timeout_seconds)
        record.wall_clock_seconds = time.monotonic() - started
        if code != 0:
            _record_launch_failure(record, code, tail)
            # The killed/failed run may have half-rebuilt the flat caches; force
            # the next arm to rebuild rather than trust an unknown state.
            cache_signature = None
            ledger.upsert(record)
            return
        run_dir, problem = _claim_named_run(
            run_id, cfg.pipeline_output_base, merged, launched_at, claimed_runs
        )
        if problem:
            record.status = "failed"
            record.error = problem
            log.warning("arm_attribution_failed", arm_id=aid, error=problem)
            cache_signature = None
            ledger.upsert(record)
            return
        cache_signature = signature
        record.run_dir = str(run_dir) if run_dir else None
        if not arm and run_dir is not None:
            reference_runs[rung] = run_dir
        if scorer is not None and run_dir is not None:
            try:
                record.score = scorer(run_dir, reference_runs.get(rung))
            except Exception as exc:  # scoring must never kill the sweep
                record.note = f"{record.note or ''}; scoring failed: {exc}".strip("; ")
        record.status = "completed"
        ledger.upsert(record)

    if not _run_stage1_ladder(cfg, ledger, stage1, rungs, _execute, _max_fits_reached):
        return ledger

    if cfg.include_stage2:
        winners = stage2_winners(ledger.records.values(), cfg.winner_z, rung=final_rung)
        seen = {r.arm_id for r in ledger.records.values()}
        for arm, note in stage2_arms(winners, descriptor, available_columns, seen):
            reason = _max_fits_reached()
            if reason:
                log.warning("sweep_truncated", stage=2, reason=reason)
                return ledger
            _execute(2, arm, note, final_rung)

    if cfg.stage3_fits > 0:
        seen = {r.arm_id for r in ledger.records.values()}
        for arm, note in stage3_arms(
            descriptor, cfg.stage3_fits, cfg.sweep_id, available_columns, seen
        ):
            reason = _max_fits_reached()
            if reason:
                log.warning("sweep_truncated", stage=3, reason=reason)
                return ledger
            _execute(3, arm, note, final_rung)

    return ledger
