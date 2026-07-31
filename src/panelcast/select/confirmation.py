"""Multi-seed confirmation for a sweep winner (#102, A4).

Productizes the manual #40 recipe (.audit/transform_latent_bakeoff/MULTISEED.md):
for each confirmation seed, fit the reference (shipped defaults) and the winner
on that seed, then pair their per-point held-out ELPD. The winner confirms only
when the direction holds on EVERY seed — a single-seed z is one draw from the
selection lottery. Seeds are feature-affecting, so every confirmation run
rebuilds the flat caches (strictly serial, like the sweep).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
import yaml  # type: ignore[import-untyped]

from panelcast.config.descriptor import load_descriptor
from panelcast.paths import RunPathError
from panelcast.select.runner import (
    SweepConfig,
    _default_panelcast_bin,
    launch_arm,
    refusal_detail,
    resolve_arm_timeout,
    sweep_run_dir,
)
from panelcast.select.scoring import PairedElpd, paired_elpd
from panelcast.select.space import default_arm

log = structlog.get_logger()

_CONFIRMATION_STAGES = ["splits", "features", "train", "evaluate"]


@dataclass
class SeedResult:
    seed: int
    reference_run: str | None = None
    winner_run: str | None = None
    elpd: dict[str, Any] | None = None
    winner_converged: bool | None = None
    error: str | None = None


@dataclass
class ConfirmationResult:
    """Per-seed paired verdicts plus the holds-on-every-seed conclusion.

    ``sampler``/``version``/``seeds_planned``/``dataset_descriptor_hash``/
    ``base_config_hash`` echo the protocol so a resumed confirmation can prove
    the stored ledger belongs to the same call, on the same data, from the same
    base configuration.
    """

    winner_knobs: dict[str, Any]
    seeds: list[SeedResult] = field(default_factory=list)
    promote_z: float = 2.0
    sampler: dict[str, Any] | None = None
    version: str | None = None
    seeds_planned: list[int] = field(default_factory=list)
    dataset_descriptor_hash: str | None = None
    base_config_hash: str | None = None

    @property
    def confirmed(self) -> bool:
        """Direction holds AND the winner converges on every measured seed.

        Confirmation is the publication-scale gate: a winner screened at reduced
        samples only earns a recommendation if, refit at 5000, it clears the
        pre-registered z on every seed and its rhat/ess gate passes there too.
        """
        measured = [s for s in self.seeds if s.elpd is not None]
        if len(measured) < len(self.seeds) or not measured:
            return False
        return all(
            s.elpd is not None
            and s.elpd.get("z") is not None
            and s.elpd["z"] >= self.promote_z
            and s.winner_converged is True
            for s in measured
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner_knobs": self.winner_knobs,
            "promote_z": self.promote_z,
            "confirmed": self.confirmed,
            "sampler": self.sampler,
            "version": self.version,
            "seeds_planned": self.seeds_planned,
            "dataset_descriptor_hash": self.dataset_descriptor_hash,
            "base_config_hash": self.base_config_hash,
            "seeds": [asdict(s) for s in self.seeds],
        }


def _sampler_echo(cfg: SweepConfig, sampler_overrides: dict[str, int] | None) -> dict[str, Any]:
    return {
        "num_chains": cfg.num_chains,
        "num_samples": cfg.num_samples,
        "num_warmup": cfg.num_warmup,
        "overrides": dict(sampler_overrides or {}),
    }


def _descriptor_hash(cfg: SweepConfig) -> str | None:
    """Hash of the descriptor these fits will resolve, or None if it won't load.

    Resolved from ``cfg.dataset`` rather than taken from the caller, because
    that is the reference ``_write_config`` puts in every confirmation config —
    the domain the fits actually run against. An unset dataset resolves the
    same default the fits will, so the identity holds the hash their manifests
    record either way. A descriptor that cannot be loaded is not a value: None
    would equal the None a manifest that recorded no hash produces, so it
    refuses reuse in ``_cached_run_mismatch`` rather than matching there.
    """
    try:
        return load_descriptor(cfg.dataset).descriptor_hash()
    except (OSError, ValueError, yaml.YAMLError):
        return None


def _base_config_payload(
    cfg: SweepConfig, sampler_overrides: dict[str, int] | None
) -> dict[str, Any]:
    """Every option a confirmation fit is launched with except the per-fit ones.

    Built by the same helper that writes the fit configs, minus what varies
    inside one confirmation (the arm's knobs, the seed, the run id), so an
    output-affecting option cannot reach the fits without moving the cache
    identity with it. This is the identity's half; what each *run* is checked
    against is its own arm's full payload, which is where an option an arm
    overrides gets the value that arm's manifest recorded.
    """
    return _fit_config_payload(cfg, {}, sampler_overrides)


def _canonical_hash(payload: dict[str, Any]) -> str:
    """SHA-256 of a payload's canonical JSON.

    ``default=str`` is a backstop rather than a coercion that matters:
    ``_write_config`` hands the same payload to ``yaml.safe_dump``, which
    refuses anything that is not a YAML scalar, so a value that can reach a fit
    at all is already JSON-shaped. It only keeps the identity from raising
    *before* that clearer failure.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _cached_run_mismatch(
    run_dir: Path,
    descriptor_hash: str | None,
    fit_config: dict[str, Any],
    discriminating: frozenset[str] = frozenset(),
) -> tuple[str, str | None] | None:
    """Why this run's manifest is not the fit it stands for; None = it is.

    Returns ``(reason, config key)`` — the key is None when what failed is not
    a config comparison, so a reader is never left guessing whether a value is
    prose or a knob name.

    The identity file proves what the *previous call* declared; only the run's
    own manifest proves what was fit. Both are needed: an identity file written
    by a version that recorded less, or copied alongside its snapshots, would
    otherwise carry a foreign run into this verdict. ``fit_config`` is the
    arm's own payload rather than the arm-free base, so the reference and the
    winner of a seed are told apart — swap the two run dirs and the knobs no
    longer agree with the manifests. A run that recorded no experiment identity
    cannot be tied to this one either way, so it refits: the paired z is the
    evidence a promotion is argued from, and an unprovable snapshot is not
    evidence.

    Only keys the manifest recorded are compared: a config knob it never
    mentions has no recorded value to disagree with. That rule needs a floor,
    or a manifest that records none of the keys telling the two runs apart
    passes with zero comparisons — removing a key would defeat the gate more
    cheaply than forging one. ``discriminating`` is the set that must be
    *present*: the seed, the knobs the winner differs from the reference by,
    and the dataset when one is set.
    """
    if descriptor_hash is None:
        return "this call could not resolve its own dataset descriptor", None
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "no readable manifest", None
    if manifest.get("success") is not True:
        return "manifest does not record a successful run", None
    identity = manifest.get("experiment_identity")
    if not isinstance(identity, dict) or not identity:
        return "manifest records no experiment identity", None
    recorded_hash = identity.get("descriptor_hash")
    if recorded_hash is None:
        return "manifest records no dataset descriptor hash", None
    if recorded_hash != descriptor_hash:
        return "the run was fit on another dataset", "dataset_descriptor_hash"
    recorded = identity.get("config_payload")
    if not isinstance(recorded, dict):
        return "manifest records no config payload", None
    missing = sorted(discriminating - set(recorded))
    if missing:
        return "manifest does not record the key that identifies this fit", missing[0]
    for key, value in fit_config.items():
        if key not in recorded:
            continue
        # The recorded value is JSON, so a tuple this call holds is a list there.
        expected = list(value) if isinstance(value, tuple) else value
        if recorded[key] != expected:
            return "the run was fit with another value", key
    return None


def _identity_changes(recorded: dict[str, Any], identity: dict[str, Any]) -> list[str]:
    """Identity keys whose stored value contradicts this call's.

    A descriptor hash the *ledger* never resolved is unknown rather than
    different: archiving on it would make a stale mount cost a full
    publication-scale refit on the healthy call that follows it. The mirror
    case — this call blind against a ledger that knows its hash — is not
    exempted, because it buys nothing: reuse is refused per run while the hash
    is unresolved, so the seeds refit and the per-seed checkpoint overwrites
    the ledger either way. Archiving at least keeps a copy of what it replaced.
    """
    return sorted(
        key
        for key, value in identity.items()
        if recorded.get(key) != value
        and not (key == "dataset_descriptor_hash" and recorded.get(key) is None)
    )


def _reusable_prior_seeds(
    out_path: Path,
    identity: dict[str, Any],
    descriptor_hash: str | None,
    fit_config: Callable[[str, int], dict[str, Any]],
    discriminating: frozenset[str],
) -> dict[int, SeedResult]:
    """Prior seeds whose snapshots survive, IF the stored protocol matches this call.

    Any identity mismatch (knobs, z bar, sampler scale, seeds tuple, version,
    dataset, base config) archives the old ledger and starts fresh — evidence is
    never mixed across protocols. Old-format files without the echo fields
    mismatch by construction. A seed whose runs survive the identity file still
    has both its run dirs checked against their own manifests, each against the
    arm and seed it stands for.
    """
    if not out_path.exists():
        return {}
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    changed = _identity_changes(payload, identity)
    if changed:
        archived = out_path.with_name(f"confirmation_{time.strftime('%Y%m%dT%H%M%S')}.json")
        out_path.replace(archived)
        log.warning("confirmation_protocol_changed", archived=str(archived), changed=changed)
        return {}
    reusable: dict[int, SeedResult] = {}
    for entry in payload.get("seeds", []):
        ref, win = entry.get("reference_run"), entry.get("winner_run")
        if not ref or not win or entry.get("error"):
            continue
        if not (
            (Path(ref) / "evaluation" / "log_likelihood.nc").exists()
            and (Path(win) / "evaluation" / "log_likelihood.nc").exists()
        ):
            continue
        seed = int(entry["seed"])
        rejected = [
            (label, mismatch)
            for label, run in (("reference", ref), ("winner", win))
            if (
                mismatch := _cached_run_mismatch(
                    Path(run), descriptor_hash, fit_config(label, seed), discriminating
                )
            )
        ]
        if rejected:
            for label, (reason, key) in rejected:
                log.warning(
                    "confirmation_cached_run_rejected",
                    seed=seed,
                    label=label,
                    reason=reason,
                    key=key,
                )
            continue
        reusable[seed] = SeedResult(seed=seed, reference_run=ref, winner_run=win)
    return reusable


def _score_cached_seed(cached: SeedResult) -> SeedResult | None:
    """Re-pair a prior seed from its persisted snapshots; None → refit it."""
    win, ref = Path(cached.winner_run), Path(cached.reference_run)
    try:
        pair = paired_elpd(
            win / "evaluation" / "log_likelihood.nc", ref / "evaluation" / "log_likelihood.nc"
        )
    except Exception:  # corrupt snapshot: refit rather than crash the resume
        return None
    cached.winner_converged = _run_converged(win)
    cached.elpd = {"diff": pair.diff, "dse": pair.dse, "z": pair.z, "n": pair.n}
    return cached


def _run_converged(run_dir: Path | None) -> bool:
    """Whether a fit's convergence gate passed; missing/unreadable → not converged."""
    if run_dir is None:
        return False
    try:
        payload = json.loads(
            (run_dir / "evaluation" / "diagnostics.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("passed") is True


def _fit_config_payload(
    cfg: SweepConfig,
    merged: dict[str, Any],
    sampler_overrides: dict[str, int] | None,
    seed: int | None = None,
) -> dict[str, Any]:
    """One confirmation fit's config, less the run id it is named by."""
    payload: dict[str, Any] = {
        **cfg.extra_config,
        **merged,
        "stages": _CONFIRMATION_STAGES,
    }
    if seed is not None:
        payload["seed"] = seed
    if cfg.dataset is not None:
        payload["dataset"] = cfg.dataset
    for key, value in (
        ("num_chains", cfg.num_chains),
        ("num_samples", cfg.num_samples),
        ("num_warmup", cfg.num_warmup),
    ):
        if value is not None:
            payload[key] = value
    payload.update(sampler_overrides or {})
    return payload


def _write_config(
    cfg: SweepConfig,
    merged: dict[str, Any],
    seed: int,
    path: Path,
    sampler_overrides: dict[str, int] | None = None,
    run_id: str | None = None,
) -> None:
    payload = _fit_config_payload(cfg, merged, sampler_overrides, seed)
    if run_id is not None:
        payload["run_id"] = run_id
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _confirmation_timeout(
    cfg: SweepConfig,
    sampler_overrides: dict[str, int] | None,
    winner_knobs: dict[str, Any] | None = None,
    dims: dict[str, int] | None = None,
) -> float | None:
    """Per-fit timeout for confirmation, scaled from the screening arm timeout.

    Confirmation may run at publication scale (5-10x the screening sampler):
    reusing the screening timeout would kill legitimate fits, while no timeout
    lets one hang stall `panelcast select` forever. The screening timeout is
    the floor. With ``arm_timeout_seconds="auto"`` the screening base is the
    larger of the reference arm's and the winner arm's resolved auto timeouts
    (both sides fit every seed under one shared threshold), falling back to
    the configured floor when there are no dims to predict from.
    """
    if cfg.arm_timeout_seconds is None:
        return None
    if cfg.arm_timeout_seconds == "auto":
        base_arm = default_arm()
        resolved = [
            timeout
            for merged in (base_arm, {**base_arm, **(winner_knobs or {})})
            if (timeout := resolve_arm_timeout(cfg, merged, dims)[0]) is not None
        ]
        if not resolved:
            # The auto resolver had no dims to predict from and no floor:
            # treat as no timeout, same as arm_timeout_seconds=None.
            return None
        screening = max(resolved)
    else:
        screening = float(cfg.arm_timeout_seconds)
    overrides = sampler_overrides or {}
    base = (cfg.num_samples or 1000) + (cfg.num_warmup or 1000)
    scaled = overrides.get("num_samples", cfg.num_samples or 1000) + overrides.get(
        "num_warmup", cfg.num_warmup or 1000
    )
    return screening * max(1.0, scaled / base)


def run_confirmation(
    winner_knobs: dict[str, Any],
    cfg: SweepConfig,
    seeds: tuple[int, ...] = (42, 43, 44),
    promote_z: float = 2.0,
    sampler_overrides: dict[str, int] | None = None,
    launch: Callable[..., tuple[int, str]] | None = None,
    dims: dict[str, int] | None = None,
) -> ConfirmationResult:
    """Fit reference + winner on each seed, pair per seed, demand consistency.

    ``sampler_overrides`` applies to EVERY confirmation fit (both sides of
    every seed), so tiers with ``publication_confirm`` run the whole
    confirmation at publication scale. Results checkpoint to
    ``<sweep_dir>/confirmation.json`` after every seed, and a re-entry reuses
    any prior seed whose run dirs still carry their log-likelihood snapshots
    and whose manifests still prove they are this experiment (re-pairing is
    cheap; only missing or unprovable seeds refit).
    """
    from panelcast import __version__

    launch = launch or launch_arm
    panelcast_bin = cfg.panelcast_bin or _default_panelcast_bin()
    out_path = cfg.sweep_dir / "confirmation.json"
    cfg.sweep_dir.mkdir(parents=True, exist_ok=True)
    base = default_arm()
    descriptor_hash = _descriptor_hash(cfg)
    base_config = _base_config_payload(cfg, sampler_overrides)
    result = ConfirmationResult(
        winner_knobs=winner_knobs,
        promote_z=promote_z,
        sampler=_sampler_echo(cfg, sampler_overrides),
        version=__version__,
        seeds_planned=list(seeds),
        dataset_descriptor_hash=descriptor_hash,
        base_config_hash=_canonical_hash(base_config),
    )
    identity = {
        "winner_knobs": winner_knobs,
        "promote_z": promote_z,
        "sampler": result.sampler,
        "version": result.version,
        "seeds_planned": result.seeds_planned,
        # The winner knobs say what varies; these say what it varies *from*.
        # Without them one sweep id could confirm a winner against another
        # domain's snapshots and report the verdict as this domain's.
        "dataset_descriptor_hash": descriptor_hash,
        "base_config_hash": result.base_config_hash,
    }
    # One source for both sides: the fits are launched from this, and cached
    # runs are checked against it, so the two cannot drift into a cache that
    # rejects every run it built.
    arms = {"reference": dict(base), "winner": {**base, **winner_knobs}}
    # What a manifest has to record before it can be believed to be this fit
    # rather than the other side of the pair, or another seed, or another domain.
    discriminating = frozenset(
        {"seed"}
        | {k for k, v in arms["winner"].items() if arms["reference"].get(k) != v}
        | ({"dataset"} if cfg.dataset is not None else set())
    )

    def fit_config(label: str, seed: int) -> dict[str, Any]:
        """What the run behind ``label`` on ``seed`` must say it was fit with."""
        return _fit_config_payload(cfg, arms[label], sampler_overrides, seed)

    prior = _reusable_prior_seeds(
        out_path, identity, descriptor_hash, fit_config, discriminating
    )
    timeout = _confirmation_timeout(cfg, sampler_overrides, winner_knobs, dims)

    def _resolve(run_id: str, seed: int, label: str, *, after_fit: bool) -> Path:
        """This fit's run dir, contained — raised in the shape ``_one_fit`` uses.

        Re-raising as ``RuntimeError`` keeps the fail-closed path from depending
        on how wide the seed loop's handler happens to be. The two phases mean
        very different things to whoever reads the log, so they say so: a
        pre-launch refusal costs nothing, while a post-fit one means a full fit
        has already run. Where it wrote depends on why the name was refused,
        which is ``refusal_detail``'s job — the same wording the arm handshake
        emits — not this function's. The refused name is left exactly as it is;
        this lookup does not delete (#413).
        """
        try:
            return sweep_run_dir(cfg.pipeline_output_base, run_id, field="confirmation run_id")
        except RunPathError as exc:
            detail = refusal_detail(
                cfg.pipeline_output_base,
                run_id,
                exc,
                field="confirmation run_id",
                after_fit=after_fit,
            )
            phase = "after its fit" if after_fit else "before launching"
            raise RuntimeError(f"{label} fit on seed {seed} refused {phase} {detail}") from exc

    def _one_fit(seed: int, label: str) -> Path | None:
        config_path = cfg.sweep_dir / f"confirm_{label}_seed{seed}.yaml"
        # Named up front (#167 handshake) — no dependence on the mutable
        # `latest` pointer; unique per attempt so retries never collide.
        run_id = f"sel_{cfg.sweep_id}_confirm_{label}_seed{seed}_{datetime.now():%Y%m%dT%H%M%S%f}"
        # The id's shape is decidable before the fit, so refuse a bad one for
        # free rather than after a publication-scale run. The return value is
        # deliberately discarded — nothing exists at that name yet, so only the
        # post-fit call below can see a symlink escape
        # (test_confirmation_lookup_refuses_a_symlinked_escape covers both).
        _resolve(run_id, seed, label, after_fit=False)
        _write_config(cfg, arms[label], seed, config_path, sampler_overrides, run_id=run_id)
        log.info("confirmation_fit_start", label=label, seed=seed, timeout=timeout)
        started = time.monotonic()
        code, tail = launch(config_path, panelcast_bin, timeout)
        log.info(
            "confirmation_fit_done",
            label=label,
            seed=seed,
            returncode=code,
            seconds=round(time.monotonic() - started, 1),
        )
        if code != 0:
            raise RuntimeError(f"{label} fit failed on seed {seed}: {tail[-500:]}")
        # Again now that the directory exists: the symlink half of containment
        # has nothing to resolve through until then, and this is the path whose
        # contents get read and recorded.
        run_dir = _resolve(run_id, seed, label, after_fit=True)
        return run_dir if run_dir.exists() else None

    for seed in seeds:
        cached = _score_cached_seed(prior[seed]) if seed in prior else None
        if cached is not None:
            log.info("confirmation_seed_reused", seed=seed, winner_run=cached.winner_run)
            result.seeds.append(cached)
            out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            continue
        seed_result = SeedResult(seed=seed)
        try:
            ref_run = _one_fit(seed, "reference")
            seed_result.reference_run = str(ref_run) if ref_run else None
            win_run = _one_fit(seed, "winner")
            seed_result.winner_run = str(win_run) if win_run else None
            seed_result.winner_converged = _run_converged(win_run)
            if ref_run is None or win_run is None:
                raise RuntimeError("run directory not resolved after fit")
            if win_run == ref_run:
                raise RuntimeError(
                    f"winner fit resolved to the reference run ({ref_run}); "
                    "stale latest pointer — refusing to self-pair"
                )
            pair: PairedElpd = paired_elpd(
                win_run / "evaluation" / "log_likelihood.nc",
                ref_run / "evaluation" / "log_likelihood.nc",
            )
            seed_result.elpd = {"diff": pair.diff, "dse": pair.dse, "z": pair.z, "n": pair.n}
        except (RuntimeError, OSError, ValueError) as exc:
            seed_result.error = str(exc)
            log.warning("confirmation_seed_failed", seed=seed, error=str(exc))
        result.seeds.append(seed_result)
        out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    log.info("confirmation_complete", confirmed=result.confirmed, n_seeds=len(seeds))
    return result


def render_confirmation(result: ConfirmationResult) -> str:
    """Markdown block for the report: the recommendation, never the flip."""
    lines = [
        "## Multi-seed confirmation",
        "",
        "| seed | elpd_diff | dse | z | converged | runs |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in result.seeds:
        conv = "-" if s.winner_converged is None else ("PASS" if s.winner_converged else "FAIL")
        if s.elpd and s.elpd.get("z") is not None:
            lines.append(
                f"| {s.seed} | {s.elpd['diff']:+.1f} | {s.elpd['dse']:.1f} | "
                f"{s.elpd['z']:+.2f} | {conv} | ok |"
            )
        elif s.elpd:
            # A zero-variance paired diff leaves z undefined (winner ≈ reference).
            lines.append(
                f"| {s.seed} | {s.elpd['diff']:+.1f} | {s.elpd['dse']:.1f} "
                f"| - | {conv} | degenerate |"
            )
        else:
            lines.append(f"| {s.seed} | - | - | - | {conv} | {s.error or 'failed'} |")
    lines.append("")
    if result.confirmed:
        lines.append(
            f"CONFIRMED: the winner clears z ≥ {result.promote_z:g} and converges on "
            "every seed at publication scale. `select` recommends promotion; the "
            "default flip remains a manual PR with this table as its evidence."
        )
    else:
        conv_failed = any(
            s.winner_converged is False for s in result.seeds if s.elpd is not None
        )
        reason = (
            "the winner failed the convergence gate at publication scale on at least one seed"
            if conv_failed
            else "the effect does not hold across seeds at the pre-registered threshold"
        )
        lines.append(f"NOT CONFIRMED: {reason}. Treat the sweep ranking as noise-level.")
    return "\n".join(lines) + "\n"


__all__ = [
    "ConfirmationResult",
    "SeedResult",
    "render_confirmation",
    "run_confirmation",
]
