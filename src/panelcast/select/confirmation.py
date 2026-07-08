"""Multi-seed confirmation for a sweep winner (#102, A4).

Productizes the manual #40 recipe (.audit/transform_latent_bakeoff/MULTISEED.md):
for each confirmation seed, fit the reference (shipped defaults) and the winner
on that seed, then pair their per-point held-out ELPD. The winner confirms only
when the direction holds on EVERY seed — a single-seed z is one draw from the
selection lottery. Seeds are feature-affecting, so every confirmation run
rebuilds the flat caches (strictly serial, like the sweep).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml  # type: ignore[import-untyped]

from panelcast.select.runner import (
    SweepConfig,
    _default_panelcast_bin,
    launch_arm,
    resolve_arm_timeout,
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
    """Per-seed paired verdicts plus the holds-on-every-seed conclusion."""

    winner_knobs: dict[str, Any]
    seeds: list[SeedResult] = field(default_factory=list)
    promote_z: float = 2.0

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
            s.elpd.get("z") is not None
            and s.elpd["z"] >= self.promote_z
            and s.winner_converged is True
            for s in measured
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner_knobs": self.winner_knobs,
            "promote_z": self.promote_z,
            "confirmed": self.confirmed,
            "seeds": [asdict(s) for s in self.seeds],
        }


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


def _write_config(
    cfg: SweepConfig,
    merged: dict[str, Any],
    seed: int,
    path: Path,
    sampler_overrides: dict[str, int] | None = None,
) -> None:
    payload: dict[str, Any] = {
        **cfg.extra_config,
        **merged,
        "seed": seed,
        "stages": _CONFIRMATION_STAGES,
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
    payload.update(sampler_overrides or {})
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
        screening = max(
            resolve_arm_timeout(cfg, merged, dims)[0]
            for merged in (base_arm, {**base_arm, **(winner_knobs or {})})
        )
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
    ``<sweep_dir>/confirmation.json`` after every seed.
    """
    from panelcast.paths import resolve_latest

    launch = launch or launch_arm
    panelcast_bin = cfg.panelcast_bin or _default_panelcast_bin()
    out_path = cfg.sweep_dir / "confirmation.json"
    cfg.sweep_dir.mkdir(parents=True, exist_ok=True)
    base = default_arm()
    result = ConfirmationResult(winner_knobs=winner_knobs, promote_z=promote_z)
    timeout = _confirmation_timeout(cfg, sampler_overrides, winner_knobs, dims)

    def _one_fit(merged: dict[str, Any], seed: int, label: str) -> Path | None:
        config_path = cfg.sweep_dir / f"confirm_{label}_seed{seed}.yaml"
        _write_config(cfg, merged, seed, config_path, sampler_overrides)
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
        run_dir = resolve_latest()
        # Dereference so the pairing survives the mutable `latest` link moving.
        return run_dir.resolve() if run_dir is not None else None

    for seed in seeds:
        seed_result = SeedResult(seed=seed)
        try:
            ref_run = _one_fit(dict(base), seed, "reference")
            seed_result.reference_run = str(ref_run) if ref_run else None
            win_run = _one_fit({**base, **winner_knobs}, seed, "winner")
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
