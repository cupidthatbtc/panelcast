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

from panelcast.select.runner import SweepConfig, _default_panelcast_bin, launch_arm
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
    error: str | None = None


@dataclass
class ConfirmationResult:
    """Per-seed paired verdicts plus the holds-on-every-seed conclusion."""

    winner_knobs: dict[str, Any]
    seeds: list[SeedResult] = field(default_factory=list)
    promote_z: float = 2.0

    @property
    def confirmed(self) -> bool:
        """Direction holds (z at or above threshold) on every measured seed."""
        measured = [s for s in self.seeds if s.elpd is not None]
        if len(measured) < len(self.seeds) or not measured:
            return False
        return all(
            s.elpd.get("z") is not None and s.elpd["z"] >= self.promote_z for s in measured
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner_knobs": self.winner_knobs,
            "promote_z": self.promote_z,
            "confirmed": self.confirmed,
            "seeds": [asdict(s) for s in self.seeds],
        }


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


def run_confirmation(
    winner_knobs: dict[str, Any],
    cfg: SweepConfig,
    seeds: tuple[int, ...] = (42, 43, 44),
    promote_z: float = 2.0,
    sampler_overrides: dict[str, int] | None = None,
    launch: Callable[[Path, str], tuple[int, str]] | None = None,
) -> ConfirmationResult:
    """Fit reference + winner on each seed, pair per seed, demand consistency.

    ``sampler_overrides`` lets the thorough tier run the final pair at
    publication scale. Results checkpoint to
    ``<sweep_dir>/confirmation.json`` after every seed.
    """
    from panelcast.paths import resolve_latest

    launch = launch or launch_arm
    panelcast_bin = cfg.panelcast_bin or _default_panelcast_bin()
    out_path = cfg.sweep_dir / "confirmation.json"
    cfg.sweep_dir.mkdir(parents=True, exist_ok=True)
    base = default_arm()
    result = ConfirmationResult(winner_knobs=winner_knobs, promote_z=promote_z)

    def _one_fit(merged: dict[str, Any], seed: int, label: str) -> Path | None:
        config_path = cfg.sweep_dir / f"confirm_{label}_seed{seed}.yaml"
        _write_config(cfg, merged, seed, config_path, sampler_overrides)
        log.info("confirmation_fit_start", label=label, seed=seed)
        started = time.monotonic()
        code, tail = launch(config_path, panelcast_bin)
        log.info(
            "confirmation_fit_done",
            label=label,
            seed=seed,
            returncode=code,
            seconds=round(time.monotonic() - started, 1),
        )
        if code != 0:
            raise RuntimeError(f"{label} fit failed on seed {seed}: {tail[-500:]}")
        return resolve_latest()

    for seed in seeds:
        seed_result = SeedResult(seed=seed)
        try:
            ref_run = _one_fit(dict(base), seed, "reference")
            seed_result.reference_run = str(ref_run) if ref_run else None
            win_run = _one_fit({**base, **winner_knobs}, seed, "winner")
            seed_result.winner_run = str(win_run) if win_run else None
            if ref_run is None or win_run is None:
                raise RuntimeError("run directory not resolved after fit")
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
        "| seed | elpd_diff | dse | z | runs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in result.seeds:
        if s.elpd and s.elpd.get("z") is not None:
            lines.append(
                f"| {s.seed} | {s.elpd['diff']:+.1f} | {s.elpd['dse']:.1f} | "
                f"{s.elpd['z']:+.2f} | ok |"
            )
        elif s.elpd:
            # A zero-variance paired diff leaves z undefined (winner ≈ reference).
            lines.append(
                f"| {s.seed} | {s.elpd['diff']:+.1f} | {s.elpd['dse']:.1f} | - | degenerate |"
            )
        else:
            lines.append(f"| {s.seed} | - | - | - | {s.error or 'failed'} |")
    lines.append("")
    if result.confirmed:
        lines.append(
            f"CONFIRMED: the winner clears z ≥ {result.promote_z:g} on every seed. "
            "`select` recommends promotion; the default flip remains a manual PR "
            "with this table as its evidence."
        )
    else:
        lines.append(
            "NOT CONFIRMED: the effect does not hold across seeds at the "
            "pre-registered threshold. Treat the sweep ranking as noise-level."
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "ConfirmationResult",
    "SeedResult",
    "render_confirmation",
    "run_confirmation",
]
