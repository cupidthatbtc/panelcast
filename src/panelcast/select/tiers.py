"""Effort tiers for `panelcast select` (#103, A8).

A tier names how heavy a sweep is — which stages run, the sampler settings, and
whether the winner gets multi-seed and publication-scale confirmation — so the
user picks intent (`quick` / `standard` / `thorough`) rather than raw fit
counts. Tiers live in the same YAML as the rules and grid, so a domain can ship
its own. Raw `--max-fits` / `--budget-hours` remain overrides on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from panelcast.select.rules import DEFAULT_RULES_PATH
from panelcast.select.runner import SweepConfig


@dataclass(frozen=True)
class EffortTier:
    """One effort level: search depth, sampler settings, confirmation policy."""

    name: str
    stages: tuple[int, ...]
    num_chains: int
    num_samples: int
    num_warmup: int
    stage3_fits: int = 0
    confirm: bool = False
    publication_confirm: dict[str, int] | None = None

    @property
    def include_stage2(self) -> bool:
        return 2 in self.stages


_SHIPPED_TIERS: dict[str, EffortTier] = {
    "quick": EffortTier("quick", (1,), 2, 500, 500, stage3_fits=0, confirm=False),
    "standard": EffortTier(
        "standard",
        (1, 2),
        4,
        1000,
        1000,
        stage3_fits=0,
        confirm=True,
        publication_confirm={"num_chains": 4, "num_samples": 5000, "num_warmup": 5000},
    ),
    "thorough": EffortTier(
        "thorough",
        (1, 2, 3),
        4,
        1000,
        1000,
        stage3_fits=8,
        confirm=True,
        publication_confirm={"num_chains": 4, "num_samples": 5000, "num_warmup": 5000},
    ),
}


def load_tiers(path: Path | None = None) -> dict[str, EffortTier]:
    """Effort tiers from the YAML ``tiers:`` block, shipped defaults when absent.

    An entry present in the YAML overrides the shipped tier of that name field
    by field, so a domain can retune one knob without restating the block.
    """
    path = path or DEFAULT_RULES_PATH
    tiers = dict(_SHIPPED_TIERS)
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return tiers
    block = payload.get("tiers") or {}
    for name, spec in block.items():
        base = tiers.get(name)
        merged = {
            "name": name,
            "stages": tuple(spec.get("stages", base.stages if base else (1,))),
            "num_chains": int(spec.get("num_chains", base.num_chains if base else 4)),
            "num_samples": int(spec.get("num_samples", base.num_samples if base else 1000)),
            "num_warmup": int(spec.get("num_warmup", base.num_warmup if base else 1000)),
            "stage3_fits": int(spec.get("stage3_fits", base.stage3_fits if base else 0)),
            "confirm": bool(spec.get("confirm", base.confirm if base else False)),
            "publication_confirm": spec.get(
                "publication_confirm", base.publication_confirm if base else None
            ),
        }
        tiers[name] = EffortTier(**merged)
    return tiers


def resolve_tier(effort: str, path: Path | None = None) -> EffortTier:
    tiers = load_tiers(path)
    if effort not in tiers:
        raise ValueError(f"Unknown effort tier '{effort}'. Known: {sorted(tiers)}.")
    return tiers[effort]


def tier_to_sweep_config(
    tier: EffortTier,
    sweep_id: str,
    dataset: str | None = None,
    output_root: Path = Path("outputs/select"),
    max_fits: int | None = None,
    budget_hours: float | None = None,
    promote_z: float = 2.0,
    panelcast_bin: str | None = None,
    arm_timeout_seconds: float | None = None,
) -> SweepConfig:
    """Map a tier (plus raw overrides) onto the runner's SweepConfig."""
    return SweepConfig(
        sweep_id=sweep_id,
        dataset=dataset,
        output_root=output_root,
        max_fits=max_fits,
        budget_hours=budget_hours,
        include_stage2=tier.include_stage2,
        stage3_fits=tier.stage3_fits if 3 in tier.stages else 0,
        winner_z=promote_z,
        num_chains=tier.num_chains,
        num_samples=tier.num_samples,
        num_warmup=tier.num_warmup,
        panelcast_bin=panelcast_bin,
        arm_timeout_seconds=arm_timeout_seconds,
    )


__all__ = [
    "EffortTier",
    "load_tiers",
    "resolve_tier",
    "tier_to_sweep_config",
]
