"""Feature registry.

Register and construct feature blocks by name from config. The default
registry is descriptor-aware: generic blocks (temporal, entity history) close
over the descriptor's column names, and domain packs named in
``descriptor.feature_packs`` contribute their domain-specific blocks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from panelcast.config.descriptor import DatasetDescriptor

from .artist import ArtistHistoryBlock, ArtistReputationBlock
from .base import FeatureBlock
from .basis import BasisBlock
from .core import CoreNumericBlock
from .history import EntityHistoryBlock
from .temporal import TemporalBlock


@dataclass
class FeatureSpec:
    name: str
    params: dict[str, Any]


class FeatureRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, Callable[[dict[str, Any]], FeatureBlock]] = {}

    def register(self, name: str, builder: Callable[[dict[str, Any]], FeatureBlock]) -> None:
        if name in self._builders:
            raise ValueError(f"Feature block already registered: {name}")
        self._builders[name] = builder

    def build(self, spec: FeatureSpec) -> FeatureBlock:
        if spec.name not in self._builders:
            raise KeyError(f"Unknown feature block: {spec.name}")
        return self._builders[spec.name](spec.params)

    def build_all(self, specs: list[FeatureSpec]) -> list[FeatureBlock]:
        return [self.build(spec) for spec in specs]


def parse_feature_specs(config: dict[str, Any]) -> list[FeatureSpec]:
    blocks = config.get("features", {}).get("blocks", [])
    specs: list[FeatureSpec] = []
    for block in blocks:
        name = block.get("name")
        params = block.get("params", {})
        if not name:
            raise ValueError("Feature block missing name")
        specs.append(FeatureSpec(name=name, params=params))
    return specs


def _score_specs(descriptor: DatasetDescriptor) -> tuple[tuple[str, str], ...]:
    """(score column, output prefix) pairs for the descriptor's targets."""
    specs = [(descriptor.target_col, descriptor.model_prefix)]
    if descriptor.secondary_target_col is not None and descriptor.secondary_prefix is not None:
        specs.append((descriptor.secondary_target_col, descriptor.secondary_prefix))
    return tuple(specs)


def build_default_registry(descriptor: DatasetDescriptor | None = None) -> FeatureRegistry:
    """Build the default feature registry for a descriptor.

    Generic blocks close over the descriptor's column names; packs listed in
    ``descriptor.feature_packs`` add their domain-specific blocks. The legacy
    AOTY names (artist_history, artist_reputation) stay registered with fixed
    AOTY defaults for backwards compatibility.
    """
    descriptor = descriptor or DatasetDescriptor()
    registry = FeatureRegistry()
    registry.register(
        "temporal",
        lambda params: TemporalBlock(
            params,
            entity_col=descriptor.entity_col,
            date_col=descriptor.parsed_date_col,
            year_col=descriptor.year_col,
            event_col=descriptor.event_col,
        ),
    )
    # Generic numeric pass-through; columns come from params, not the
    # descriptor, so a domain YAML names them per-block.
    registry.register("core_numeric", lambda params: CoreNumericBlock(params))
    registry.register("basis", lambda params: BasisBlock(params))
    registry.register(
        "entity_history",
        lambda params: EntityHistoryBlock(
            params,
            entity_col=descriptor.entity_col,
            date_col=descriptor.parsed_date_col,
            event_col=descriptor.event_col,
            score_specs=_score_specs(descriptor),
        ),
    )
    # Legacy AOTY aliases (fixed defaults, fixed block names).
    registry.register("artist_reputation", lambda params: ArtistReputationBlock(params))
    registry.register("artist_history", lambda params: ArtistHistoryBlock(params))

    for pack_name in descriptor.feature_packs:
        registrar = _PACK_REGISTRARS.get(pack_name)
        if registrar is None:
            raise KeyError(
                f"Unknown feature pack: {pack_name!r}. Available packs: {sorted(_PACK_REGISTRARS)}."
            )
        registrar(registry)
    return registry


def _register_aoty_pack(registry: FeatureRegistry) -> None:
    from panelcast.features.packs import aoty

    aoty.register(registry)


_PACK_REGISTRARS: dict[str, Callable[[FeatureRegistry], None]] = {
    "aoty": _register_aoty_pack,
}
