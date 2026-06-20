"""Feature blocks package."""

from .album_type import AlbumTypeBlock
from .artist import ArtistHistoryBlock, ArtistReputationBlock
from .collaboration import CollaborationBlock
from .genre import GenreBlock, GenrePCABlock
from .history import EntityHistoryBlock
from .registry import FeatureRegistry, FeatureSpec, build_default_registry, parse_feature_specs
from .temporal import TemporalBlock

__all__ = [
    "AlbumTypeBlock",
    "ArtistHistoryBlock",
    "ArtistReputationBlock",
    "CollaborationBlock",
    "EntityHistoryBlock",
    "GenreBlock",
    "GenrePCABlock",
    "TemporalBlock",
    "FeatureRegistry",
    "FeatureSpec",
    "build_default_registry",
    "parse_feature_specs",
]
