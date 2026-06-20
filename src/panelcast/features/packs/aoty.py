"""AOTY (music-domain) feature pack.

Registers the blocks that only make sense for the music dataset: genre PCA,
album type, and collaboration features. The class implementations stay in
their original modules; only the registration lives here so non-music
descriptors (``feature_packs: []``) never see these names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from panelcast.features.album_type import AlbumTypeBlock
from panelcast.features.collaboration import CollaborationBlock
from panelcast.features.genre import GenreBlock, GenrePCABlock

if TYPE_CHECKING:
    from panelcast.features.registry import FeatureRegistry


def register(registry: "FeatureRegistry") -> None:
    """Register the music-domain blocks on a feature registry."""
    registry.register("genre", lambda params: GenreBlock(params))
    registry.register("genre_pca", lambda params: GenrePCABlock(params))  # Backwards compat
    registry.register("album_type", lambda params: AlbumTypeBlock(params))
    registry.register("collaboration", lambda params: CollaborationBlock(params))
