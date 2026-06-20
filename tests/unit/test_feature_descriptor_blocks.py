"""Descriptor-driven feature block construction on a non-AOTY (aero) domain.

The AOTY default path is pinned by the feature golden-hash guard
(tests/integration/test_feature_golden_hashes.py) and the existing block
tests; these tests prove the registry and blocks retarget through the
descriptor alone.
"""

import pandas as pd
import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.features.album_type import AlbumTypeBlock
from panelcast.features.artist import ArtistHistoryBlock
from panelcast.features.base import FeatureContext
from panelcast.features.collaboration import CollaborationBlock
from panelcast.features.genre import GenreBlock
from panelcast.features.history import EntityHistoryBlock
from panelcast.features.registry import FeatureSpec, build_default_registry
from panelcast.features.temporal import TemporalBlock
from panelcast.pipelines.build_features import get_feature_blocks
from tests.helpers.aero_data import make_aero_descriptor


@pytest.fixture
def ctx() -> FeatureContext:
    return FeatureContext(config={}, random_state=42)


def _aero_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Airframe": ["A", "A", "A", "B", "B"],
            "Flight_ID": ["A-F1", "A-F2", "A-F3", "B-F1", "B-F2"],
            "Flight_Date_Parsed": pd.to_datetime(
                ["2021-01-10", "2021-03-01", "2021-06-15", "2021-02-01", "2021-04-20"]
            ),
            "Year": [2021.0] * 5,
            "date_risk": ["low"] * 5,
            "Perf_Score": [6.0, 7.0, 8.0, 5.0, 5.5],
            "Sensor_Samples": [100, 120, 90, 60, 70],
        }
    )


class TestEntityHistoryBlockAero:
    def test_emits_prefixed_features_only_for_specs(self, ctx):
        block = EntityHistoryBlock(
            entity_col="Airframe",
            date_col="Flight_Date_Parsed",
            event_col="Flight_ID",
            score_specs=(("Perf_Score", "perf"),),
        )
        df = _aero_frame()
        output = block.fit_transform(df, ctx)
        assert output.feature_names == [
            "perf_prior_mean",
            "perf_prior_std",
            "perf_prior_count",
            "perf_trajectory",
            "is_debut",
        ]
        assert not any(c.startswith("critic_") for c in output.data.columns)

    def test_loo_semantics_and_debut(self, ctx):
        block = EntityHistoryBlock(
            entity_col="Airframe",
            date_col="Flight_Date_Parsed",
            event_col="Flight_ID",
            score_specs=(("Perf_Score", "perf"),),
        )
        df = _aero_frame()
        output = block.fit_transform(df, ctx)
        data = output.data
        # First flight per airframe is a debut with global-mean imputation.
        assert data.loc[0, "is_debut"] == 1
        assert data.loc[3, "is_debut"] == 1
        assert data.loc[0, "perf_prior_mean"] == pytest.approx(df["Perf_Score"].mean())
        # Third flight of airframe A sees mean of the first two only (LOO).
        assert data.loc[2, "perf_prior_mean"] == pytest.approx(6.5)
        assert data.loc[2, "perf_prior_count"] == 2

    def test_required_columns_derived(self):
        block = EntityHistoryBlock(
            entity_col="Airframe",
            date_col="Flight_Date_Parsed",
            event_col="Flight_ID",
            score_specs=(("Perf_Score", "perf"),),
        )
        assert block.required_columns == [
            "Airframe",
            "Flight_Date_Parsed",
            "Perf_Score",
            "Flight_ID",
        ]

    def test_empty_score_specs_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            EntityHistoryBlock(score_specs=())

    def test_aoty_subclass_keeps_legacy_defaults(self):
        block = ArtistHistoryBlock()
        assert block.name == "artist_history"
        assert block.required_columns == [
            "Artist",
            "Release_Date_Parsed",
            "User_Score",
            "Critic_Score",
            "Album",
        ]


class TestTemporalBlockAero:
    def test_temporal_features_with_custom_columns(self, ctx):
        block = TemporalBlock(
            entity_col="Airframe",
            date_col="Flight_Date_Parsed",
            year_col="Year",
            event_col="Flight_ID",
        )
        df = _aero_frame()
        output = block.fit_transform(df, ctx)
        data = output.data
        assert data.loc[0, "album_sequence"] == 1
        assert data.loc[2, "album_sequence"] == 3
        assert data.loc[1, "release_gap_days"] == 50
        assert data.loc[0, "release_gap_days"] == 0


class TestDescriptorRegistry:
    def test_aero_registry_has_generic_blocks_only(self):
        registry = build_default_registry(make_aero_descriptor())
        temporal = registry.build(FeatureSpec(name="temporal", params={}))
        history = registry.build(FeatureSpec(name="entity_history", params={}))
        assert temporal.entity_col == "Airframe"
        assert temporal.date_col == "Flight_Date_Parsed"
        assert history.score_specs == [("Perf_Score", "perf")]
        with pytest.raises(KeyError, match="genre"):
            registry.build(FeatureSpec(name="genre", params={}))
        with pytest.raises(KeyError, match="collaboration"):
            registry.build(FeatureSpec(name="collaboration", params={}))

    def test_default_registry_includes_aoty_pack(self):
        registry = build_default_registry()
        assert isinstance(registry.build(FeatureSpec(name="genre", params={})), GenreBlock)
        assert isinstance(registry.build(FeatureSpec(name="album_type", params={})), AlbumTypeBlock)
        assert isinstance(
            registry.build(FeatureSpec(name="collaboration", params={})), CollaborationBlock
        )

    def test_unknown_pack_raises(self):
        descriptor = DatasetDescriptor(feature_packs=["nonexistent"])
        with pytest.raises(KeyError, match="nonexistent"):
            build_default_registry(descriptor)

    def test_secondary_target_adds_score_spec(self):
        registry = build_default_registry()
        history = registry.build(FeatureSpec(name="entity_history", params={}))
        assert history.score_specs == [("User_Score", "user"), ("Critic_Score", "critic")]


class TestGetFeatureBlocksDescriptor:
    def test_aero_blocks_from_descriptor(self):
        blocks = get_feature_blocks(descriptor=make_aero_descriptor())
        assert [b.name for b in blocks] == ["temporal", "entity_history", "core_numeric"]

    def test_aero_ablation_maps_artist_flag_to_entity_history(self):
        blocks = get_feature_blocks(enable_artist=False, descriptor=make_aero_descriptor())
        # core_numeric is in no ablation group, so it survives --no-artist
        assert [b.name for b in blocks] == ["temporal", "core_numeric"]

    def test_default_blocks_match_legacy_composition(self):
        blocks = get_feature_blocks()
        assert [b.name for b in blocks] == [
            "temporal",
            "album_type",
            "artist_history",
            "genre",
            "collaboration",
        ]
        genre = blocks[3]
        assert genre.params == {"min_genre_count": 20, "n_components": 10}
