"""Tests for the DatasetDescriptor (default-equals-AOTY contract)."""

from __future__ import annotations

import pytest

from panelcast.config.descriptor import (
    DEFAULT_DESCRIPTOR,
    DatasetDescriptor,
    FeatureBlockSpec,
    load_descriptor,
)
from panelcast.data.cleaning import RAW_TO_CANONICAL
from panelcast.data.validation import OPTIONAL_RAW_COLUMNS, REQUIRED_RAW_COLUMNS


class TestDefaultEqualsAoty:
    """DatasetDescriptor() must reproduce every AOTY literal it replaced."""

    def test_identity_columns(self):
        d = DatasetDescriptor()
        assert d.entity_col == "Artist"
        assert d.event_col == "Album"
        assert d.date_col == "Release_Date"
        assert d.parsed_date_col == "Release_Date_Parsed"
        assert d.year_col == "Year"
        assert d.date_format == "%B %d, %Y"

    def test_targets(self):
        d = DatasetDescriptor()
        assert d.target_col == "User_Score"
        assert d.target_bounds == (0.0, 100.0)
        assert d.model_prefix == "user"
        assert d.n_obs_col == "User_Ratings"
        assert d.secondary_target_col == "Critic_Score"
        assert d.secondary_prefix == "critic"
        assert d.secondary_n_obs_col == "Critic_Reviews"

    def test_raw_source(self):
        d = DatasetDescriptor()
        assert d.raw_path_env == "AOTY_DATASET_PATH"
        assert d.raw_path_default == "data/raw/all_albums_full.csv"
        assert d.encoding == "utf-8-sig"

    def test_raw_column_map_matches_cleaning_constant(self):
        assert DatasetDescriptor().raw_column_map == RAW_TO_CANONICAL

    def test_raw_column_lists_match_validation_constants(self):
        d = DatasetDescriptor()
        assert d.required_raw_columns == REQUIRED_RAW_COLUMNS
        assert d.optional_raw_columns == OPTIONAL_RAW_COLUMNS

    def test_prep_defaults(self):
        d = DatasetDescriptor()
        assert d.min_obs_thresholds == [5, 10, 25]
        assert d.primary_min_obs == 10
        assert d.processed_name() == "user_score_minratings_10"
        assert d.processed_name(5) == "user_score_minratings_5"

    def test_cleaning_semantics(self):
        d = DatasetDescriptor()
        assert d.multi_entity_col == "All_Artists"
        assert d.multi_entity_separator == " | "
        assert d.unknown_entity_sentinel == "[unknown artist]"
        assert d.min_year == 1950
        assert d.group_size_bins["solo"] == [1, 1]
        assert d.group_size_bins["ensemble"] == [5, None]

    def test_feature_blocks_mirror_current_pipeline(self):
        d = DatasetDescriptor()
        assert [b.name for b in d.feature_blocks] == [
            "temporal",
            "album_type",
            "artist_history",
            "genre",
            "collaboration",
        ]
        genre = next(b for b in d.feature_blocks if b.name == "genre")
        assert genre.params == {"min_genre_count": 20, "n_components": 10}
        assert d.feature_packs == ["aoty"]

    def test_module_level_default_instance(self):
        assert DEFAULT_DESCRIPTOR == DatasetDescriptor()


class TestValidation:
    def test_primary_min_obs_must_be_in_thresholds(self):
        with pytest.raises(ValueError, match="primary_min_obs"):
            DatasetDescriptor(min_obs_thresholds=[5, 25], primary_min_obs=10)

    def test_target_bounds_ordering(self):
        with pytest.raises(ValueError, match="target_bounds"):
            DatasetDescriptor(target_bounds=(100.0, 0.0))

    def test_secondary_fields_all_or_none(self):
        with pytest.raises(ValueError, match="secondary"):
            DatasetDescriptor(secondary_target_col=None)
        d = DatasetDescriptor(
            secondary_target_col=None,
            secondary_prefix=None,
            secondary_n_obs_col=None,
        )
        assert d.secondary_target_col is None

    def test_template_requires_placeholder(self):
        with pytest.raises(ValueError, match="min_ratings"):
            DatasetDescriptor(processed_name_template="no_placeholder")


class TestDescriptorHash:
    def test_hash_stable_for_equal_descriptors(self):
        assert DatasetDescriptor().descriptor_hash() == DatasetDescriptor().descriptor_hash()

    def test_hash_changes_with_content(self):
        a = DatasetDescriptor()
        b = DatasetDescriptor(entity_col="Airframe")
        assert a.descriptor_hash() != b.descriptor_hash()

    def test_summary_block_keys(self):
        block = DatasetDescriptor().to_summary_block()
        assert block["name"] == "aoty"
        assert block["model_prefix"] == "user"
        assert block["target_bounds"] == [0.0, 100.0]
        assert len(block["descriptor_hash"]) == 64


class TestLoadDescriptor:
    def test_none_returns_default(self):
        assert load_descriptor(None) == DEFAULT_DESCRIPTOR

    def test_yaml_path_overrides_subset(self, tmp_path):
        yaml_path = tmp_path / "aero.yaml"
        yaml_path.write_text(
            "name: aero\n"
            "entity_col: Airframe\n"
            "target_col: Perf_Score\n"
            "target_bounds: [0.0, 10.0]\n"
            "model_prefix: perf\n"
            "secondary_target_col: null\n"
            "secondary_prefix: null\n"
            "secondary_n_obs_col: null\n",
            encoding="utf-8",
        )
        d = load_descriptor(yaml_path)
        assert d.name == "aero"
        assert d.entity_col == "Airframe"
        assert d.target_bounds == (0.0, 10.0)
        # Omitted keys keep AOTY defaults.
        assert d.event_col == "Album"
        assert d.min_obs_thresholds == [5, 10, 25]

    def test_bare_name_resolves_to_configs_datasets(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ds_dir = tmp_path / "configs" / "datasets"
        ds_dir.mkdir(parents=True)
        (ds_dir / "mydomain.yaml").write_text("name: mydomain\n", encoding="utf-8")
        d = load_descriptor("mydomain")
        assert d.name == "mydomain"

    def test_missing_descriptor_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="configs/datasets"):
            load_descriptor("nonexistent")

    def test_env_var_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_RAW_PATH", "/data/somewhere.csv")
        yaml_path = tmp_path / "envtest.yaml"
        yaml_path.write_text(
            "raw_path_default: ${MY_RAW_PATH}\n",
            encoding="utf-8",
        )
        d = load_descriptor(yaml_path)
        assert d.raw_path_default == "/data/somewhere.csv"


class TestFeatureBlockSpec:
    def test_defaults(self):
        spec = FeatureBlockSpec(name="temporal")
        assert spec.params == {}
