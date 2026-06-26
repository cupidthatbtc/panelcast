import pytest

from panelcast.features.base import BaseFeatureBlock
from panelcast.features.registry import (
    FeatureRegistry,
    FeatureSpec,
    build_default_registry,
    parse_feature_specs,
)


def test_registry_builds_blocks():
    config = {"features": {"blocks": [{"name": "temporal", "params": {}}]}}
    registry = build_default_registry()
    specs = parse_feature_specs(config)
    blocks = registry.build_all(specs)
    assert blocks[0].name == "temporal"


# ---------------------------------------------------------------------------
# FeatureSpec dataclass tests
# ---------------------------------------------------------------------------


class TestFeatureSpec:
    def test_feature_spec_stores_name_and_params(self):
        spec = FeatureSpec(name="test_block", params={"key": "value"})
        assert spec.name == "test_block"
        assert spec.params == {"key": "value"}

    def test_feature_spec_empty_params(self):
        spec = FeatureSpec(name="test_block", params={})
        assert spec.params == {}

    def test_feature_spec_equality(self):
        spec1 = FeatureSpec(name="a", params={"x": 1})
        spec2 = FeatureSpec(name="a", params={"x": 1})
        assert spec1 == spec2

    def test_feature_spec_inequality_name(self):
        spec1 = FeatureSpec(name="a", params={})
        spec2 = FeatureSpec(name="b", params={})
        assert spec1 != spec2

    def test_feature_spec_inequality_params(self):
        spec1 = FeatureSpec(name="a", params={"x": 1})
        spec2 = FeatureSpec(name="a", params={"x": 2})
        assert spec1 != spec2


# ---------------------------------------------------------------------------
# FeatureRegistry tests
# ---------------------------------------------------------------------------


class TestFeatureRegistry:
    def test_register_and_build(self):
        registry = FeatureRegistry()
        registry.register("my_block", lambda params: BaseFeatureBlock(params))
        spec = FeatureSpec(name="my_block", params={"a": 1})
        block = registry.build(spec)
        assert isinstance(block, BaseFeatureBlock)
        assert block.params == {"a": 1}

    def test_register_duplicate_raises_value_error(self):
        registry = FeatureRegistry()
        registry.register("my_block", lambda params: BaseFeatureBlock(params))
        with pytest.raises(ValueError, match="already registered"):
            registry.register("my_block", lambda params: BaseFeatureBlock(params))

    def test_build_unknown_block_raises_key_error(self):
        registry = FeatureRegistry()
        spec = FeatureSpec(name="nonexistent", params={})
        with pytest.raises(KeyError, match="Unknown feature block"):
            registry.build(spec)

    def test_build_all_returns_list(self):
        registry = FeatureRegistry()
        registry.register("block_a", lambda params: BaseFeatureBlock(params))
        registry.register("block_b", lambda params: BaseFeatureBlock(params))
        specs = [
            FeatureSpec(name="block_a", params={}),
            FeatureSpec(name="block_b", params={"x": 1}),
        ]
        blocks = registry.build_all(specs)
        assert len(blocks) == 2
        assert isinstance(blocks[0], BaseFeatureBlock)
        assert isinstance(blocks[1], BaseFeatureBlock)

    def test_build_all_empty_specs(self):
        registry = FeatureRegistry()
        blocks = registry.build_all([])
        assert blocks == []

    def test_build_passes_params_to_builder(self):
        received = {}

        def builder(params):
            received.update(params)
            return BaseFeatureBlock(params)

        registry = FeatureRegistry()
        registry.register("custom", builder)
        spec = FeatureSpec(name="custom", params={"alpha": 0.5, "beta": 2})
        registry.build(spec)
        assert received == {"alpha": 0.5, "beta": 2}

    def test_multiple_registrations_independent(self):
        registry = FeatureRegistry()
        registry.register("block_a", lambda params: BaseFeatureBlock(params))
        registry.register("block_b", lambda params: BaseFeatureBlock(params))
        spec_a = FeatureSpec(name="block_a", params={})
        spec_b = FeatureSpec(name="block_b", params={})
        block_a = registry.build(spec_a)
        block_b = registry.build(spec_b)
        assert block_a is not block_b


# ---------------------------------------------------------------------------
# parse_feature_specs tests
# ---------------------------------------------------------------------------


class TestParseFeatureSpecs:
    def test_parses_single_block(self):
        config = {"features": {"blocks": [{"name": "temporal", "params": {"x": 1}}]}}
        specs = parse_feature_specs(config)
        assert len(specs) == 1
        assert specs[0].name == "temporal"
        assert specs[0].params == {"x": 1}

    def test_parses_multiple_blocks(self):
        config = {
            "features": {
                "blocks": [
                    {"name": "core_numeric", "params": {}},
                    {"name": "temporal", "params": {"n": 5}},
                    {"name": "genre", "params": {"min_genre_count": 10}},
                ]
            }
        }
        specs = parse_feature_specs(config)
        assert len(specs) == 3
        assert specs[0].name == "core_numeric"
        assert specs[1].name == "temporal"
        assert specs[2].name == "genre"

    def test_empty_blocks_returns_empty_list(self):
        config = {"features": {"blocks": []}}
        specs = parse_feature_specs(config)
        assert specs == []

    def test_missing_features_key_returns_empty(self):
        config = {}
        specs = parse_feature_specs(config)
        assert specs == []

    def test_missing_blocks_key_returns_empty(self):
        config = {"features": {}}
        specs = parse_feature_specs(config)
        assert specs == []

    def test_missing_name_raises_value_error(self):
        config = {"features": {"blocks": [{"params": {}}]}}
        with pytest.raises(ValueError, match="missing name"):
            parse_feature_specs(config)

    def test_empty_name_raises_value_error(self):
        config = {"features": {"blocks": [{"name": "", "params": {}}]}}
        with pytest.raises(ValueError, match="missing name"):
            parse_feature_specs(config)

    def test_missing_params_defaults_to_empty_dict(self):
        config = {"features": {"blocks": [{"name": "core_numeric"}]}}
        specs = parse_feature_specs(config)
        assert specs[0].params == {}

    def test_none_name_raises_value_error(self):
        config = {"features": {"blocks": [{"name": None, "params": {}}]}}
        with pytest.raises(ValueError, match="missing name"):
            parse_feature_specs(config)


# ---------------------------------------------------------------------------
# build_default_registry tests
# ---------------------------------------------------------------------------


class TestBuildDefaultRegistry:
    def test_returns_feature_registry(self):
        registry = build_default_registry()
        assert isinstance(registry, FeatureRegistry)

    def test_core_numeric_registered(self):
        registry = build_default_registry()
        spec = FeatureSpec(name="core_numeric", params={"columns": ["x"]})
        block = registry.build(spec)
        assert block.name == "core_numeric"

    def test_contains_temporal(self):
        registry = build_default_registry()
        spec = FeatureSpec(name="temporal", params={})
        block = registry.build(spec)
        assert block.name == "temporal"

    def test_contains_artist_reputation(self):
        registry = build_default_registry()
        spec = FeatureSpec(name="artist_reputation", params={})
        block = registry.build(spec)
        assert block.name == "artist_history"

    def test_contains_artist_history(self):
        registry = build_default_registry()
        spec = FeatureSpec(name="artist_history", params={})
        block = registry.build(spec)
        assert block.name == "artist_history"

    def test_contains_genre(self):
        registry = build_default_registry()
        spec = FeatureSpec(name="genre", params={})
        block = registry.build(spec)
        assert block.name == "genre"

    def test_contains_genre_pca(self):
        registry = build_default_registry()
        spec = FeatureSpec(name="genre_pca", params={})
        block = registry.build(spec)
        assert block.name == "genre"

    def test_contains_album_type(self):
        registry = build_default_registry()
        spec = FeatureSpec(name="album_type", params={})
        block = registry.build(spec)
        assert block.name == "album_type"

    def test_contains_collaboration(self):
        registry = build_default_registry()
        spec = FeatureSpec(name="collaboration", params={})
        block = registry.build(spec)
        assert block.name == "collaboration"

    def test_unknown_name_raises_key_error(self):
        registry = build_default_registry()
        spec = FeatureSpec(name="nonexistent_block", params={})
        with pytest.raises(KeyError, match="Unknown feature block"):
            registry.build(spec)

    def test_all_blocks_accept_params(self):
        registry = build_default_registry()
        block_names = [
            "temporal",
            "artist_reputation",
            "artist_history",
            "genre",
            "genre_pca",
            "album_type",
            "collaboration",
        ]
        for name in block_names:
            spec = FeatureSpec(name=name, params={"custom_param": "test"})
            block = registry.build(spec)
            assert block.params == {"custom_param": "test"}


# ---------------------------------------------------------------------------
# Integration: parse_feature_specs -> build_default_registry -> build_all
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_end_to_end_single_block(self):
        config = {"features": {"blocks": [{"name": "temporal", "params": {}}]}}
        registry = build_default_registry()
        specs = parse_feature_specs(config)
        blocks = registry.build_all(specs)
        assert len(blocks) == 1
        assert blocks[0].name == "temporal"

    def test_end_to_end_multiple_blocks(self):
        config = {
            "features": {
                "blocks": [
                    {"name": "temporal", "params": {}},
                    {"name": "album_type", "params": {}},
                    {"name": "collaboration", "params": {}},
                ]
            }
        }
        registry = build_default_registry()
        specs = parse_feature_specs(config)
        blocks = registry.build_all(specs)
        assert len(blocks) == 3
        names = [b.name for b in blocks]
        assert names == ["temporal", "album_type", "collaboration"]

    def test_end_to_end_empty_config(self):
        config = {}
        registry = build_default_registry()
        specs = parse_feature_specs(config)
        blocks = registry.build_all(specs)
        assert blocks == []
