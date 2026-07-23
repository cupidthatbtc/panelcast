"""Entry-point plugin discovery for feature packs and likelihoods (#172)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.features.base import BaseFeatureBlock
from panelcast.features.registry import build_default_registry, discovered_plugins
from panelcast.models.bayes import likelihoods
from panelcast.models.bayes.likelihoods import (
    REGISTRY,
    available_families,
    find_likelihood,
    get_likelihood,
)


class FixtureBlock(BaseFeatureBlock):
    name = "fixture_block"

    def transform(self, df, ctx):  # pragma: no cover - never exercised here
        raise NotImplementedError


def _register_fixture_pack(registry) -> None:
    registry.register("fixture_block", lambda params: FixtureBlock(params))


FIXTURE_LIKELIHOOD = replace(REGISTRY["normal"], name="fixture_normal")


@dataclass
class FakeEntryPoint:
    name: str
    value: str
    _obj: Any
    dist: Any = None

    def load(self) -> Any:
        return self._obj


@pytest.fixture
def fixture_plugins(monkeypatch):
    entries = {
        "panelcast.feature_packs": [
            FakeEntryPoint(
                "fixture_pack",
                "fixture_plugin:register",
                _register_fixture_pack,
                SimpleNamespace(name="fixture-plugin", version="1.2.3"),
            )
        ],
        "panelcast.likelihoods": [
            FakeEntryPoint(
                "fixture_normal",
                "fixture_plugin:SPEC",
                FIXTURE_LIKELIHOOD,
                SimpleNamespace(name="fixture-plugin", version="1.2.3"),
            ),
            # A plugin trying to shadow a builtin must be ignored.
            FakeEntryPoint("studentt", "evil:SPEC", FIXTURE_LIKELIHOOD),
        ],
    }

    def fake_entry_points(*, group: str):
        return entries.get(group, [])

    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    likelihoods._discovered_likelihoods.cache_clear()
    yield entries
    likelihoods._discovered_likelihoods.cache_clear()


class TestFeaturePackDiscovery:
    def test_descriptor_resolves_a_plugin_pack(self, fixture_plugins):
        descriptor = DatasetDescriptor(feature_packs=["fixture_pack"])
        registry = build_default_registry(descriptor)
        from panelcast.features.registry import FeatureSpec

        block = registry.build(FeatureSpec(name="fixture_block", params={}))
        assert isinstance(block, FixtureBlock)

    def test_unknown_pack_error_lists_builtin_and_plugin_names(self, fixture_plugins):
        descriptor = DatasetDescriptor(feature_packs=["nope"])
        with pytest.raises(KeyError, match="(?s)aoty.*fixture_pack|fixture_pack.*aoty"):
            build_default_registry(descriptor)


class TestLikelihoodDiscovery:
    def test_plugin_family_resolves(self, fixture_plugins):
        assert get_likelihood("fixture_normal") is FIXTURE_LIKELIHOOD
        assert find_likelihood("fixture_normal") is FIXTURE_LIKELIHOOD
        assert "fixture_normal" in available_families()

    def test_builtins_shadow_plugins(self, fixture_plugins):
        assert get_likelihood("studentt") is REGISTRY["studentt"]

    def test_unknown_family_lists_everything(self, fixture_plugins):
        with pytest.raises(KeyError, match="fixture_normal"):
            get_likelihood("nope")

    def test_pipeline_config_accepts_a_plugin_family(self, fixture_plugins):
        from panelcast.pipelines.orchestrator import PipelineConfig

        config = PipelineConfig(likelihood_family="fixture_normal")
        assert config.likelihood_family == "fixture_normal"

    def test_select_space_enumerates_plugin_families(self, fixture_plugins):
        from panelcast.select.space import _family_values

        assert "fixture_normal" in _family_values()

    def test_non_spec_entry_point_is_a_type_error(self, fixture_plugins, monkeypatch):
        import importlib.metadata

        bad = [FakeEntryPoint("bad", "x:y", object())]
        monkeypatch.setattr(
            importlib.metadata, "entry_points", lambda *, group: bad if "likelihood" in group else []
        )
        likelihoods._discovered_likelihoods.cache_clear()
        with pytest.raises(TypeError, match="LikelihoodSpec"):
            likelihoods._discovered_likelihoods()
        likelihoods._discovered_likelihoods.cache_clear()


def test_no_direct_registry_lookups_outside_likelihoods():
    """Every family resolution must go through the plugin-aware accessors.

    A direct REGISTRY[...] lookup in a consumer is a seam plugins can't reach
    (the rollout path was missed exactly this way in review).
    """
    src = Path(__file__).resolve().parents[2] / "src" / "panelcast"
    offenders = [
        path.relative_to(src).as_posix()
        for path in sorted(src.rglob("*.py"))
        if path.name != "likelihoods.py" and "REGISTRY[" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"direct REGISTRY lookups outside likelihoods.py: {offenders}"


class TestProvenance:
    def test_discovered_plugins_records_dists_and_versions(self, fixture_plugins):
        plugins = discovered_plugins()
        assert plugins["panelcast.feature_packs:fixture_pack"] == {
            "value": "fixture_plugin:register",
            "dist": "fixture-plugin",
            "version": "1.2.3",
        }
        assert "panelcast.likelihoods:fixture_normal" in plugins

    def test_no_plugins_is_empty(self, monkeypatch):
        import importlib.metadata

        monkeypatch.setattr(importlib.metadata, "entry_points", lambda *, group: [])
        likelihoods._discovered_likelihoods.cache_clear()
        assert discovered_plugins() == {}
        assert available_families() == tuple(sorted(REGISTRY))
        likelihoods._discovered_likelihoods.cache_clear()
