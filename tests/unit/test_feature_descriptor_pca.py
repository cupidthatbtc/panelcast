import pandas as pd
import pytest

from panelcast.features.base import BaseFeatureBlock, FeatureContext, FeatureOutput
from panelcast.features.descriptor_pca import DescriptorPCABlock


def test_descriptor_pca_block_transform_raises_not_implemented():
    """DescriptorPCABlock is a stub — transform() must raise NotImplementedError."""
    df = pd.DataFrame({"Artist": ["a"], "Year": [2000]})
    ctx = FeatureContext(config={}, random_state=0)
    block = DescriptorPCABlock({})
    with pytest.raises(NotImplementedError, match="stub"):
        block.transform(df, ctx)


def test_descriptor_pca_fit_transform_raises_not_implemented():
    df = pd.DataFrame({"Artist": ["a"], "Year": [2000]})
    ctx = FeatureContext(config={}, random_state=0)
    block = DescriptorPCABlock({})
    with pytest.raises(NotImplementedError, match="stub"):
        block.fit_transform(df, ctx)


class TestDescriptorPCABlockAttributes:
    def test_name_is_descriptor_pca(self):
        block = DescriptorPCABlock({})
        assert block.name == "descriptor_pca"

    def test_requires_is_empty(self):
        block = DescriptorPCABlock({})
        assert block.requires == []

    def test_inherits_from_base_feature_block(self):
        block = DescriptorPCABlock({})
        assert isinstance(block, BaseFeatureBlock)

    def test_default_params_empty(self):
        block = DescriptorPCABlock()
        assert block.params == {}

    def test_custom_params_stored(self):
        block = DescriptorPCABlock({"n_components": 5, "alpha": 0.1})
        assert block.params == {"n_components": 5, "alpha": 0.1}

    def test_none_params_default_to_empty_dict(self):
        block = DescriptorPCABlock(None)
        assert block.params == {}
