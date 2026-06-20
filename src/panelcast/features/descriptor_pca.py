"""Descriptor PCA feature block (placeholder, not active in default pipeline)."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput


class DescriptorPCABlock(BaseFeatureBlock):
    name: ClassVar[str] = "descriptor_pca"
    requires: ClassVar[list[str]] = []

    def transform(self, df: pd.DataFrame, ctx: FeatureContext) -> FeatureOutput:
        raise NotImplementedError(
            "DescriptorPCABlock is a stub — implement transform() before "
            "registering in build_default_registry()."
        )
