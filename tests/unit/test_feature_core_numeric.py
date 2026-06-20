"""Tests for the core_numeric pass-through feature block."""

import numpy as np
import pandas as pd
import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.features.base import BaseFeatureBlock, FeatureContext, NotFittedError
from panelcast.features.core import CoreNumericBlock
from panelcast.features.registry import FeatureSpec, build_default_registry

CTX = FeatureContext(config={}, random_state=0)


def _train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Thrust_Margin": [1.0, 2.0, np.nan, 4.0],
            "Payload_Fraction": [0.1, 0.2, 0.3, 0.4],
            "Airframe": ["a", "a", "b", "b"],
        }
    )


class TestParamValidation:
    def test_missing_columns_param_raises(self):
        with pytest.raises(ValueError, match="columns"):
            CoreNumericBlock({})

    def test_none_params_raises(self):
        with pytest.raises(ValueError, match="columns"):
            CoreNumericBlock(None)

    def test_empty_columns_raises(self):
        with pytest.raises(ValueError, match="columns"):
            CoreNumericBlock({"columns": []})

    def test_non_string_columns_raises(self):
        with pytest.raises(ValueError, match="strings"):
            CoreNumericBlock({"columns": ["ok", 5]})

    def test_duplicate_columns_raises(self):
        with pytest.raises(ValueError, match="unique"):
            CoreNumericBlock({"columns": ["x", "x"]})

    def test_unknown_impute_raises(self):
        with pytest.raises(ValueError, match="impute"):
            CoreNumericBlock({"columns": ["x"], "impute": "mode"})


class TestAttributes:
    def test_name_and_requires(self):
        block = CoreNumericBlock({"columns": ["x"]})
        assert block.name == "core_numeric"
        assert block.requires == []
        assert isinstance(block, BaseFeatureBlock)

    def test_required_columns_mirror_params(self):
        block = CoreNumericBlock({"columns": ["x", "y"]})
        assert block.required_columns == ["x", "y"]


class TestFitTransform:
    def test_transform_before_fit_raises(self):
        block = CoreNumericBlock({"columns": ["Thrust_Margin"]})
        with pytest.raises(NotFittedError):
            block.transform(_train_df(), CTX)

    def test_missing_column_raises_at_fit(self):
        block = CoreNumericBlock({"columns": ["nope"]})
        with pytest.raises(ValueError, match="nope"):
            block.fit(_train_df(), CTX)

    def test_passthrough_values_and_names(self):
        block = CoreNumericBlock({"columns": ["Payload_Fraction"]})
        out = block.fit_transform(_train_df(), CTX)
        assert out.feature_names == ["Payload_Fraction"]
        assert list(out.data["Payload_Fraction"]) == [0.1, 0.2, 0.3, 0.4]

    def test_median_imputation_fitted_on_train(self):
        block = CoreNumericBlock({"columns": ["Thrust_Margin"]})
        block.fit(_train_df(), CTX)
        test_df = pd.DataFrame({"Thrust_Margin": [np.nan, 10.0]})
        out = block.transform(test_df, CTX)
        # train median of [1, 2, 4] = 2.0 — NOT a test-split statistic
        assert out.data["Thrust_Margin"].tolist() == [2.0, 10.0]
        assert out.metadata["fill_values"] == {"Thrust_Margin": 2.0}

    def test_mean_imputation(self):
        block = CoreNumericBlock({"columns": ["Thrust_Margin"], "impute": "mean"})
        block.fit(_train_df(), CTX)
        out = block.transform(pd.DataFrame({"Thrust_Margin": [np.nan]}), CTX)
        assert out.data["Thrust_Margin"].tolist() == [pytest.approx(7.0 / 3.0)]

    def test_zero_imputation(self):
        block = CoreNumericBlock({"columns": ["Thrust_Margin"], "impute": "zero"})
        block.fit(_train_df(), CTX)
        out = block.transform(pd.DataFrame({"Thrust_Margin": [np.nan]}), CTX)
        assert out.data["Thrust_Margin"].tolist() == [0.0]

    def test_non_numeric_strings_coerced_then_imputed(self):
        df = pd.DataFrame({"x": ["1.5", "bad", "2.5"]})
        block = CoreNumericBlock({"columns": ["x"]})
        out = block.fit_transform(df, CTX)
        assert out.data["x"].tolist() == [1.5, 2.0, 2.5]

    def test_all_missing_column_raises_at_fit(self):
        df = pd.DataFrame({"x": [np.nan, np.nan]})
        block = CoreNumericBlock({"columns": ["x"]})
        with pytest.raises(ValueError, match="no numeric values"):
            block.fit(df, CTX)

    def test_index_preserved(self):
        df = _train_df()
        df.index = [10, 20, 30, 40]
        block = CoreNumericBlock({"columns": ["Payload_Fraction"]})
        out = block.fit_transform(df, CTX)
        assert list(out.data.index) == [10, 20, 30, 40]

    def test_output_is_float_dtype(self):
        df = pd.DataFrame({"x": [1, 2, 3]})  # ints in
        block = CoreNumericBlock({"columns": ["x"]})
        out = block.fit_transform(df, CTX)
        assert out.data["x"].dtype == float


class TestRegistryIntegration:
    def test_buildable_from_default_registry(self):
        registry = build_default_registry(DatasetDescriptor())
        block = registry.build(
            FeatureSpec(name="core_numeric", params={"columns": ["Thrust_Margin"]})
        )
        assert isinstance(block, CoreNumericBlock)

    def test_registry_propagates_param_validation(self):
        registry = build_default_registry(DatasetDescriptor())
        with pytest.raises(ValueError, match="columns"):
            registry.build(FeatureSpec(name="core_numeric", params={}))
