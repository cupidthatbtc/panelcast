"""Tests for BaseFeatureBlock fit/transform enforcement.

These tests verify that the fit/transform pattern is properly enforced
to prevent data leakage by ensuring blocks are fitted before transform.
"""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from panelcast.features.base import BaseFeatureBlock, FeatureContext, FeatureOutput
from panelcast.features.errors import (
    FittedStatistics,
    FittedVocabulary,
    NotFittedError,
)


class ConcreteFeatureBlock(BaseFeatureBlock):
    """Concrete implementation for testing."""

    name = "test_block"

    def transform(self, df, ctx: FeatureContext) -> FeatureOutput:
        """Transform that properly checks fitted state."""
        self._check_is_fitted()
        return FeatureOutput(
            data=pd.DataFrame({"test_feature": [1] * len(df)}, index=df.index),
            feature_names=["test_feature"],
            metadata={"block": self.name},
        )


@pytest.fixture
def ctx():
    """Create a test FeatureContext."""
    return FeatureContext(config={}, random_state=42)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


class TestFitTransformEnforcement:
    """Tests for fit/transform state tracking."""

    def test_transform_before_fit_raises_not_fitted_error(self, sample_df, ctx):
        """Transform without fit must raise NotFittedError."""
        block = ConcreteFeatureBlock()

        with pytest.raises(NotFittedError) as exc_info:
            block.transform(sample_df, ctx)

        assert "test_block" in str(exc_info.value)
        assert "has not been fitted yet" in str(exc_info.value)

    def test_is_fitted_false_before_fit(self):
        """is_fitted should be False before fit() is called."""
        block = ConcreteFeatureBlock()
        assert block.is_fitted is False

    def test_is_fitted_true_after_fit(self, sample_df, ctx):
        """is_fitted should be True after fit() is called."""
        block = ConcreteFeatureBlock()
        block.fit(sample_df, ctx)
        assert block.is_fitted is True

    def test_fit_then_transform_succeeds(self, sample_df, ctx):
        """Transform after fit should succeed without exception."""
        block = ConcreteFeatureBlock()
        block.fit(sample_df, ctx)

        output = block.transform(sample_df, ctx)

        assert output.data is not None
        assert output.data.index.equals(sample_df.index)
        assert output.feature_names == ["test_feature"]

    def test_fit_transform_sets_fitted_state(self, sample_df, ctx):
        """fit_transform should set is_fitted to True."""
        block = ConcreteFeatureBlock()
        assert block.is_fitted is False

        output = block.fit_transform(sample_df, ctx)

        assert block.is_fitted is True
        assert output.data is not None


class TestFittedVocabulary:
    """Tests for FittedVocabulary frozen dataclass."""

    def test_fitted_vocabulary_is_immutable(self):
        """FittedVocabulary should not allow attribute modification."""
        vocab = FittedVocabulary(
            categories=("A", "B", "C"),
            category_to_idx={"A": 0, "B": 1, "C": 2},
            unknown_idx=-1,
        )

        with pytest.raises(FrozenInstanceError):
            vocab.categories = ("X", "Y", "Z")

    def test_fitted_vocabulary_encode_handles_unknown(self):
        """FittedVocabulary.encode should map unknown values to unknown_idx."""
        vocab = FittedVocabulary(
            categories=("A", "B", "C"),
            category_to_idx={"A": 0, "B": 1, "C": 2},
            unknown_idx=-1,
        )

        encoded = vocab.encode(["A", "D"])

        assert encoded[0] == 0  # "A" maps to index 0
        assert encoded[1] == -1  # "D" (unknown) maps to unknown_idx

    def test_fitted_vocabulary_encode_all_known(self):
        """FittedVocabulary.encode should correctly map all known values."""
        vocab = FittedVocabulary(
            categories=("A", "B", "C"),
            category_to_idx={"A": 0, "B": 1, "C": 2},
            unknown_idx=-1,
        )

        encoded = vocab.encode(["B", "A", "C"])

        assert encoded == [1, 0, 2]

    def test_fitted_vocabulary_custom_unknown_idx(self):
        """FittedVocabulary should respect custom unknown_idx value."""
        vocab = FittedVocabulary(
            categories=("A", "B"),
            category_to_idx={"A": 0, "B": 1},
            unknown_idx=99,  # Custom unknown index
        )

        encoded = vocab.encode(["A", "UNKNOWN"])

        assert encoded[0] == 0
        assert encoded[1] == 99


class TestFittedStatistics:
    """Tests for FittedStatistics frozen dataclass."""

    def test_fitted_statistics_is_immutable(self):
        """FittedStatistics should not allow attribute modification."""
        stats = FittedStatistics(mean=50.0, std=10.0, min_val=20.0, max_val=80.0)

        with pytest.raises(FrozenInstanceError):
            stats.mean = 100.0

    def test_fitted_statistics_stores_values(self):
        """FittedStatistics should correctly store all values."""
        stats = FittedStatistics(mean=50.0, std=10.0, min_val=20.0, max_val=80.0)

        assert stats.mean == 50.0
        assert stats.std == 10.0
        assert stats.min_val == 20.0
        assert stats.max_val == 80.0


class TestNotFittedError:
    """Tests for NotFittedError exception."""

    def test_not_fitted_error_is_value_error(self):
        """NotFittedError should be catchable as ValueError."""
        with pytest.raises(ValueError):
            raise NotFittedError("test message")

    def test_not_fitted_error_is_attribute_error(self):
        """NotFittedError should be catchable as AttributeError."""
        with pytest.raises(AttributeError):
            raise NotFittedError("test message")

    def test_not_fitted_error_message(self):
        """NotFittedError should preserve its message."""
        msg = "Custom error message"
        error = NotFittedError(msg)
        assert str(error) == msg


# ---------------------------------------------------------------------------
# Additional BaseFeatureBlock tests
# ---------------------------------------------------------------------------


class TestBaseFeatureBlockInit:
    """Tests for BaseFeatureBlock initialization."""

    def test_default_params_is_empty_dict(self):
        block = ConcreteFeatureBlock()
        assert block.params == {}

    def test_none_params_default_to_empty_dict(self):
        block = ConcreteFeatureBlock(params=None)
        assert block.params == {}

    def test_custom_params_stored(self):
        block = ConcreteFeatureBlock(params={"alpha": 0.5, "n": 10})
        assert block.params == {"alpha": 0.5, "n": 10}

    def test_default_name(self):
        block = BaseFeatureBlock()
        assert block.name == "base"

    def test_default_requires_empty(self):
        block = BaseFeatureBlock()
        assert block.requires == []

    def test_default_required_columns_empty(self):
        block = BaseFeatureBlock()
        assert block.required_columns == []


class TestBaseFeatureBlockValidateColumns:
    """Tests for validate_columns method."""

    def test_validate_columns_no_requirements(self, sample_df):
        block = BaseFeatureBlock()
        block.validate_columns(sample_df)  # Should not raise

    def test_validate_columns_all_present(self):
        block = BaseFeatureBlock()
        block.required_columns = ["col1", "col2"]
        df = pd.DataFrame({"col1": [1], "col2": [2], "col3": [3]})
        block.validate_columns(df)  # Should not raise

    def test_validate_columns_missing_raises_value_error(self):
        block = BaseFeatureBlock()
        block.required_columns = ["col1", "missing_col"]
        df = pd.DataFrame({"col1": [1], "col2": [2]})
        with pytest.raises(ValueError, match="missing required columns"):
            block.validate_columns(df)

    def test_validate_columns_all_missing(self):
        block = BaseFeatureBlock()
        block.required_columns = ["x", "y"]
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError) as exc_info:
            block.validate_columns(df)
        assert "x" in str(exc_info.value)
        assert "y" in str(exc_info.value)


class TestBaseFeatureBlockTransform:
    """Tests for base transform method."""

    def test_base_transform_raises_not_implemented(self, sample_df, ctx):
        block = BaseFeatureBlock()
        block.fit(sample_df, ctx)
        with pytest.raises(NotImplementedError):
            block.transform(sample_df, ctx)

    def test_base_transform_unfitted_raises_not_fitted(self, sample_df, ctx):
        block = BaseFeatureBlock()
        with pytest.raises(NotFittedError):
            block.transform(sample_df, ctx)


class TestBaseFeatureBlockFitTransform:
    """Tests for fit_transform convenience method."""

    def test_fit_transform_sets_fitted_and_returns_output(self, sample_df, ctx):
        block = ConcreteFeatureBlock()
        output = block.fit_transform(sample_df, ctx)
        assert block.is_fitted is True
        assert isinstance(output, FeatureOutput)

    def test_fit_transform_calls_fit_then_transform(self, sample_df, ctx):
        block = ConcreteFeatureBlock()
        output = block.fit_transform(sample_df, ctx)
        assert output.data.shape[0] == len(sample_df)
        assert output.feature_names == ["test_feature"]


class TestBaseFeatureBlockFit:
    """Tests for fit method behavior."""

    def test_fit_validates_columns(self, ctx):
        block = BaseFeatureBlock()
        block.required_columns = ["required_col"]
        df = pd.DataFrame({"other_col": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            block.fit(df, ctx)

    def test_fit_returns_self(self, sample_df, ctx):
        block = BaseFeatureBlock()
        result = block.fit(sample_df, ctx)
        assert result is block

    def test_fit_can_be_called_multiple_times(self, sample_df, ctx):
        block = ConcreteFeatureBlock()
        block.fit(sample_df, ctx)
        block.fit(sample_df, ctx)
        assert block.is_fitted is True


class TestFeatureContext:
    """Tests for FeatureContext dataclass."""

    def test_stores_config(self):
        ctx = FeatureContext(config={"key": "value"}, random_state=0)
        assert ctx.config == {"key": "value"}

    def test_stores_random_state(self):
        ctx = FeatureContext(config={}, random_state=42)
        assert ctx.random_state == 42

    def test_empty_config(self):
        ctx = FeatureContext(config={}, random_state=0)
        assert ctx.config == {}

    def test_equality(self):
        ctx1 = FeatureContext(config={"a": 1}, random_state=42)
        ctx2 = FeatureContext(config={"a": 1}, random_state=42)
        assert ctx1 == ctx2


class TestFeatureOutput:
    """Tests for FeatureOutput dataclass."""

    def test_stores_data(self):
        df = pd.DataFrame({"a": [1]})
        output = FeatureOutput(data=df)
        pd.testing.assert_frame_equal(output.data, df)

    def test_default_feature_names_empty(self):
        output = FeatureOutput(data=pd.DataFrame())
        assert output.feature_names == []

    def test_default_metadata_empty(self):
        output = FeatureOutput(data=pd.DataFrame())
        assert output.metadata == {}

    def test_custom_feature_names(self):
        output = FeatureOutput(
            data=pd.DataFrame(),
            feature_names=["a", "b", "c"],
        )
        assert output.feature_names == ["a", "b", "c"]

    def test_custom_metadata(self):
        output = FeatureOutput(
            data=pd.DataFrame(),
            metadata={"block": "test", "version": 1},
        )
        assert output.metadata == {"block": "test", "version": 1}
