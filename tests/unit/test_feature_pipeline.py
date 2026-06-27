"""Tests for FeaturePipeline class proving fit/transform separation prevents leakage."""

from typing import ClassVar

import pandas as pd
import pytest

from panelcast.features.base import BaseFeatureBlock, FeatureContext, FeatureOutput
from panelcast.features.errors import NotFittedError
from panelcast.features.pipeline import (
    FeaturePipeline,
    build_blocks_from_config,
)


class MockTrackingBlock(BaseFeatureBlock):
    """A feature block that tracks fit/transform calls for testing."""

    def __init__(self, name: str = "testable", requires: list[str] | None = None) -> None:
        super().__init__()
        self.name = name
        self.requires = requires or []
        self.fit_called = False
        self.fit_data_hash = None
        self.transform_called = False
        self.transform_data_hash = None

    def fit(self, df: pd.DataFrame, ctx: FeatureContext) -> "MockTrackingBlock":
        """Fit and record that fit was called."""
        super().fit(df, ctx)
        self.fit_called = True
        self.fit_data_hash = hash(tuple(df.index.tolist()))
        return self

    def transform(self, df: pd.DataFrame, ctx: FeatureContext) -> FeatureOutput:
        """Transform and record that transform was called."""
        self._check_is_fitted()
        self.transform_called = True
        self.transform_data_hash = hash(tuple(df.index.tolist()))
        return FeatureOutput(
            data=pd.DataFrame({"test_feature": [1] * len(df)}, index=df.index),
            feature_names=["test_feature"],
            metadata={"block": self.name},
        )


class MeanTrackingBlock(BaseFeatureBlock):
    """A block that learns mean during fit and returns it during transform."""

    name = "mean_tracking"
    requires: ClassVar[list[str]] = []

    def __init__(self) -> None:
        super().__init__()
        self._fitted_mean: float | None = None

    def fit(self, df: pd.DataFrame, ctx: FeatureContext) -> "MeanTrackingBlock":
        """Learn mean from training data."""
        super().fit(df, ctx)
        self._fitted_mean = df["value"].mean()
        return self

    def transform(self, df: pd.DataFrame, ctx: FeatureContext) -> FeatureOutput:
        """Return the fitted mean (not recomputed from transform data)."""
        self._check_is_fitted()
        # Return fitted mean, not current data mean
        return FeatureOutput(
            data=pd.DataFrame({"learned_mean": [self._fitted_mean] * len(df)}, index=df.index),
            feature_names=["learned_mean"],
            metadata={"fitted_mean": self._fitted_mean},
        )


class TestFeaturePipelineBasic:
    """Basic FeaturePipeline functionality tests."""

    def test_feature_pipeline_runs(self):
        """Test that pipeline can run with configured blocks."""
        config = {"features": {"blocks": [{"name": "album_type", "params": {}}]}}
        df = pd.DataFrame({"Artist": ["a"], "Album_Type": ["Album"]})
        ctx = FeatureContext(config=config, random_state=0)
        blocks = build_blocks_from_config(config)
        pipeline = FeaturePipeline(blocks)
        out = pipeline.fit_transform(df, ctx)
        assert isinstance(out, FeatureOutput)
        assert out.data.index.equals(df.index)

    def test_pipeline_fit_sets_fitted_state(self):
        """Test that fit() sets is_fitted to True."""
        block = MockTrackingBlock()
        pipeline = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1, 2, 3]})

        assert pipeline.is_fitted is False
        pipeline.fit(df, ctx)
        assert pipeline.is_fitted is True

    def test_pipeline_is_fitted_false_initially(self):
        """Test that is_fitted is False before fit() is called."""
        block = MockTrackingBlock()
        pipeline = FeaturePipeline([block])
        assert pipeline.is_fitted is False


class TestPipelineFitTransformSeparation:
    """Tests proving pipeline properly separates fit and transform."""

    def test_pipeline_transform_before_fit_raises_error(self):
        """Test that transform() before fit() raises NotFittedError."""
        block = MockTrackingBlock()
        pipeline = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(NotFittedError) as exc_info:
            pipeline.transform(df, ctx)

        assert "FeaturePipeline has not been fitted" in str(exc_info.value)
        assert "fit" in str(exc_info.value)

    def test_pipeline_fit_only_uses_train_data(self):
        """Test that fit() only sees training data."""
        block = MockTrackingBlock()
        pipeline = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)

        train_df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
        test_df = pd.DataFrame({"a": [10, 20, 30]}, index=[100, 101, 102])

        # Fit on train
        pipeline.fit(train_df, ctx)

        # Block should have been fitted on train_df
        assert block.fit_called is True
        assert block.fit_data_hash == hash(tuple(train_df.index.tolist()))

        # Transform on test - should NOT call fit again
        pipeline.transform(test_df, ctx)

        # Fit hash should still be from train_df
        assert block.fit_data_hash == hash(tuple(train_df.index.tolist()))

    def test_pipeline_transform_uses_fitted_state(self):
        """Test that transform uses fitted state, not recomputed values."""
        block = MeanTrackingBlock()
        pipeline = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)

        # Train has mean = 10
        train_df = pd.DataFrame({"value": [5, 10, 15]})
        # Test has mean = 100
        test_df = pd.DataFrame({"value": [50, 100, 150]})

        # Fit on train (learns mean = 10)
        pipeline.fit(train_df, ctx)

        # Transform test data
        output = pipeline.transform(test_df, ctx)

        # Output should use fitted mean (10), not test mean (100)
        assert output.data["learned_mean"].iloc[0] == 10.0
        assert output.metadata["blocks"][0]["fitted_mean"] == 10.0


class TestPipelineDependencies:
    """Tests for block dependency handling."""

    def test_pipeline_blocks_fitted_in_dependency_order(self):
        """Test that blocks are fitted in dependency order."""
        block_a = MockTrackingBlock(name="block_a", requires=[])
        block_b = MockTrackingBlock(name="block_b", requires=["block_a"])
        block_c = MockTrackingBlock(name="block_c", requires=["block_b"])

        pipeline = FeaturePipeline([block_a, block_b, block_c])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1, 2, 3]})

        pipeline.fit(df, ctx)

        # All blocks should be fitted
        assert block_a.is_fitted is True
        assert block_b.is_fitted is True
        assert block_c.is_fitted is True

    def test_pipeline_missing_dependency_raises_error(self):
        """Test that missing dependencies raise ValueError."""
        block = MockTrackingBlock(name="dependent", requires=["nonexistent"])
        pipeline = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(ValueError) as exc_info:
            pipeline.fit(df, ctx)

        assert "missing dependencies" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)


class TestPipelineFitTransform:
    """Tests for fit_transform convenience method."""

    def test_pipeline_fit_transform_combines_operations(self):
        """Test that fit_transform fits and transforms in one step."""
        block = MockTrackingBlock()
        pipeline = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1, 2, 3]})

        output = pipeline.fit_transform(df, ctx)

        assert pipeline.is_fitted is True
        assert isinstance(output, FeatureOutput)
        assert block.fit_called is True
        assert block.transform_called is True

    def test_pipeline_fit_transform_returns_feature_output(self):
        """Test that fit_transform returns FeatureOutput with correct structure."""
        block = MockTrackingBlock()
        pipeline = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1, 2, 3]})

        output = pipeline.fit_transform(df, ctx)

        assert isinstance(output, FeatureOutput)
        assert output.data is not None
        assert "test_feature" in output.feature_names


# === COMMENTED OUT: 2026-01-21 ===
# run_feature_blocks was removed from pipeline.py (deprecated).
# Keeping tests as reference per project decision (CONTEXT.md).
# See .backup/dead_code/run_feature_blocks.py for full backup.
#
# class TestDeprecatedRunFeatureBlocks:
#     """Tests for deprecated run_feature_blocks function."""
#
#     def test_deprecated_run_feature_blocks_warns(self):
#         """Test that run_feature_blocks emits DeprecationWarning."""
#         config = {"features": {"blocks": [{"name": "core_numeric", "params": {}}]}}
#         df = pd.DataFrame({"Artist": ["a"], "Year": [2000]})
#         ctx = FeatureContext(config=config, random_state=0)
#         blocks = build_blocks_from_config(config)
#
#         with pytest.warns(DeprecationWarning) as record:
#             run_feature_blocks(df, ctx, blocks)
#
#         assert len(record) == 1
#         assert "deprecated" in str(record[0].message).lower()
#         assert "FeaturePipeline" in str(record[0].message)
#
#     def test_deprecated_run_feature_blocks_still_works(self):
#         """Test that deprecated function still produces output."""
#         config = {"features": {"blocks": [{"name": "core_numeric", "params": {}}]}}
#         df = pd.DataFrame({"Artist": ["a"], "Year": [2000]})
#         ctx = FeatureContext(config=config, random_state=0)
#         blocks = build_blocks_from_config(config)
#
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", DeprecationWarning)
#             out = run_feature_blocks(df, ctx, blocks)
#
#         assert isinstance(out, FeatureOutput)
#         assert out.data.index.equals(df.index)


class TestPipelineMultipleBlocks:
    """Tests for pipelines with multiple blocks."""

    def test_pipeline_concatenates_block_outputs(self):
        """Test that pipeline correctly concatenates outputs from multiple blocks."""
        block1 = MockTrackingBlock(name="block1")
        block2 = MockTrackingBlock(name="block2")
        pipeline = FeaturePipeline([block1, block2])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1, 2, 3]})

        output = pipeline.fit_transform(df, ctx)

        # Should have features from both blocks
        assert output.data.shape[0] == 3  # Same number of rows
        assert len(output.metadata["blocks"]) == 2

    def test_pipeline_empty_blocks_returns_empty_dataframe(self):
        """Test that pipeline with no blocks returns empty DataFrame."""
        pipeline = FeaturePipeline([])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1, 2, 3]})

        output = pipeline.fit_transform(df, ctx)

        assert output.data.shape == (3, 0)  # Same rows, no columns
        assert output.feature_names == []


# ---------------------------------------------------------------------------
# Additional pipeline tests
# ---------------------------------------------------------------------------


class TestPipelineFitReturnsSelf:
    """Test that fit returns self for method chaining."""

    def test_fit_returns_self(self):
        block = MockTrackingBlock()
        pipeline = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1]})
        result = pipeline.fit(df, ctx)
        assert result is pipeline


class TestPipelineCheckIsFitted:
    """Test _check_is_fitted helper."""

    def test_unfitted_pipeline_check_raises(self):
        pipeline = FeaturePipeline([])
        with pytest.raises(NotFittedError, match="FeaturePipeline has not been fitted"):
            pipeline._check_is_fitted()

    def test_fitted_pipeline_check_does_not_raise(self):
        pipeline = FeaturePipeline([])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1]})
        pipeline.fit(df, ctx)
        pipeline._check_is_fitted()  # Should not raise


class TestPipelineOutputMetadata:
    """Test metadata structure in pipeline output."""

    def test_metadata_has_blocks_key(self):
        block = MockTrackingBlock()
        pipeline = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1]})
        output = pipeline.fit_transform(df, ctx)
        assert "blocks" in output.metadata

    def test_metadata_blocks_list_length_matches(self):
        block1 = MockTrackingBlock(name="b1")
        block2 = MockTrackingBlock(name="b2")
        block3 = MockTrackingBlock(name="b3")
        pipeline = FeaturePipeline([block1, block2, block3])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1]})
        output = pipeline.fit_transform(df, ctx)
        assert len(output.metadata["blocks"]) == 3


class TestPipelineIndexPreservation:
    """Test that pipeline preserves DataFrame index."""

    def test_preserves_custom_index(self):
        block = MockTrackingBlock()
        pipeline = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[100, 200, 300])
        output = pipeline.fit_transform(df, ctx)
        assert list(output.data.index) == [100, 200, 300]

    def test_preserves_string_index(self):
        block = MockTrackingBlock()
        pipeline = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        output = pipeline.fit_transform(df, ctx)
        assert list(output.data.index) == ["x", "y"]


class TestPipelineFitTransformSeparation2:
    """Additional fit/transform separation tests."""

    def test_fit_transform_produces_same_result_as_separate(self):
        block = MockTrackingBlock()
        pipeline1 = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)
        df = pd.DataFrame({"a": [1, 2, 3]})
        output1 = pipeline1.fit_transform(df, ctx)

        block2 = MockTrackingBlock()
        pipeline2 = FeaturePipeline([block2])
        pipeline2.fit(df, ctx)
        output2 = pipeline2.transform(df, ctx)

        pd.testing.assert_frame_equal(output1.data, output2.data)

    def test_transform_on_different_data_after_fit(self):
        block = MockTrackingBlock()
        pipeline = FeaturePipeline([block])
        ctx = FeatureContext(config={}, random_state=0)
        train_df = pd.DataFrame({"a": [1, 2, 3]})
        test_df = pd.DataFrame({"a": [10, 20]}, index=[5, 6])
        pipeline.fit(train_df, ctx)
        output = pipeline.transform(test_df, ctx)
        assert list(output.data.index) == [5, 6]
        assert output.data.shape[0] == 2


class TestBuildBlocksFromConfig:
    """Tests for build_blocks_from_config helper function."""

    def test_returns_list(self):
        config = {"features": {"blocks": [{"name": "temporal", "params": {}}]}}
        blocks = build_blocks_from_config(config)
        assert isinstance(blocks, list)
        assert len(blocks) == 1

    def test_empty_config_returns_empty(self):
        config = {}
        blocks = build_blocks_from_config(config)
        assert blocks == []

    def test_multiple_blocks(self):
        config = {
            "features": {
                "blocks": [
                    {"name": "temporal", "params": {}},
                    {"name": "album_type", "params": {}},
                ]
            }
        }
        blocks = build_blocks_from_config(config)
        assert len(blocks) == 2
        assert blocks[0].name == "temporal"
        assert blocks[1].name == "album_type"
