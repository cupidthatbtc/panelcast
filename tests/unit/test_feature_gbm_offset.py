"""Tests for the stacked-GBM offset feature block (#86)."""

import numpy as np
import pandas as pd
import pytest

from panelcast.data.alignment import ROW_ID_COL
from panelcast.features.base import FeatureContext
from panelcast.features.core import CoreNumericBlock
from panelcast.features.gbm_offset import FEATURE_NAME, GbmOffsetBlock
from panelcast.features.pipeline import FeaturePipeline
from panelcast.pipelines.build_features import (
    _transform_with_train_history,
    get_feature_blocks,
)

_CTX = FeatureContext(config={}, random_state=42)


def _synthetic(n: int, start_row_id: int = 0, seed: int = 0) -> pd.DataFrame:
    """Learnable regression data: y_score = 3*x1 - 2*x2 + noise."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "y_score": 3.0 * x1 - 2.0 * x2 + rng.normal(scale=0.1, size=n),
            ROW_ID_COL: np.arange(start_row_id, start_row_id + n, dtype=np.int64),
        }
    )


def _block() -> GbmOffsetBlock:
    return GbmOffsetBlock(
        [CoreNumericBlock({"columns": ["x1", "x2"]})],
        target_col="y_score",
    )


def _fitted_pipeline(train_df):
    block = _block()
    pipeline = FeaturePipeline([*block.base_blocks, block])
    pipeline.fit(train_df, _CTX)
    return pipeline, block


class TestGbmOffsetBlock:
    def test_train_rows_get_oof_not_full_fit(self):
        """Leakage tripwire: train values must come from out-of-fold models."""
        train_df = _synthetic(200)
        pipeline, block = _fitted_pipeline(train_df)
        out = pipeline.transform(train_df, _CTX).data[FEATURE_NAME].to_numpy()
        X = train_df[["x1", "x2"]].astype(float)
        full_fit = block._full_model_.predict(X)
        assert not np.allclose(out, full_fit)

    def test_held_out_rows_get_full_train_model(self):
        train_df = _synthetic(200)
        test_df = _synthetic(50, start_row_id=1000, seed=1)
        _, block = _fitted_pipeline(train_df)
        out = block.transform(test_df, _CTX).data[FEATURE_NAME].to_numpy()
        full_fit = block._full_model_.predict(test_df[["x1", "x2"]].astype(float))
        np.testing.assert_allclose(out, full_fit)

    def test_deterministic(self):
        train_df = _synthetic(200)
        test_df = _synthetic(50, start_row_id=1000, seed=1)
        outs = []
        for _ in range(2):
            _, block = _fitted_pipeline(train_df)
            outs.append(block.transform(test_df, _CTX).data[FEATURE_NAME].to_numpy())
        np.testing.assert_array_equal(outs[0], outs[1])

    def test_offset_correlates_with_target(self):
        train_df = _synthetic(300)
        test_df = _synthetic(100, start_row_id=1000, seed=1)
        _, block = _fitted_pipeline(train_df)
        out = block.transform(test_df, _CTX).data[FEATURE_NAME]
        assert np.isfinite(out).all()
        assert np.corrcoef(out, test_df["y_score"])[0, 1] > 0.9

    def test_combined_transform_resolves_rows_by_id(self):
        """The masked-concat path: target rows full-fit, train rows OOF."""
        train_df = _synthetic(200)
        test_df = _synthetic(50, start_row_id=1000, seed=1)
        pipeline, block = _fitted_pipeline(train_df)
        test_features = _transform_with_train_history(
            pipeline, train_df, test_df, _CTX, mask_target_score_cols=("y_score",)
        )
        full_fit = block._full_model_.predict(test_df[["x1", "x2"]].astype(float))
        np.testing.assert_allclose(test_features[FEATURE_NAME].to_numpy(), full_fit)
        assert np.isfinite(test_features[FEATURE_NAME]).all()

    def test_missing_target_raises(self):
        train_df = _synthetic(50)
        train_df.loc[3, "y_score"] = np.nan
        block = _block()
        pipeline = FeaturePipeline([*block.base_blocks, block])
        with pytest.raises(ValueError, match="non-finite"):
            pipeline.fit(train_df, _CTX)

    def test_missing_row_id_raises_on_transform(self):
        train_df = _synthetic(50)
        _, block = _fitted_pipeline(train_df)
        with pytest.raises(ValueError, match=ROW_ID_COL):
            block.transform(train_df.drop(columns=[ROW_ID_COL]), _CTX)

    def test_requires_names_base_blocks(self):
        block = _block()
        assert block.requires == ["core_numeric"]


class TestBlockEnablement:
    def test_default_roster_has_no_gbm_block(self):
        names = [b.name for b in get_feature_blocks()]
        assert FEATURE_NAME not in names

    def test_enabled_roster_appends_gbm_block_last(self):
        blocks = get_feature_blocks(gbm_offset=True)
        assert blocks[-1].name == FEATURE_NAME
        base_names = [b.name for b in blocks[:-1]]
        assert blocks[-1].requires == base_names
        assert base_names == [b.name for b in get_feature_blocks()]

    def test_enabled_roster_respects_ablations(self):
        blocks = get_feature_blocks(enable_genre=False, gbm_offset=True)
        assert "genre" not in [b.name for b in blocks]
        assert blocks[-1].name == FEATURE_NAME
