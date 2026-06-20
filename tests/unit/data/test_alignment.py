"""Tests for keyed split/feature row alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.data.alignment import join_splits_with_features


def _make_frames(n: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_df = pd.DataFrame(
        {
            "original_row_id": np.arange(100, 100 + n),
            "Artist": [f"artist_{i % 3}" for i in range(n)],
            "User_Score": np.linspace(60, 90, n),
        }
    )
    features_df = pd.DataFrame(
        {
            "original_row_id": np.arange(100, 100 + n),
            "feature_a": np.arange(n, dtype=float),
            "n_reviews": np.full(n, 25),
        }
    )
    return split_df, features_df


class TestKeyedJoin:
    def test_aligned_frames_join_cleanly(self):
        split_df, features_df = _make_frames()
        joined = join_splits_with_features(split_df, features_df)
        assert list(joined["feature_a"]) == list(range(6))
        assert "original_row_id" in joined.columns
        assert joined.index.equals(split_df.index)

    def test_shuffled_features_realigned_by_key(self):
        """Row order of the features frame must not matter."""
        split_df, features_df = _make_frames()
        shuffled = features_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        joined = join_splits_with_features(split_df, shuffled)
        # feature_a was constructed as (row_id - 100), so alignment is provable.
        expected = (joined["original_row_id"] - 100).astype(float)
        assert np.allclose(joined["feature_a"], expected)

    def test_key_set_mismatch_raises(self):
        split_df, features_df = _make_frames()
        features_df.loc[0, "original_row_id"] = 999
        with pytest.raises(ValueError, match="key sets differ"):
            join_splits_with_features(split_df, features_df)

    def test_duplicate_keys_raise(self):
        split_df, features_df = _make_frames()
        features_df.loc[1, "original_row_id"] = features_df.loc[0, "original_row_id"]
        with pytest.raises(ValueError, match="duplicate"):
            join_splits_with_features(split_df, features_df)

    def test_row_count_mismatch_raises(self):
        split_df, features_df = _make_frames()
        with pytest.raises(ValueError, match="row count mismatch"):
            join_splits_with_features(split_df, features_df.iloc[:-1])

    def test_overlap_columns_keep_features_version(self):
        split_df, features_df = _make_frames()
        split_df["feature_a"] = -1.0
        joined = join_splits_with_features(split_df, features_df)
        assert np.allclose(joined["feature_a"], np.arange(6, dtype=float))


class TestLegacyFallback:
    def test_positional_join_when_key_missing(self):
        split_df, features_df = _make_frames()
        features_df = features_df.drop(columns=["original_row_id"])
        joined = join_splits_with_features(split_df, features_df)
        assert "feature_a" in joined.columns
        assert len(joined) == len(split_df)

    def test_index_mismatch_without_key_raises(self):
        split_df, features_df = _make_frames()
        features_df = features_df.drop(columns=["original_row_id"])
        features_df.index = features_df.index + 1
        with pytest.raises(ValueError, match="different indices"):
            join_splits_with_features(split_df, features_df)
