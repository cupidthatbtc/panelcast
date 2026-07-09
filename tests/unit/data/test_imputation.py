"""Missing-covariate treatment (#158): fit/apply of medians + indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.data.imputation import INDICATOR_SUFFIX, apply_imputation, fit_imputation


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, np.nan],  # median of observed = 2.0
            "b": [10.0, 20.0, 30.0, 40.0],  # fully observed
        }
    )


class TestFitImputation:
    def test_medians_recorded_for_every_column(self):
        df = _train_frame()
        _, record = fit_imputation(df, ["a", "b"])
        assert record["medians"] == {"a": 2.0, "b": 25.0}

    def test_indicator_only_for_columns_with_missingness(self):
        df = _train_frame()
        cols, record = fit_imputation(df, ["a", "b"])
        assert record["indicator_cols"] == [f"a{INDICATOR_SUFFIX}"]
        assert cols == ["a", "b", f"a{INDICATOR_SUFFIX}"]
        assert df[f"a{INDICATOR_SUFFIX}"].tolist() == [0.0, 0.0, 0.0, 1.0]
        assert f"b{INDICATOR_SUFFIX}" not in df.columns

    def test_train_frame_filled_with_medians_not_zero(self):
        df = _train_frame()
        fit_imputation(df, ["a", "b"])
        assert df["a"].tolist() == [1.0, 2.0, 3.0, 2.0]

    def test_all_nan_column_gets_zero_median(self):
        df = pd.DataFrame({"a": [np.nan, np.nan]})
        _, record = fit_imputation(df, ["a"])
        assert record["medians"]["a"] == 0.0
        assert df["a"].tolist() == [0.0, 0.0]


class TestCollisionGuard:
    def test_reserved_suffix_in_feature_cols_rejected(self):
        df = pd.DataFrame({"a": [1.0], f"a{INDICATOR_SUFFIX}": [0.0]})
        with pytest.raises(ValueError, match="reserved"):
            fit_imputation(df, ["a", f"a{INDICATOR_SUFFIX}"])


class TestApplyImputation:
    def test_no_record_is_the_legacy_fillna_zero(self):
        df = pd.DataFrame({"a": [1.0, np.nan]})
        apply_imputation(df, ["a"], None)
        assert df["a"].tolist() == [1.0, 0.0]

    def test_gate_on_fills_with_train_medians(self):
        train = _train_frame()
        cols, record = fit_imputation(train, ["a", "b"])
        test = pd.DataFrame({"a": [np.nan, 5.0], "b": [np.nan, 1.0]})
        apply_imputation(test, cols, record)
        # a from its train median; b (train-complete) also from ITS train
        # median, never a statistic of the test frame.
        assert test["a"].tolist() == [2.0, 5.0]
        assert test["b"].tolist() == [25.0, 1.0]
        assert test[f"a{INDICATOR_SUFFIX}"].tolist() == [1.0, 0.0]

    def test_recorded_indicator_materialized_even_when_test_is_complete(self):
        train = _train_frame()
        cols, record = fit_imputation(train, ["a", "b"])
        test = pd.DataFrame({"a": [4.0], "b": [2.0]})
        apply_imputation(test, cols, record)
        assert test[f"a{INDICATOR_SUFFIX}"].tolist() == [0.0]

    def test_indicator_for_absent_base_column_defaults_to_zero(self):
        record = {"medians": {"a": 2.0}, "indicator_cols": [f"a{INDICATOR_SUFFIX}"]}
        test = pd.DataFrame({"other": [1.0]})
        apply_imputation(test, [f"a{INDICATOR_SUFFIX}"], record)
        assert test[f"a{INDICATOR_SUFFIX}"].tolist() == [0.0]

    def test_unrecorded_column_falls_back_to_zero(self):
        record = {"medians": {}, "indicator_cols": []}
        test = pd.DataFrame({"new_col": [np.nan, 3.0]})
        apply_imputation(test, ["new_col"], record)
        assert test["new_col"].tolist() == [0.0, 3.0]
