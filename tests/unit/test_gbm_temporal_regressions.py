import numpy as np
import pandas as pd
import pytest

from panelcast.data.alignment import ROW_ID_COL
from panelcast.features.base import FeatureContext
from panelcast.features.core import CoreNumericBlock
from panelcast.features.gbm_offset import GbmOffsetBlock
from panelcast.features.pipeline import FeaturePipeline


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(8, dtype=float),
            "target": np.arange(8, dtype=float),
            "entity": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "date": [None, "2020-01-01", "2020-01-02", "2020-01-03",
                     "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04"],
            ROW_ID_COL: np.arange(8, dtype=np.int64),
        }
    )


def _pipeline(frame: pd.DataFrame) -> GbmOffsetBlock:
    block = GbmOffsetBlock(
        [CoreNumericBlock({"columns": ["x"]})],
        target_col="target",
        entity_col="entity",
        date_col="date",
        n_splits=3,
    )
    FeaturePipeline([*block.base_blocks, block]).fit(frame, FeatureContext({}, 3))
    return block


def test_temporal_oof_accepts_missing_training_dates_and_bounds_fit_count(monkeypatch):
    import panelcast.features.gbm_offset as module

    count = 0
    real = module.HistGradientBoostingRegressor

    class CountingRegressor(real):
        def fit(self, X, y):
            nonlocal count
            count += 1
            return super().fit(X, y)

    monkeypatch.setattr(module, "HistGradientBoostingRegressor", CountingRegressor)
    block = _pipeline(_frame())
    assert count <= 1 + 2 * block.n_splits
    assert np.isfinite(list(block._oof_by_row_id_.values())).all()
    missing_folds = [record for record in block.fold_manifest if record["held_date_missing"]]
    assert len(missing_folds) == 1
    assert missing_folds[0]["estimand"] == "cold_start"
    assert missing_folds[0]["effective_date_cutoff"] is None
    assert missing_folds[0]["n_fit_missing_dates"] == 0


def test_all_missing_dates_fail_with_the_configured_column():
    frame = _frame()
    frame["date"] = None
    with pytest.raises(ValueError, match="date_col 'date' has no parseable dates"):
        _pipeline(frame)


def test_single_entity_panel_rejects_the_unidentified_cold_start_estimand():
    frame = _frame()
    frame["entity"] = "only"
    frame["date"] = pd.date_range("2020-01-01", periods=len(frame))
    with pytest.raises(ValueError, match="single-entity panel"):
        _pipeline(frame)


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf])
def test_invalid_entity_identity_fails_closed_with_row_id(invalid):
    frame = _frame()
    frame.loc[2, "entity"] = invalid
    with pytest.raises(ValueError, match=r"row_ids \[2\]"):
        _pipeline(frame)
