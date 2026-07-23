import numpy as np
import pandas as pd

from panelcast.data.alignment import ROW_ID_COL
from panelcast.features.base import FeatureContext
from panelcast.features.core import CoreNumericBlock
from panelcast.features.gbm_offset import GbmOffsetBlock
from panelcast.features.pipeline import FeaturePipeline


def test_temporal_oof_never_fits_future_same_entity(monkeypatch):
    import panelcast.features.gbm_offset as module

    frame = pd.DataFrame(
        {
            "x": np.arange(6, dtype=float),
            "target": np.arange(6, dtype=float),
            "entity": ["c", "a", "b", "a", "b", "c"],
            "date": pd.to_datetime(
                [
                    "2019-12-31",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                ]
            ),
            "event": [f"e{i}" for i in range(6)],
            ROW_ID_COL: np.arange(6, dtype=np.int64),
        }
    )
    fits = []
    fold_calls = []

    class SpyRegressor:
        def __init__(self, random_state=None):
            self.mean = 0.0
            self.fit_indices = []

        def fit(self, X, y):
            self.fit_indices = list(X.index)
            fits.append(self.fit_indices)
            self.mean = float(np.mean(y))
            return self

        def predict(self, X):
            fold_calls.append((self.fit_indices, list(X.index)))
            return np.full(len(X), self.mean)

    monkeypatch.setattr(module, "HistGradientBoostingRegressor", SpyRegressor)
    block = GbmOffsetBlock(
        [CoreNumericBlock({"columns": ["x"]})],
        target_col="target",
        entity_col="entity",
        date_col="date",
        event_col="event",
    )
    FeaturePipeline([*block.base_blocks, block]).fit(frame, FeatureContext({}, 7))

    assert len(fits) <= 1 + 2 * block.n_splits
    assert len(fold_calls) == len(block.fold_manifest)
    for (fit_indices, held_indices), record in zip(
        fold_calls, block.fold_manifest, strict=True
    ):
        assert not set(fit_indices) & set(held_indices)
        if record["estimand"] == "cold_start":
            assert not set(frame.loc[fit_indices, "entity"]) & set(
                frame.loc[held_indices, "entity"]
            )
            assert record["entity_overlap"] is False
        else:
            assert frame.loc[fit_indices, "date"].max() < frame.loc[held_indices, "date"].min()
            assert record["max_fit_rank"] < record["min_held_rank"]

    same_date_fit, _ = next(call for call in fold_calls if 3 in call[1])
    assert 1 not in same_date_fit

    assert {record["protocol"] for record in block.fold_manifest} == {
        "entity_aware_temporal_v1"
    }
    assert {record["estimand"] for record in block.fold_manifest} == {
        "cold_start",
        "prospective_within_entity",
    }
    assert {"fit_row_hash", "held_row_hash", "effective_date_cutoff"} <= set(
        block.fold_manifest[-1]
    )
