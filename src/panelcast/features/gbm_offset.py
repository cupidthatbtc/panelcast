"""GBM stacking offset feature block (#76 / #86).

Adds a single ``gbm_offset`` column: a gradient-boosted prediction of the
target from the other blocks' feature outputs, entering the Bayesian mean as
one more covariate. Train rows get out-of-fold predictions (the model that
scored a row never saw its target); held-out rows get the full-train model.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

from panelcast.data.alignment import ROW_ID_COL

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput

FEATURE_NAME = "gbm_offset"


class GbmOffsetBlock(BaseFeatureBlock):
    """Stacked-GBM offset over the other feature blocks' outputs.

    The block holds references to the already-constructed base blocks of its
    pipeline (which the pipeline fits before this one — enforced through
    ``requires``) and re-runs their transforms to assemble its input matrix,
    so its inputs are exactly the leakage-safe features the model itself sees.
    The GBM never persists: it lives only for the build_features invocation
    that fits and transforms every split.
    """

    name: ClassVar[str] = FEATURE_NAME

    def __init__(
        self,
        base_blocks: list[BaseFeatureBlock],
        *,
        target_col: str,
        entity_col: str | None = None,
        n_splits: int = 5,
        random_state: int = 0,
    ) -> None:
        super().__init__(params=None)
        if not base_blocks:
            raise ValueError("gbm_offset requires at least one base feature block.")
        self.base_blocks = list(base_blocks)
        self.requires = [b.name for b in self.base_blocks]
        self.target_col = target_col
        self.entity_col = entity_col
        self.n_splits = n_splits
        self.random_state = random_state
        self.required_columns = [ROW_ID_COL, target_col]

    def _features(self, df: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        frames = [b.transform(df, ctx).data for b in self.base_blocks]
        frames = [f for f in frames if f is not None]
        X = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)
        X = X.select_dtypes(include=np.number)
        if hasattr(self, "_feature_cols_"):
            # Same columns and order as fit; unseen columns become NaN, which
            # the histogram GBM handles natively.
            X = X.reindex(columns=self._feature_cols_)
        return X

    def fit(self, df: pd.DataFrame, ctx: FeatureContext) -> GbmOffsetBlock:
        self.validate_columns(df)
        X = self._features(df, ctx)
        self._feature_cols_ = list(X.columns)
        y = pd.to_numeric(df[self.target_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            n_bad = int((~np.isfinite(y)).sum())
            raise ValueError(
                f"gbm_offset: training target '{self.target_col}' has {n_bad} "
                "non-finite values; the block must fit on fully labeled train rows."
            )

        seed = int(getattr(ctx, "random_state", self.random_state))
        self._full_model_ = HistGradientBoostingRegressor(random_state=seed).fit(X, y)

        oof = self._full_model_.predict(X)
        # Group out-of-fold folds by entity so a train row's offset never
        # conditions on the same entity's other (chronologically later) targets;
        # held-out rows only ever see the past-only full-train model, so the
        # covariate would otherwise carry more information in train than in test.
        groups = None
        if self.entity_col is not None and self.entity_col in df.columns:
            groups = df[self.entity_col].to_numpy()
        n_splits = min(self.n_splits, len(df))
        if groups is not None:
            n_splits = min(n_splits, int(pd.unique(groups).size))
        if n_splits >= 2:
            if groups is not None:
                splits = GroupKFold(n_splits=n_splits).split(X, y, groups)
            else:
                splits = KFold(
                    n_splits=n_splits, shuffle=True, random_state=seed
                ).split(X)
            for fit_idx, held_idx in splits:
                fold_model = HistGradientBoostingRegressor(random_state=seed)
                fold_model.fit(X.iloc[fit_idx], y[fit_idx])
                oof[held_idx] = fold_model.predict(X.iloc[held_idx])
        self._oof_by_row_id_ = dict(
            zip(df[ROW_ID_COL].astype(np.int64), oof.astype(float), strict=True)
        )
        self._fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, ctx: FeatureContext) -> FeatureOutput:
        self._check_is_fitted()
        if ROW_ID_COL not in df.columns:
            raise ValueError(
                f"gbm_offset: '{ROW_ID_COL}' missing from transform input; the "
                "out-of-fold substitution for train rows needs row identity."
            )
        X = self._features(df, ctx)
        preds = self._full_model_.predict(X).astype(float)
        row_ids = df[ROW_ID_COL].astype(np.int64).to_numpy()
        values = np.array(
            [self._oof_by_row_id_.get(rid, pred) for rid, pred in zip(row_ids, preds)]
        )
        data = pd.DataFrame({FEATURE_NAME: values}, index=df.index)
        return FeatureOutput(
            data=data,
            feature_names=[FEATURE_NAME],
            metadata={
                "block": self.name,
                "n_input_features": len(self._feature_cols_),
                "n_oof_rows": len(self._oof_by_row_id_),
            },
        )
