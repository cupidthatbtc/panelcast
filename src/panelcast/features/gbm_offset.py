"""GBM stacking offset feature block (#76 / #86).

Adds a single ``gbm_offset`` column: a gradient-boosted prediction of the
target from the other blocks' feature outputs, entering the Bayesian mean as
one more covariate. Train rows get out-of-fold predictions (the model that
scored a row never saw its target); held-out rows get the full-train model.
"""

from __future__ import annotations

import hashlib
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

from panelcast.data.alignment import ROW_ID_COL
from panelcast.data.chronology import DATE_MISSING_COL, normalize_chronology

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
        date_col: str | None = None,
        event_col: str | None = None,
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
        self.date_col = date_col
        self.event_col = event_col
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

    @staticmethod
    def _row_hash(row_ids: pd.Series) -> str:
        values = ",".join(str(value) for value in sorted(row_ids.astype(int).tolist()))
        return hashlib.sha256(values.encode()).hexdigest()

    def _temporal_oof(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """Score rows with a bounded set of leakage-safe temporal/entity folds."""
        assert self.entity_col is not None and self.date_col is not None
        if self.date_col not in df.columns:
            raise ValueError(f"gbm_offset: configured date_col '{self.date_col}' is absent")

        ordered = normalize_chronology(
            df,
            entity_col=self.entity_col,
            date_col=self.date_col,
            event_col=self.event_col,
        )
        if not bool(ordered[self.date_col].notna().any()):
            raise ValueError(
                f"gbm_offset: date_col '{self.date_col}' has no parseable dates; "
                "temporal OOF cannot establish an admissible history"
            )
        ordered_by_id = ordered.set_index(ROW_ID_COL)
        row_ids = df[ROW_ID_COL].astype(np.int64)
        dates = pd.Series(
            ordered_by_id.loc[row_ids, self.date_col].to_numpy(),
            index=df.index,
        )
        date_missing = pd.Series(
            ordered_by_id.loc[row_ids, DATE_MISSING_COL].to_numpy(dtype="int8"),
            index=df.index,
        )
        rank_by_id = pd.Series(
            np.arange(len(ordered), dtype=np.int64),
            index=ordered[ROW_ID_COL].astype(np.int64),
        )
        rank = pd.Series(row_ids.map(rank_by_id).to_numpy(dtype=np.int64), index=df.index)
        entities = df[self.entity_col]

        first_for_entity = rank.groupby(entities, sort=False).transform("min") == rank
        cold_positions = np.flatnonzero(first_for_entity.to_numpy())
        prospective_positions = np.flatnonzero((~first_for_entity).to_numpy())
        oof = np.full(len(df), np.nan, dtype=float)
        manifest: list[dict[str, object]] = []

        def fit_fold(
            fit_positions: np.ndarray,
            held_positions: np.ndarray,
            estimand: str,
            cutoff: object,
        ) -> None:
            if not len(fit_positions) or not len(held_positions):
                return
            if np.intersect1d(fit_positions, held_positions).size:
                raise AssertionError("gbm_offset OOF fold includes a held row in its fit set")
            model = HistGradientBoostingRegressor(random_state=seed)
            model.fit(X.iloc[fit_positions], y[fit_positions])
            oof[held_positions] = model.predict(X.iloc[held_positions])
            fit_entities = set(entities.iloc[fit_positions])
            held_entities = set(entities.iloc[held_positions])
            overlap_entities = fit_entities & held_entities
            if estimand == "cold_start" and overlap_entities:
                raise AssertionError("gbm_offset cold-start fold has entity overlap")
            min_held_rank = int(rank.iloc[held_positions].min())
            max_fit_rank = int(rank.iloc[fit_positions].max())
            if estimand == "prospective_within_entity" and max_fit_rank >= min_held_rank:
                raise AssertionError("gbm_offset prospective fold includes future observations")
            fit_date_max = dates.iloc[fit_positions].max()
            held_date_min = dates.iloc[held_positions].min()
            if (
                estimand == "prospective_within_entity"
                and not pd.isna(held_date_min)
                and not pd.isna(fit_date_max)
                and fit_date_max >= held_date_min
            ):
                raise AssertionError(
                    "gbm_offset prospective fold includes contemporaneous observations"
                )
            manifest.append(
                {
                    "protocol": "entity_aware_temporal_v1",
                    "estimand": estimand,
                    "held_row_hash": self._row_hash(df.iloc[held_positions][ROW_ID_COL]),
                    "fit_row_hash": self._row_hash(df.iloc[fit_positions][ROW_ID_COL]),
                    "effective_date_cutoff": cutoff,
                    "fit_effective_date_max": (
                        None if pd.isna(fit_date_max) else fit_date_max.isoformat()
                    ),
                    "min_held_rank": min_held_rank,
                    "max_fit_rank": max_fit_rank,
                    "n_fit_rows": int(len(fit_positions)),
                    "n_held_rows": int(len(held_positions)),
                    "n_fit_missing_dates": int(dates.iloc[fit_positions].isna().sum()),
                    "held_date_missing": bool(dates.iloc[held_positions].isna().any()),
                    "entity_overlap": bool(overlap_entities),
                    "entity_overlap_count": len(overlap_entities),
                }
            )

        if len(cold_positions):
            cold_entities = entities.iloc[cold_positions].to_numpy()
            unique_entities = pd.unique(cold_entities)
            n_entity_folds = min(self.n_splits, len(unique_entities))
            if n_entity_folds < 2:
                raise ValueError(
                    "gbm_offset: cold-start OOF requires at least two entities; "
                    "a single-entity panel cannot identify the cold-start estimand"
                )
            for _, held_local in GroupKFold(n_splits=n_entity_folds).split(
                cold_positions, groups=cold_entities
            ):
                held = cold_positions[held_local]
                held_values = set(entities.iloc[held])
                fit = np.flatnonzero((~entities.isin(held_values)).to_numpy())
                fit_fold(fit, held, "cold_start", None)

        if len(prospective_positions):
            prospective_positions = prospective_positions[
                np.argsort(rank.iloc[prospective_positions].to_numpy())
            ]
            dated = prospective_positions[
                date_missing.iloc[prospective_positions].eq(0).to_numpy()
            ]
            undated = prospective_positions[
                date_missing.iloc[prospective_positions].eq(1).to_numpy()
            ]
            dated_fold_limit = self.n_splits - int(bool(len(undated)))
            if len(undated):
                cutoff_rank = int(rank.iloc[undated].min())
                fit = np.flatnonzero((rank < cutoff_rank).to_numpy())
                fit_fold(fit, undated, "prospective_within_entity", None)
            if len(dated):
                for held in np.array_split(dated, min(max(1, dated_fold_limit), len(dated))):
                    cutoff_date = dates.iloc[held].min()
                    fit = np.flatnonzero(
                        ((date_missing == 1) | (dates < cutoff_date)).to_numpy()
                    )
                    fit_fold(
                        fit,
                        held,
                        "prospective_within_entity",
                        cutoff_date.isoformat(),
                    )

        if np.isnan(oof).any():
            missing_ids = df.loc[np.isnan(oof), ROW_ID_COL].astype(int).tolist()
            raise ValueError(f"gbm_offset: no admissible OOF fold for row_ids {missing_ids}")
        self._fold_manifest_ = manifest
        return oof

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

        if self.entity_col is not None and self.entity_col not in df.columns:
            raise ValueError(
                f"gbm_offset: configured entity_col '{self.entity_col}' is absent from "
                "the fit frame; refusing entity-blind fallback."
            )
        if self.entity_col is not None:
            entities = df[self.entity_col]
            invalid = entities.isna()
            invalid |= entities.map(
                lambda value: isinstance(value, (int, float, np.number))
                and not np.isfinite(value)
            )
            if bool(invalid.any()):
                row_ids = df.loc[invalid, ROW_ID_COL].astype(int).tolist()
                raise ValueError(
                    f"gbm_offset: entity_col '{self.entity_col}' has null or non-finite "
                    f"identities for row_ids {row_ids}"
                )

        self._full_model_ = HistGradientBoostingRegressor(random_state=seed).fit(X, y)
        if self.entity_col is not None and self.date_col is not None:
            oof = self._temporal_oof(df, X, y, seed)
        else:
            # Explicit legacy/non-panel migration path. Repository defaults pass
            # entity+date and therefore use entity_aware_temporal_v1.
            oof = self._full_model_.predict(X)
            groups = df[self.entity_col].to_numpy() if self.entity_col is not None else None
            n_splits = min(self.n_splits, len(df))
            if groups is not None:
                n_splits = min(n_splits, int(pd.unique(groups).size))
            if n_splits >= 2:
                splits = (
                    GroupKFold(n_splits=n_splits).split(X, y, groups)
                    if groups is not None
                    else KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X)
                )
                for fit_idx, held_idx in splits:
                    fold_model = HistGradientBoostingRegressor(random_state=seed)
                    fold_model.fit(X.iloc[fit_idx], y[fit_idx])
                    oof[held_idx] = fold_model.predict(X.iloc[held_idx])
            self._fold_manifest_ = [
                {
                    "protocol": "legacy_group_kfold" if groups is not None else "legacy_kfold",
                    "n_fit_rows": int(len(df)),
                }
            ]
        self._oof_by_row_id_ = dict(
            zip(df[ROW_ID_COL].astype(np.int64), oof.astype(float), strict=True)
        )
        self._fitted_ = True
        return self

    @property
    def fold_manifest(self) -> list[dict[str, object]]:
        self._check_is_fitted()
        return [dict(record) for record in self._fold_manifest_]

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
