"""Entity history feature block with leave-one-out computation.

Generalizes the original artist-history block: the entity grouping column,
the sequencing date column, the tie-break event column and the list of
(score column, output prefix) pairs all come from the constructor, so the
block retargets to any domain. Defaults are the AOTY literals; the
:class:`panelcast.features.artist.ArtistHistoryBlock` subclass pins them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from panelcast.data.chronology import normalize_chronology

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput

# AOTY default: (score column, output prefix) pairs.
DEFAULT_SCORE_SPECS: tuple[tuple[str, str], ...] = (
    ("User_Score", "user"),
    ("Critic_Score", "critic"),
)


def _compute_trajectory_slope(scores: pd.Series) -> float:
    """Compute linear slope of prior scores.

    Parameters
    ----------
    scores : pd.Series
        Series of prior album scores (may contain NaN).

    Returns
    -------
    float
        Slope of linear fit, or NaN if fewer than 2 valid scores.
    """
    valid = scores.dropna()
    if len(valid) < 2:
        return np.nan
    x = np.arange(len(valid))
    slope, _ = np.polyfit(x, valid.values, 1)
    return slope


class EntityHistoryBlock(BaseFeatureBlock):
    """Entity history features using leave-one-out expanding windows.

    Computes prior mean, std, count, and trajectory for each entity,
    excluding the current event (LOO pattern via shift()).

    Output features per (score, prefix) spec:
    ``{prefix}_prior_mean``, ``{prefix}_prior_std``, ``{prefix}_prior_count``,
    ``{prefix}_trajectory`` — plus a single ``is_debut`` flag derived from the
    first spec.

    Debut events have NaN prior statistics which are imputed with global
    means learned from training data during fit().
    """

    name = "entity_history"
    requires: list[str] = []

    def __init__(
        self,
        params: dict | None = None,
        *,
        entity_col: str = "Artist",
        date_col: str = "Release_Date_Parsed",
        event_col: str = "Album",
        score_specs: Sequence[tuple[str, str]] = DEFAULT_SCORE_SPECS,
    ) -> None:
        super().__init__(params)
        self.entity_col = entity_col
        self.date_col = date_col
        self.event_col = event_col
        self.score_specs = [tuple(spec) for spec in score_specs]
        if not self.score_specs:
            raise ValueError("EntityHistoryBlock needs at least one (score, prefix) spec.")
        self.required_columns = [
            entity_col,
            date_col,
            *[col for col, _ in self.score_specs],
            event_col,
        ]

    def fit(self, df: pd.DataFrame, ctx: FeatureContext) -> EntityHistoryBlock:
        """Learn global statistics from training data for debut imputation."""
        self.validate_columns(df)

        # Compute global statistics for debut imputation. Stored per prefix
        # and mirrored as _global_{prefix}_mean_/_global_{prefix}_std_
        # attributes (the original AOTY attribute names).
        self._global_means_: dict[str, float] = {}
        self._global_stds_: dict[str, float] = {}
        for score_col, prefix in self.score_specs:
            mean = df[score_col].mean()
            std = df[score_col].std()
            self._global_means_[prefix] = mean
            self._global_stds_[prefix] = std
            setattr(self, f"_global_{prefix}_mean_", mean)
            setattr(self, f"_global_{prefix}_std_", std)

        self._fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, ctx: FeatureContext) -> FeatureOutput:
        """Compute LOO entity history features.

        Raises
        ------
        NotFittedError
            If fit() has not been called.
        """
        self._check_is_fitted()
        self.validate_columns(df)

        df_sorted = normalize_chronology(
            df,
            entity_col=self.entity_col,
            date_col=self.date_col,
            event_col=self.event_col,
        )

        # Compute LOO expanding statistics for each score type
        for score_col, prefix in self.score_specs:
            grp = df_sorted.groupby(self.entity_col, sort=False)[score_col]

            # shift(1) excludes current event, expanding() accumulates prior
            df_sorted[f"{prefix}_prior_mean"] = grp.transform(
                lambda x: x.shift(1).expanding().mean()
            )
            df_sorted[f"{prefix}_prior_std"] = grp.transform(lambda x: x.shift(1).expanding().std())
            df_sorted[f"{prefix}_prior_count"] = grp.transform(
                lambda x: x.shift(1).expanding().count()
            )

            # Compute trajectory slope (requires 2+ prior events)
            df_sorted[f"{prefix}_trajectory"] = grp.transform(
                lambda x: x.shift(1).expanding().apply(_compute_trajectory_slope, raw=False)
            )

        # Mark debuts BEFORE imputation (using the first spec's prior_mean NaN)
        # Note: count() returns 0 for debuts, but mean() returns NaN
        first_prefix = self.score_specs[0][1]
        df_sorted["is_debut"] = df_sorted[f"{first_prefix}_prior_mean"].isna().astype(int)

        # Impute debut NaN values with global statistics from fit()
        for _, prefix in self.score_specs:
            df_sorted[f"{prefix}_prior_mean"] = df_sorted[f"{prefix}_prior_mean"].fillna(
                self._global_means_[prefix]
            )
            df_sorted[f"{prefix}_prior_std"] = df_sorted[f"{prefix}_prior_std"].fillna(
                self._global_stds_[prefix]
            )
            df_sorted[f"{prefix}_prior_count"] = df_sorted[f"{prefix}_prior_count"].fillna(0)
            df_sorted[f"{prefix}_trajectory"] = df_sorted[f"{prefix}_trajectory"].fillna(0)

        # Re-align to original index order
        result = df_sorted.loc[df.index]

        feature_cols = [
            f"{prefix}_{stat}"
            for _, prefix in self.score_specs
            for stat in ("prior_mean", "prior_std", "prior_count", "trajectory")
        ] + ["is_debut"]

        metadata: dict = {"block": self.name}
        for _, prefix in self.score_specs:
            metadata[f"global_{prefix}_mean"] = self._global_means_[prefix]
            metadata[f"global_{prefix}_std"] = self._global_stds_[prefix]

        return FeatureOutput(
            data=result[feature_cols],
            feature_names=feature_cols,
            metadata=metadata,
        )
