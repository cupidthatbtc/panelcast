"""Temporal feature block for album career context.

Computes temporal features that capture career trajectory context:
- album_sequence: Sequential album number for artist (1, 2, 3...)
- career_years: Years since artist's first album
- release_gap_days: Days since artist's previous album (0 for debuts)
- release_year: Calendar year for trend capture
- date_risk_ordinal: Risk level of date accuracy (low=0, medium=1, high=2)
- date_missing: 1 where the release date is unknown (unimputed tier-3 rows)
"""

from __future__ import annotations

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput


class TemporalBlock(BaseFeatureBlock):
    """Feature block computing temporal context features.

    This block is stateless - no statistics are learned during fit.
    The fit() method validates required columns and sets fitted state.

    The entity/date/year/event column names come from the constructor
    (defaults are the AOTY literals); output feature names are fixed
    canonical names regardless of domain.

    Default required columns: Artist, Release_Date_Parsed, Year, date_risk, Album

    Features computed:
        - album_sequence: 1-indexed event number within entity
        - career_years: Years since entity's first event
        - release_gap_days: Days since previous event (0 for debuts)
        - release_year: Calendar year of release
        - date_risk_ordinal: Ordinal encoding of date risk level
        - date_missing: 1 where the release date is unknown (unimputed)

    Examples
    --------
    >>> block = TemporalBlock()
    >>> block.fit(train_df, ctx)
    >>> output = block.transform(test_df, ctx)
    >>> output.feature_names[:3]
    ['album_sequence', 'career_years', 'release_gap_days']
    """

    name = "temporal"
    requires: list[str] = []

    def __init__(
        self,
        params: dict | None = None,
        *,
        entity_col: str = "Artist",
        date_col: str = "Release_Date_Parsed",
        year_col: str = "Year",
        event_col: str = "Album",
    ) -> None:
        super().__init__(params)
        self.entity_col = entity_col
        self.date_col = date_col
        self.year_col = year_col
        self.event_col = event_col
        self.required_columns = [
            entity_col,
            date_col,
            year_col,
            "date_risk",
            event_col,
        ]

    def fit(self, df, ctx: FeatureContext) -> TemporalBlock:
        """Fit the temporal block on training data.

        Validates required columns exist. This block is stateless,
        so no statistics are learned from training data.

        Parameters
        ----------
        df : DataFrame
            Training data with required columns.
        ctx : FeatureContext
            Shared context (unused for this stateless block).

        Returns
        -------
        TemporalBlock
            Self, for method chaining.
        """
        self.validate_columns(df)
        self._fitted_ = True
        return self

    def transform(self, df, ctx: FeatureContext) -> FeatureOutput:
        """Transform data to compute temporal features.

        Parameters
        ----------
        df : DataFrame
            Data to transform (train, val, or test).
        ctx : FeatureContext
            Shared context (unused for this block).

        Returns
        -------
        FeatureOutput
            DataFrame with 5 temporal feature columns.

        Raises
        ------
        NotFittedError
            If fit() has not been called.
        """
        self._check_is_fitted()

        # Sort by entity, date, event for deterministic ordering
        # Event as tiebreaker ensures same-date events have consistent order
        df_sorted = df.sort_values([self.entity_col, self.date_col, self.event_col]).copy()

        # Event sequence (1-indexed): cumcount + 1 within entity
        df_sorted["album_sequence"] = df_sorted.groupby(self.entity_col).cumcount() + 1

        # Career length: years since entity's first event
        df_sorted["first_release"] = df_sorted.groupby(self.entity_col)[self.date_col].transform(
            "min"
        )
        df_sorted["career_years"] = (
            df_sorted[self.date_col] - df_sorted["first_release"]
        ).dt.days / 365.25

        # Release gap: days since previous event (0 for debuts)
        df_sorted["prev_release"] = df_sorted.groupby(self.entity_col)[self.date_col].shift(1)
        df_sorted["release_gap_days"] = (
            df_sorted[self.date_col] - df_sorted["prev_release"]
        ).dt.days
        df_sorted["release_gap_days"] = df_sorted["release_gap_days"].fillna(0)

        # Release year for trend capture
        df_sorted["release_year"] = df_sorted[self.date_col].dt.year

        # Date risk as ordinal (low=0, medium=1, high=2)
        risk_map = {"low": 0, "medium": 1, "high": 2}
        df_sorted["date_risk_ordinal"] = df_sorted["date_risk"].map(risk_map).fillna(1)

        # Explicit missingness indicator: career/gap features for these rows
        # are NaN (later zero-filled), so the model needs a flag to separate
        # "truly debut/zero" from "date unknown".
        df_sorted["date_missing"] = df_sorted[self.date_col].isna().astype(int)

        # Re-align to original index before returning
        result = df_sorted.loc[df.index]

        feature_cols = [
            "album_sequence",
            "career_years",
            "release_gap_days",
            "release_year",
            "date_risk_ordinal",
            "date_missing",
        ]

        return FeatureOutput(
            data=result[feature_cols],
            feature_names=feature_cols,
            metadata={"block": self.name, "params": self.params},
        )
