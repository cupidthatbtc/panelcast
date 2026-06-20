"""Album type feature block with one-hot encoding.

Produces one-hot encoded columns for album types (Album, EP, Mixtape, Compilation)
using a frozen vocabulary learned from training data.
"""

from __future__ import annotations

import pandas as pd

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput
from .errors import FittedVocabulary


class AlbumTypeBlock(BaseFeatureBlock):
    """Feature block for one-hot encoding album types.

    Learns vocabulary of album types from training data during fit()
    and produces one-hot encoded columns during transform().

    Required columns: Album_Type

    Features computed (depending on training data):
        - is_album: 1 if Album type, 0 otherwise
        - is_ep: 1 if EP type, 0 otherwise
        - is_mixtape: 1 if Mixtape type, 0 otherwise
        - is_compilation: 1 if Compilation type, 0 otherwise

    Missing Album_Type values default to "Album".
    Unknown types (not seen in training) get all zeros.

    Examples
    --------
    >>> block = AlbumTypeBlock()
    >>> block.fit(train_df, ctx)
    >>> block.vocabulary.categories
    ('Album', 'Compilation', 'EP', 'Mixtape')
    >>> output = block.transform(test_df, ctx)
    >>> output.feature_names
    ['is_album', 'is_compilation', 'is_ep', 'is_mixtape']
    """

    name = "album_type"
    requires: list[str] = []
    required_columns: list[str] = ["Album_Type"]

    def fit(self, df, ctx: FeatureContext) -> "AlbumTypeBlock":
        """Fit the album type block on training data.

        Learns vocabulary of unique album types from training data
        and stores as a frozen FittedVocabulary.

        Parameters
        ----------
        df : DataFrame
            Training data with Album_Type column.
        ctx : FeatureContext
            Shared context (unused for this block).

        Returns
        -------
        AlbumTypeBlock
            Self, for method chaining.
        """
        self.validate_columns(df)

        # Learn vocabulary from training data (sorted for determinism)
        unique_types = tuple(sorted(df["Album_Type"].dropna().unique()))
        self._vocabulary_ = FittedVocabulary(
            categories=unique_types,
            category_to_idx={cat: idx for idx, cat in enumerate(unique_types)},
            unknown_idx=len(unique_types),
        )

        self._fitted_ = True
        return self

    @property
    def vocabulary(self) -> FittedVocabulary:
        """Get the fitted vocabulary.

        Returns
        -------
        FittedVocabulary
            The frozen vocabulary learned from training data.

        Raises
        ------
        NotFittedError
            If fit() has not been called.
        """
        self._check_is_fitted()
        return self._vocabulary_

    def transform(self, df, ctx: FeatureContext) -> FeatureOutput:
        """Transform data to compute one-hot encoded album types.

        Parameters
        ----------
        df : DataFrame
            Data to transform (train, val, or test).
        ctx : FeatureContext
            Shared context (unused for this block).

        Returns
        -------
        FeatureOutput
            DataFrame with one-hot columns for each album type.

        Raises
        ------
        NotFittedError
            If fit() has not been called.
        """
        self._check_is_fitted()

        # Fill missing Album_Type with "Album" (default)
        album_types = df["Album_Type"].fillna("Album")

        # Create one-hot columns for each category in vocabulary
        data = pd.DataFrame(index=df.index)
        for cat in self._vocabulary_.categories:
            col_name = f"is_{cat.lower()}"
            data[col_name] = (album_types == cat).astype(int)

        # Unknown types (not in vocabulary) get all zeros automatically
        # since they won't match any category in the loop

        feature_names = list(data.columns)
        return FeatureOutput(
            data=data,
            feature_names=feature_names,
            metadata={
                "block": self.name,
                "params": self.params,
                "vocabulary_size": len(self._vocabulary_.categories),
            },
        )
