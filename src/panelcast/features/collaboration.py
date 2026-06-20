"""Collaboration feature block for encoding collaboration information.

Passes through and encodes existing collaboration columns from cleaned data:
is_collaboration, num_artists, collab_type.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput
from .errors import FittedVocabulary

# Default collab_type ordering for ordinal encoding
DEFAULT_COLLAB_TYPES = ("solo", "duo", "small_group", "ensemble")


class CollaborationBlock(BaseFeatureBlock):
    """Collaboration feature block for encoding collaboration information.

    Passes through existing collaboration columns and encodes collab_type
    as ordinal values. The vocabulary is learned from training data for
    consistent encoding across splits.

    Required columns: is_collaboration, collab_type, num_artists

    Output features:
        - is_collaboration: binary (0/1)
        - num_artists: count
        - collab_type_ordinal: ordinal encoding (0=solo, 1=duo, 2=small_group, 3=ensemble)

    Parameters (via self.params)
    ----------------------------
    collab_type_order : list[str], optional
        Custom ordering for collab_type encoding.
        Default: ["solo", "duo", "small_group", "ensemble"]

    Attributes
    ----------
    _collab_type_vocab_ : FittedVocabulary
        Frozen vocabulary for collab_type encoding (set during fit).

    Examples
    --------
    >>> block = CollaborationBlock()
    >>> block.fit(train_df, ctx)
    >>> output = block.transform(test_df, ctx)
    >>> output.feature_names
    ['is_collaboration', 'num_artists', 'collab_type_ordinal']
    """

    name: ClassVar[str] = "collaboration"
    requires: ClassVar[list[str]] = []
    required_columns: ClassVar[list[str]] = ["is_collaboration", "collab_type", "num_artists"]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._collab_type_vocab_: FittedVocabulary | None = None

    @property
    def collab_type_order(self) -> tuple[str, ...]:
        """Ordering for collab_type ordinal encoding."""
        custom_order = self.params.get("collab_type_order")
        if custom_order is not None:
            return tuple(custom_order)
        return DEFAULT_COLLAB_TYPES

    def fit(self, df: pd.DataFrame, ctx: FeatureContext) -> "CollaborationBlock":
        """Learn collab_type vocabulary from training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with collaboration columns.
        ctx : FeatureContext
            Shared context with config and random state.

        Returns
        -------
        CollaborationBlock
            Self, for method chaining.
        """
        self.validate_columns(df)

        # Build collab_type vocabulary with defined ordering
        # Use the predefined order, but verify all types in training data are covered
        collab_types_in_data = set(df["collab_type"].dropna().unique())
        ordered_types = list(self.collab_type_order)

        # Add any types from data not in the predefined order (at the end)
        for ct in sorted(collab_types_in_data):
            if ct not in ordered_types:
                ordered_types.append(ct)

        categories = tuple(ordered_types)
        category_to_idx = {ct: i for i, ct in enumerate(categories)}

        self._collab_type_vocab_ = FittedVocabulary(
            categories=categories,
            category_to_idx=category_to_idx,
            unknown_idx=-1,
        )

        self._fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, ctx: FeatureContext) -> FeatureOutput:
        """Transform collaboration columns into features.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform (can be train, val, or test).
        ctx : FeatureContext
            Shared context with config and random state.

        Returns
        -------
        FeatureOutput
            Transformed features with metadata.

        Raises
        ------
        NotFittedError
            If fit() has not been called.
        """
        self._check_is_fitted()

        # Create output DataFrame
        output_data = pd.DataFrame(index=df.index)

        # Pass through is_collaboration as int (0/1)
        output_data["is_collaboration"] = df["is_collaboration"].fillna(False).astype(int)

        # Pass through num_artists
        output_data["num_artists"] = df["num_artists"].fillna(1).astype(int)

        # Encode collab_type as ordinal
        if self._collab_type_vocab_ is not None:
            default_type = self._collab_type_vocab_.categories[0]
            collab_types = df["collab_type"].fillna(default_type).tolist()
            output_data["collab_type_ordinal"] = self._collab_type_vocab_.encode(collab_types)

        feature_names = ["is_collaboration", "num_artists", "collab_type_ordinal"]

        metadata: dict[str, Any] = {
            "block": self.name,
            "params": self.params,
            "collab_type_categories": (
                list(self._collab_type_vocab_.categories)
                if self._collab_type_vocab_ is not None
                else []
            ),
        }

        return FeatureOutput(data=output_data, feature_names=feature_names, metadata=metadata)
