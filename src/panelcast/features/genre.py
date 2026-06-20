"""Genre feature block with multi-hot encoding and PCA reduction.

Encodes comma-separated genre strings into multi-hot representation,
then optionally reduces dimensionality via PCA. Vocabulary is learned
from training data only to prevent data leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput


class GenreBlock(BaseFeatureBlock):
    """Genre feature block with multi-hot encoding and optional PCA.

    Parses comma-separated genre strings into a multi-hot representation.
    Genres seen fewer than min_genre_count times in training are excluded.
    Optionally applies PCA to reduce dimensionality.

    Parameters (via self.params)
    ----------------------------
    min_genre_count : int, default=20
        Minimum albums a genre must appear in to be included in vocabulary.
    n_components : int or None, default=30
        Number of PCA components. Set to None to skip PCA.

    Attributes
    ----------
    _genre_vocab_ : tuple[str, ...]
        Sorted genre names in vocabulary (set during fit).
    _genre_to_idx_ : dict[str, int]
        Mapping from genre name to index (set during fit).
    _pca_ : PCA or None
        Fitted PCA transformer (if n_components is set).
    _use_pca_ : bool
        Whether PCA is applied during transform.
    _explained_variance_ratio_ : ndarray or None
        Explained variance ratio from PCA (if used).

    Examples
    --------
    >>> block = GenreBlock({"min_genre_count": 10, "n_components": 20})
    >>> block.fit(train_df, ctx)
    >>> output = block.transform(test_df, ctx)
    >>> output.data.shape[1]  # 20 PCA components
    20
    """

    name = "genre"
    requires: list[str] = []
    required_columns: list[str] = ["Genres"]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._genre_vocab_: tuple[str, ...] = ()
        self._genre_to_idx_: dict[str, int] = {}
        self._pca_: PCA | None = None
        self._use_pca_: bool = False
        self._explained_variance_ratio_: np.ndarray | None = None

    @property
    def min_genre_count(self) -> int:
        """Minimum count for genre to be in vocabulary."""
        return self.params.get("min_genre_count", 20)

    @property
    def n_components(self) -> int | None:
        """Number of PCA components (None to skip PCA)."""
        return self.params.get("n_components", 30)

    def fit(self, df: pd.DataFrame, ctx: FeatureContext) -> "GenreBlock":
        """Learn genre vocabulary and fit PCA from training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with Genres column.
        ctx : FeatureContext
            Shared context with config and random state.

        Returns
        -------
        GenreBlock
            Self, for method chaining.
        """
        self.validate_columns(df)

        # Parse all genres and count frequencies
        genre_counts: dict[str, int] = {}
        for genres_str in df["Genres"].fillna(""):
            if not genres_str:
                continue
            for g in genres_str.split(", "):
                g = g.strip()
                if g:
                    genre_counts[g] = genre_counts.get(g, 0) + 1

        # Filter to genres with sufficient count
        frequent_genres = [g for g, count in genre_counts.items() if count >= self.min_genre_count]

        # Sort for deterministic ordering
        self._genre_vocab_ = tuple(sorted(frequent_genres))
        self._genre_to_idx_ = {g: i for i, g in enumerate(self._genre_vocab_)}

        # Create multi-hot matrix for training data
        X = self._create_multihot(df)

        # Fit PCA if configured and have enough genres
        n_genres = len(self._genre_vocab_)
        if self.n_components is not None and n_genres > 0 and self.n_components < n_genres:
            self._pca_ = PCA(n_components=self.n_components, random_state=ctx.random_state)
            self._pca_.fit(X)
            self._use_pca_ = True
            self._explained_variance_ratio_ = self._pca_.explained_variance_ratio_
        else:
            self._use_pca_ = False
            self._pca_ = None
            self._explained_variance_ratio_ = None

        self._fitted_ = True
        return self

    def _create_multihot(self, df: pd.DataFrame) -> np.ndarray:
        """Create multi-hot encoding matrix from genre strings.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with Genres column.

        Returns
        -------
        np.ndarray
            Multi-hot matrix of shape (n_samples, n_genres).
        """
        n_genres = len(self._genre_vocab_)
        if n_genres == 0:
            return np.zeros((len(df), 0), dtype=np.float32)

        X = np.zeros((len(df), n_genres), dtype=np.float32)
        for i, genres_str in enumerate(df["Genres"].fillna("")):
            if not genres_str:
                continue
            for g in genres_str.split(", "):
                g = g.strip()
                if g in self._genre_to_idx_:
                    X[i, self._genre_to_idx_[g]] = 1.0
        return X

    def transform(self, df: pd.DataFrame, ctx: FeatureContext) -> FeatureOutput:
        """Transform data using fitted vocabulary and PCA.

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

        # Create multi-hot encoding
        X = self._create_multihot(df)

        # Apply PCA if configured
        if self._use_pca_ and self._pca_ is not None:
            X = self._pca_.transform(X)
            feature_names = [f"genre_pca_{i}" for i in range(X.shape[1])]
        else:
            feature_names = [f"genre_{g}" for g in self._genre_vocab_]

        # Create output DataFrame with original index
        data = pd.DataFrame(X, index=df.index, columns=feature_names)

        # Build metadata
        metadata: dict[str, Any] = {
            "block": self.name,
            "params": self.params,
            "n_genres_in_vocab": len(self._genre_vocab_),
            "use_pca": self._use_pca_,
        }
        if self._use_pca_ and self._explained_variance_ratio_ is not None:
            metadata["explained_variance_ratio"] = self._explained_variance_ratio_.tolist()
            metadata["total_explained_variance"] = float(self._explained_variance_ratio_.sum())

        return FeatureOutput(data=data, feature_names=feature_names, metadata=metadata)


# Backwards compatibility alias
GenrePCABlock = GenreBlock
