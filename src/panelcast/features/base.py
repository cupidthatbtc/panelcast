"""Feature block interfaces with fit/transform state tracking.

Keep feature logic modular and config-driven so new blocks can be added safely.
The fit/transform pattern prevents data leakage by ensuring blocks are fitted
on training data only before transforming any split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .errors import NotFittedError


@dataclass
class FeatureContext:
    """Shared context for feature blocks."""

    config: dict[str, Any]
    random_state: int


@dataclass
class FeatureOutput:
    """Output container for a feature block."""

    data: Any
    feature_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseFeatureBlock:
    """Base class for feature blocks with fit/transform state tracking.

    Provides the foundation for the fit/transform pattern that prevents
    data leakage. Blocks must be fitted on training data before transform()
    can be called.

    Attributes
    ----------
    name : str
        Identifier for this block type.
    requires : list[str]
        Names of blocks that must be fitted before this one.
    required_columns : list[str]
        DataFrame columns required for this block to work.

    Examples
    --------
    >>> block = MyFeatureBlock()
    >>> block.is_fitted
    False
    >>> block.fit(train_df, ctx)
    >>> block.is_fitted
    True
    >>> output = block.transform(test_df, ctx)
    """

    name: str = "base"
    requires: list[str] = []
    required_columns: list[str] = []

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @property
    def is_fitted(self) -> bool:
        """Check if block has been fitted.

        Returns
        -------
        bool
            True if fit() has been called, False otherwise.
        """
        return getattr(self, "_fitted_", False)

    def _check_is_fitted(self) -> None:
        """Raise NotFittedError if transform called before fit.

        Raises
        ------
        NotFittedError
            If the block has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This {self.name} block has not been fitted yet. "
                f"Call 'fit' with training data before using 'transform'."
            )

    def validate_columns(self, df) -> None:
        """Validate required columns exist in DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame to validate.

        Raises
        ------
        ValueError
            If required columns are missing from the DataFrame.
        """
        if not self.required_columns:
            return
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"{self.name} missing required columns: {missing}")

    def fit(self, df, ctx: FeatureContext) -> "BaseFeatureBlock":
        """Fit the block on training data.

        Subclasses should override this method to learn statistics,
        vocabularies, or other parameters from training data. Always
        call super().fit(df, ctx) first.

        Parameters
        ----------
        df : DataFrame
            Training data to fit on.
        ctx : FeatureContext
            Shared context with config and random state.

        Returns
        -------
        BaseFeatureBlock
            Self, for method chaining.
        """
        self.validate_columns(df)
        self._fitted_ = True
        return self

    def transform(self, df, ctx: FeatureContext) -> FeatureOutput:
        """Transform data using fitted state.

        Subclasses must override this method. Always call
        self._check_is_fitted() first to ensure fit() was called.

        Parameters
        ----------
        df : DataFrame
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
        NotImplementedError
            In base class (must be overridden).
        """
        self._check_is_fitted()
        raise NotImplementedError

    def fit_transform(self, df, ctx: FeatureContext) -> FeatureOutput:
        """Fit and transform in one step (for training data only).

        Convenience method that calls fit() then transform().
        Should only be used on training data.

        Parameters
        ----------
        df : DataFrame
            Training data to fit and transform.
        ctx : FeatureContext
            Shared context with config and random state.

        Returns
        -------
        FeatureOutput
            Transformed features with metadata.
        """
        self.fit(df, ctx)
        return self.transform(df, ctx)


class FeatureBlock(Protocol):
    """Protocol interface for feature blocks.

    Defines the expected interface for feature blocks for type checking.
    """

    name: str
    requires: list[str]

    @property
    def is_fitted(self) -> bool:
        """Check if block has been fitted."""
        ...

    def fit(self, df, ctx: FeatureContext) -> "FeatureBlock":
        """Fit the block on training data."""
        ...

    def transform(self, df, ctx: FeatureContext) -> FeatureOutput:
        """Transform data using fitted state."""
        ...

    def fit_transform(self, df, ctx: FeatureContext) -> FeatureOutput:
        """Fit and transform in one step."""
        ...
