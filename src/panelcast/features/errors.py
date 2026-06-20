"""Feature block errors and frozen state dataclasses.

Provides exception types for feature processing and immutable dataclasses
for storing learned state from training data (preventing leakage).
"""

from __future__ import annotations

from dataclasses import dataclass, field


class NotFittedError(ValueError, AttributeError):
    """Exception raised when transform is called before fit.

    This error is raised when a feature block's transform() method is called
    before the block has been fitted with training data via fit().

    Inherits from both ValueError and AttributeError to match sklearn's
    convention, allowing it to be caught by either exception type.

    Examples
    --------
    >>> block = SomeFeatureBlock()
    >>> block.transform(df, ctx)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    NotFittedError: This some_feature block has not been fitted yet.
    Call 'fit' with training data before using 'transform'.
    """


@dataclass(frozen=True)
class FittedVocabulary:
    """Immutable vocabulary learned from training data.

    Stores categorical mappings learned during fit() to ensure consistent
    encoding across train/val/test splits without data leakage.

    Parameters
    ----------
    categories : tuple[str, ...]
        Ordered unique values from training data.
    category_to_idx : dict[str, int]
        Mapping from category value to integer index.
    unknown_idx : int, default=-1
        Index to use for values not seen during training.

    Examples
    --------
    >>> vocab = FittedVocabulary(
    ...     categories=("A", "B", "C"),
    ...     category_to_idx={"A": 0, "B": 1, "C": 2},
    ...     unknown_idx=-1
    ... )
    >>> vocab.encode(["A", "D"])
    [0, -1]
    """

    categories: tuple[str, ...]
    category_to_idx: dict[str, int] = field(default_factory=dict)
    unknown_idx: int = -1

    def encode(self, values: list[str]) -> list[int]:
        """Encode values using frozen vocabulary.

        Parameters
        ----------
        values : list[str]
            List of category values to encode.

        Returns
        -------
        list[int]
            List of integer indices. Unknown values map to unknown_idx.
        """
        return [self.category_to_idx.get(v, self.unknown_idx) for v in values]


@dataclass(frozen=True)
class FittedStatistics:
    """Immutable statistics learned from training data.

    Stores numeric statistics computed during fit() to ensure consistent
    transformations across train/val/test splits without data leakage.

    Parameters
    ----------
    mean : float
        Mean value from training data.
    std : float
        Standard deviation from training data.
    min_val : float
        Minimum value from training data.
    max_val : float
        Maximum value from training data.

    Examples
    --------
    >>> stats = FittedStatistics(mean=50.0, std=10.0, min_val=20.0, max_val=80.0)
    >>> stats.mean
    50.0
    """

    mean: float
    std: float
    min_val: float
    max_val: float
