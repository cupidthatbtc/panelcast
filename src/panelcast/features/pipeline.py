"""Feature pipeline with fit/transform separation for leakage prevention.

The FeaturePipeline class orchestrates multiple feature blocks, ensuring
they are fitted on training data only before transforming any split.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput
from .errors import NotFittedError
from .registry import build_default_registry, parse_feature_specs


class FeaturePipeline:
    """Orchestrates feature blocks with fit/transform separation.

    Ensures all feature blocks are fitted on training data only,
    preventing data leakage when transforming validation/test splits.

    Parameters
    ----------
    blocks : list[BaseFeatureBlock]
        Feature blocks to include in the pipeline.

    Attributes
    ----------
    blocks : list[BaseFeatureBlock]
        The feature blocks in this pipeline.

    Examples
    --------
    >>> blocks = [CoreNumericBlock(), TemporalBlock()]
    >>> pipeline = FeaturePipeline(blocks)
    >>> pipeline.fit(train_df, ctx)
    >>> train_features = pipeline.transform(train_df, ctx)
    >>> val_features = pipeline.transform(val_df, ctx)
    >>> test_features = pipeline.transform(test_df, ctx)
    """

    def __init__(self, blocks: list[BaseFeatureBlock]) -> None:
        self.blocks = blocks

    @property
    def is_fitted(self) -> bool:
        """Check if pipeline has been fitted.

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
            If the pipeline has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(
                "This FeaturePipeline has not been fitted yet. "
                "Call 'fit' with training data before using 'transform'."
            )

    def fit(self, train_df: pd.DataFrame, ctx: FeatureContext) -> FeaturePipeline:
        """Fit all blocks on training data only.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data to fit all blocks on.
        ctx : FeatureContext
            Shared context with config and random state.

        Returns
        -------
        FeaturePipeline
            Self, for method chaining.

        Raises
        ------
        ValueError
            If a block has missing dependencies.
        """
        completed: set[str] = set()

        for block in self.blocks:
            # Check dependencies
            missing = [name for name in block.requires if name not in completed]
            if missing:
                raise ValueError(f"Block {block.name} missing dependencies: {missing}")

            block.fit(train_df, ctx)  # Fit ONLY on train
            completed.add(block.name)

        self._fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, ctx: FeatureContext) -> FeatureOutput:
        """Transform any split using fitted blocks.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform (can be train, val, or test).
        ctx : FeatureContext
            Shared context with config and random state.

        Returns
        -------
        FeatureOutput
            Combined output from all blocks.

        Raises
        ------
        NotFittedError
            If fit() has not been called.
        """
        self._check_is_fitted()

        outputs: list[FeatureOutput] = []
        for block in self.blocks:
            outputs.append(block.transform(df, ctx))  # Transform only, no fitting

        # Concatenate outputs
        frames = [out.data for out in outputs if out.data is not None]
        if frames:
            data = pd.concat(frames, axis=1)
        else:
            data = pd.DataFrame(index=df.index)

        metadata = {"blocks": [out.metadata for out in outputs]}
        return FeatureOutput(data=data, feature_names=list(data.columns), metadata=metadata)

    def fit_transform(self, train_df: pd.DataFrame, ctx: FeatureContext) -> FeatureOutput:
        """Fit and transform training data in one step.

        Convenience method that calls fit() then transform().
        Should only be used on training data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data to fit and transform.
        ctx : FeatureContext
            Shared context with config and random state.

        Returns
        -------
        FeatureOutput
            Transformed features with metadata.
        """
        self.fit(train_df, ctx)
        return self.transform(train_df, ctx)


def build_blocks_from_config(
    config: dict[str, Any],
    descriptor=None,
) -> list[BaseFeatureBlock]:
    """Build feature blocks from configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration with feature block specifications.
    descriptor : DatasetDescriptor | None
        Dataset descriptor the registry's generic blocks close over
        (None = AOTY defaults).

    Returns
    -------
    list[BaseFeatureBlock]
        List of configured feature blocks.
    """
    registry = build_default_registry(descriptor)
    specs = parse_feature_specs(config)
    return registry.build_all(specs)
