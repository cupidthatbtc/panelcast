"""Core numeric feature block.

Passes through descriptor-mapped numeric columns as model features, with
train-fitted imputation for missing values. This is the pure-YAML feature
surface: a domain lists extra numeric columns in ``raw_column_map`` and adds

    feature_blocks:
      - name: core_numeric
        params:
          columns: [Thrust_Margin, Payload_Fraction]

with no Python required.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput

_IMPUTE_STRATEGIES = ("median", "mean", "zero")


class CoreNumericBlock(BaseFeatureBlock):
    """Pass-through block for explicitly listed numeric columns.

    Leakage-safe: imputation values are learned from the training split
    during fit() and reused verbatim on every transform() split.

    Params
    ------
    columns : list[str]
        Required. Cleaned-data column names to emit as features (the
        canonical names produced by the descriptor's ``raw_column_map``).
        Output feature names equal the input column names.
    impute : str, default "median"
        Fill strategy for missing values, fitted on train: "median",
        "mean", or "zero".

    Examples
    --------
    >>> block = CoreNumericBlock({"columns": ["Thrust_Margin"]})
    >>> block.fit(train_df, ctx)
    >>> output = block.transform(test_df, ctx)
    >>> output.feature_names
    ['Thrust_Margin']
    """

    name: ClassVar[str] = "core_numeric"
    requires: ClassVar[list[str]] = []

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        columns = self.params.get("columns")
        if not columns or not isinstance(columns, list):
            raise ValueError(
                "core_numeric requires a non-empty 'columns' list param "
                "naming the cleaned-data columns to pass through."
            )
        if not all(isinstance(c, str) for c in columns):
            raise ValueError("core_numeric 'columns' entries must be strings.")
        if len(set(columns)) != len(columns):
            raise ValueError("core_numeric 'columns' entries must be unique.")
        impute = self.params.get("impute", "median")
        if impute not in _IMPUTE_STRATEGIES:
            raise ValueError(
                f"core_numeric 'impute' must be one of {_IMPUTE_STRATEGIES}, got {impute!r}."
            )
        self.columns: list[str] = list(columns)
        self.impute: str = impute
        self.required_columns = list(columns)
        self._fill_values: dict[str, float] = {}

    def _coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df[self.columns].apply(pd.to_numeric, errors="coerce")
        return numeric.astype(float)

    def fit(self, df: pd.DataFrame, ctx: FeatureContext) -> CoreNumericBlock:
        """Learn per-column imputation values from training data only."""
        self.validate_columns(df)
        numeric = self._coerce_numeric(df)
        for col in self.columns:
            series = numeric[col]
            if series.notna().sum() == 0:
                raise ValueError(
                    f"core_numeric column {col!r} has no numeric values in training data."
                )
            if self.impute == "median":
                fill = float(series.median())
            elif self.impute == "mean":
                fill = float(series.mean())
            else:
                fill = 0.0
            self._fill_values[col] = fill
        self._fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, ctx: FeatureContext) -> FeatureOutput:
        """Emit the configured columns, filling gaps with train-fitted values."""
        self._check_is_fitted()
        self.validate_columns(df)
        numeric = self._coerce_numeric(df)
        for col in self.columns:
            numeric[col] = numeric[col].fillna(self._fill_values[col])
        return FeatureOutput(
            data=numeric,
            feature_names=list(self.columns),
            metadata={
                "block": self.name,
                "params": self.params,
                "fill_values": dict(self._fill_values),
            },
        )
