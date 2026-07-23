"""Train-fitted basis expansions for descriptor-declared covariate curves."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput


def basis_matrix_from_state(values: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    """Evaluate a fitted basis state on original-scale covariate values."""
    center = float(state["spec"].get("center", 0.0))
    return BSpline.design_matrix(
        values - center,
        state["knots"],
        int(state["degree"]),
        extrapolate=True,
    ).toarray()


class BasisBlock(BaseFeatureBlock):
    """Expand numeric covariates with train-fitted cubic B-spline bases."""

    name = "basis"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        curves = self.params.get("curves", {})
        if not curves:
            raise ValueError("basis block requires at least one curve specification")
        self.curves: dict[str, dict[str, Any]] = curves
        self.required_columns = [spec["col"] for spec in curves.values()]
        self.fitted_state: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _values(df: pd.DataFrame, col: str, curve_name: str) -> np.ndarray:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Basis curve {curve_name!r} source column {col!r} must contain "
                "only finite numeric values."
            )
        return values

    def fit(self, df: pd.DataFrame, ctx: FeatureContext) -> BasisBlock:
        super().fit(df, ctx)
        for name, spec in self.curves.items():
            values = self._values(df, spec["col"], name)
            center = float(spec.get("center") or 0.0)
            centered = values - center
            lower = float(centered.min())
            upper = float(centered.max())
            if lower == upper:
                raise ValueError(
                    f"Basis curve {name!r} cannot be fitted because training column "
                    f"{spec['col']!r} is constant."
                )
            degree = 3
            requested_df = int(spec["df"])
            fitted: tuple[list[float], np.ndarray] | None = None
            seen_interiors: set[tuple[float, ...]] = set()
            for interior_count in range(requested_df - degree - 1, -1, -1):
                if interior_count:
                    probs = np.arange(1, interior_count + 1) / (interior_count + 1)
                    quantiles = np.quantile(centered, probs).astype(float)
                    interior = sorted({float(v) for v in quantiles if lower < v < upper})
                else:
                    interior = []
                interior_key = tuple(interior)
                if interior_key in seen_interiors:
                    continue
                seen_interiors.add(interior_key)
                knots = [lower] * (degree + 1) + interior + [upper] * (degree + 1)
                matrix = BSpline.design_matrix(centered, knots, degree, extrapolate=True).toarray()
                if np.linalg.matrix_rank(matrix) == matrix.shape[1]:
                    fitted = (knots, matrix)
                    break
            if fitted is None:
                raise ValueError(
                    f"Basis curve {name!r} cannot produce a full-rank cubic spline "
                    f"from training column {spec['col']!r}; provide at least four "
                    "distinct, adequately supported values or request a different basis."
                )
            knots, matrix = fitted
            fitted_df = int(matrix.shape[1])
            feature_names = [f"{name}__basis_{i:02d}" for i in range(fitted_df)]
            self.fitted_state[name] = {
                "schema_version": 2,
                "spec": dict(spec),
                "degree": degree,
                "knots": knots,
                "requested_df": requested_df,
                "fitted_df": fitted_df,
                "train_min": float(values.min()),
                "train_max": float(values.max()),
                "feature_names": feature_names,
            }
        return self

    def transform(self, df: pd.DataFrame, ctx: FeatureContext) -> FeatureOutput:
        self._check_is_fitted()
        frames: list[pd.DataFrame] = []
        names: list[str] = []
        for name, spec in self.curves.items():
            state = self.fitted_state[name]
            values = self._values(df, spec["col"], name)
            matrix = basis_matrix_from_state(values, state)
            curve_names = state["feature_names"]
            if matrix.shape[1] != len(curve_names):
                raise ValueError(
                    f"Basis curve {name!r} state declares {len(curve_names)} columns "
                    f"but its knots produce {matrix.shape[1]}."
                )
            frames.append(pd.DataFrame(matrix, index=df.index, columns=curve_names))
            names.extend(curve_names)
        return FeatureOutput(
            data=pd.concat(frames, axis=1),
            feature_names=names,
            metadata={"name": self.name, "curves": self.fitted_state},
        )
