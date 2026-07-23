"""Posterior extraction and peak summaries for fitted basis curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from panelcast.features.basis import basis_matrix_from_state


@dataclass(frozen=True)
class PosteriorCurve:
    """A common covariate grid and one fitted curve per posterior draw."""

    x: np.ndarray
    draws: np.ndarray


@dataclass(frozen=True)
class CurvePeakSummary:
    """Posterior summary of a curve's maximizing or minimizing vertex."""

    direction: Literal["max", "min"]
    vertex_draws: np.ndarray
    value_draws: np.ndarray
    vertex_median: float
    vertex_interval: tuple[float, float]
    value_median: float
    value_interval: tuple[float, float]
    boundary_fraction: float


def basis_matrix(x: Sequence[float] | np.ndarray, state: dict[str, Any]) -> np.ndarray:
    """Rebuild a basis matrix exactly from feature-manifest fitted state."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Curve grid must be a one-dimensional array of finite values.")
    spec = state.get("spec", {})
    if spec.get("type") != "spline":
        raise ValueError(f"Unsupported basis curve type: {spec.get('type')!r}.")
    return basis_matrix_from_state(values, state)


def extract_curve_draws(
    coefficient_draws: np.ndarray,
    state: dict[str, Any],
    *,
    grid: Sequence[float] | np.ndarray | None = None,
    grid_size: int = 201,
) -> PosteriorCurve:
    """Evaluate posterior basis coefficients over an original-scale grid."""
    coefficients = np.asarray(coefficient_draws, dtype=float)
    expected = len(state["feature_names"])
    if coefficients.ndim < 2 or coefficients.shape[-1] != expected:
        raise ValueError(
            "Coefficient draws must have shape (..., n_basis) with "
            f"n_basis={expected}; got {coefficients.shape}."
        )
    if not np.isfinite(coefficients).all():
        raise ValueError("Coefficient draws must contain only finite values.")
    if grid is None:
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        x = np.linspace(float(state["train_min"]), float(state["train_max"]), grid_size)
    else:
        x = np.asarray(grid, dtype=float)
    matrix = basis_matrix(x, state)
    flat = coefficients.reshape(-1, expected)
    return PosteriorCurve(x=x, draws=flat @ matrix.T)


def extract_posterior_curve(
    coefficient_draws: np.ndarray,
    feature_names: Sequence[str],
    state: dict[str, Any],
    *,
    grid: Sequence[float] | np.ndarray | None = None,
    grid_size: int = 201,
) -> PosteriorCurve:
    """Select a curve's model coefficients by manifest names and evaluate it."""
    names = list(feature_names)
    required = list(state["feature_names"])
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Posterior feature names are missing basis columns: {missing}.")
    draws = np.asarray(coefficient_draws)
    if draws.ndim < 2 or draws.shape[-1] != len(names):
        raise ValueError(
            "Coefficient draws' last dimension must match feature_names; "
            f"got {draws.shape} and {len(names)} names."
        )
    indices = [names.index(name) for name in required]
    return extract_curve_draws(draws[..., indices], state, grid=grid, grid_size=grid_size)


def summarize_curve_peak(
    curve: PosteriorCurve,
    *,
    direction: Literal["max", "min"] = "max",
    credible_mass: float = 0.9,
) -> CurvePeakSummary:
    """Summarize posterior grid vertices for a maximum or minimum claim."""
    if direction not in ("max", "min"):
        raise ValueError("direction must be 'max' or 'min'.")
    if not 0.0 < credible_mass < 1.0:
        raise ValueError("credible_mass must be between 0 and 1.")
    if curve.draws.ndim != 2 or curve.draws.shape[1] != len(curve.x):
        raise ValueError("Curve draws must have shape (draw, grid_point).")
    indices = (
        np.argmax(curve.draws, axis=1)
        if direction == "max"
        else np.argmin(curve.draws, axis=1)
    )
    vertex_draws = curve.x[indices]
    value_draws = curve.draws[np.arange(len(indices)), indices]
    alpha = (1.0 - credible_mass) / 2.0
    vertex_interval = tuple(float(v) for v in np.quantile(vertex_draws, [alpha, 1 - alpha]))
    value_interval = tuple(float(v) for v in np.quantile(value_draws, [alpha, 1 - alpha]))
    boundary_fraction = float(np.mean((indices == 0) | (indices == len(curve.x) - 1)))
    return CurvePeakSummary(
        direction=direction,
        vertex_draws=vertex_draws,
        value_draws=value_draws,
        vertex_median=float(np.median(vertex_draws)),
        vertex_interval=vertex_interval,
        value_median=float(np.median(value_draws)),
        value_interval=value_interval,
        boundary_fraction=boundary_fraction,
    )
