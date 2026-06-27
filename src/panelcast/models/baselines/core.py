"""Baseline predictors and a benchmark harness.

The Bayesian model needs something to be incrementally better *than*. These are
deliberately simple, fast, non-Bayesian predictors that nonetheless emit
predictive intervals, so they score through the exact same evaluation toolkit
(``evaluation.metrics`` / ``evaluation.calibration`` / CRPS / PPC) as the full
model. Every baseline produces Gaussian predictive samples around a point
prediction, with the spread set from train-residual scale — enough to fill the
coverage columns of the benchmark table honestly (train-residual intervals are
mildly optimistic; they are a floor, not a calibrated forecast).

Predictors:
- ``global_mean``  — the train global mean for every row.
- ``entity_mean``  — per-entity train mean, global-mean fallback for unseen
  entities (covers the entity-disjoint split).
- ``last_score``   — persistence: the entity's previous score.
- ``ridge``        — ``sklearn`` ridge regression on the feature matrix.
- ``gbm``          — ``sklearn`` histogram gradient boosting; residual-scale
  intervals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from panelcast.evaluation.calibration import compute_coverage
from panelcast.evaluation.metrics import compute_point_metrics
from panelcast.evaluation.ppc import compute_ppc_statistics

__all__ = [
    "PanelData",
    "BaselinePrediction",
    "BaselineScore",
    "Baseline",
    "GlobalMeanBaseline",
    "EntityMeanBaseline",
    "LastScoreBaseline",
    "RidgeBaseline",
    "GBMBaseline",
    "build_default_baselines",
    "score_prediction",
    "benchmark_baselines",
]

# Floor on predictive scale so degenerate (constant) targets still yield a
# proper, non-collapsed sampling distribution.
_SIGMA_FLOOR = 1e-3


@dataclass
class PanelData:
    """A split's panel in the shape every baseline consumes.

    Attributes:
        X: Standardized feature matrix, shape (n, p).
        y: Target on the score scale, shape (n,). May contain NaN for held-out
            rows; rows with NaN targets are dropped before scoring.
        entity: Entity id per row, shape (n,).
        prev_score: Previous-event score per row (debuts filled with the train
            mean), shape (n,), or None.
        bounds: (low, high) target bounds used to clip predictions/samples.
    """

    X: np.ndarray
    y: np.ndarray
    entity: np.ndarray
    prev_score: np.ndarray | None = None
    bounds: tuple[float, float] = (0.0, 100.0)


@dataclass
class BaselinePrediction:
    """Point prediction plus predictive samples for one split."""

    point: np.ndarray  # (n,)
    samples: np.ndarray  # (n_samples, n)


@dataclass
class BaselineScore:
    """Scored benchmark row for one baseline on one split."""

    model: str
    split: str
    n_obs: int
    mae: float
    rmse: float
    r2: float
    median_ae: float
    crps: float
    coverage: dict[float, float] = field(default_factory=dict)
    interval_width: dict[float, float] = field(default_factory=dict)
    ppc_skew_p: float = float("nan")
    runtime_s: float = 0.0

    def to_row(self, levels: tuple[float, ...]) -> dict[str, float | str]:
        """Flatten to a table row keyed by metric name."""
        row: dict[str, float | str] = {
            "model": self.model,
            "split": self.split,
            "n_obs": self.n_obs,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "crps": self.crps,
        }
        for level in levels:
            row[f"cov{int(round(level * 100))}"] = self.coverage.get(level, float("nan"))
        row["width95"] = self.interval_width.get(0.95, float("nan"))
        row["ppc_skew_p"] = self.ppc_skew_p
        row["runtime_s"] = self.runtime_s
        return row


def _train_residual_sigma(residuals: np.ndarray) -> float:
    """Robust-ish residual scale with a floor (std, never below the floor).

    Non-finite residuals (e.g. from coerced-NaN targets) are dropped first so a
    single NaN can't turn the whole predictive sigma — and every interval/CRPS
    metric downstream — into NaN.
    """
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return _SIGMA_FLOOR
    sigma = float(np.std(residuals))
    return max(sigma, _SIGMA_FLOOR)


def _gaussian_samples(
    point: np.ndarray,
    sigma: float,
    n_samples: int,
    bounds: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw clipped Gaussian predictive samples around a point prediction."""
    low, high = bounds
    noise = rng.standard_normal((n_samples, point.shape[0]))
    samples = point[None, :] + sigma * noise
    return np.clip(samples, low, high)


class Baseline:
    """Base predictor: fit on train, predict clipped Gaussian samples on test."""

    name: str = "baseline"

    def __init__(self, bounds: tuple[float, float] = (0.0, 100.0)) -> None:
        self.bounds = bounds
        self._sigma: float = _SIGMA_FLOOR

    def fit(self, train: PanelData) -> Baseline:  # pragma: no cover - overridden
        raise NotImplementedError

    def _point(self, test: PanelData) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def predict(
        self, test: PanelData, n_samples: int, rng: np.random.Generator
    ) -> BaselinePrediction:
        point = np.clip(self._point(test), *self.bounds)
        samples = _gaussian_samples(point, self._sigma, n_samples, self.bounds, rng)
        return BaselinePrediction(point=point, samples=samples)


class GlobalMeanBaseline(Baseline):
    name = "global_mean"

    def fit(self, train: PanelData) -> GlobalMeanBaseline:
        y = np.asarray(train.y, dtype=float)
        self._mean = float(np.nanmean(y)) if y.size else float(np.mean(self.bounds))
        self._sigma = _train_residual_sigma(y[~np.isnan(y)] - self._mean)
        return self

    def _point(self, test: PanelData) -> np.ndarray:
        return np.full(test.y.shape[0], self._mean, dtype=float)


class EntityMeanBaseline(Baseline):
    name = "entity_mean"

    def fit(self, train: PanelData) -> EntityMeanBaseline:
        y = np.asarray(train.y, dtype=float)
        ent = np.asarray(train.entity)
        valid = ~np.isnan(y)
        self._global = float(np.mean(y[valid])) if valid.any() else float(np.mean(self.bounds))
        sums: dict[object, float] = {}
        counts: dict[object, int] = {}
        for e, yi in zip(ent[valid], y[valid]):
            sums[e] = sums.get(e, 0.0) + float(yi)
            counts[e] = counts.get(e, 0) + 1
        self._means = {e: sums[e] / counts[e] for e in sums}
        fitted = np.array([self._means.get(e, self._global) for e in ent[valid]])
        self._sigma = _train_residual_sigma(y[valid] - fitted)
        return self

    def _point(self, test: PanelData) -> np.ndarray:
        return np.array(
            [self._means.get(e, self._global) for e in np.asarray(test.entity)], dtype=float
        )


class LastScoreBaseline(Baseline):
    """Persistence: predict the entity's previous score.

    Uses ``prev_score`` when the panel provides it (debuts already filled with
    the train mean); otherwise falls back to the entity's train mean, then the
    global mean. The residual scale is the train one-step persistence error.
    """

    name = "last_score"

    def fit(self, train: PanelData) -> LastScoreBaseline:
        y = np.asarray(train.y, dtype=float)
        valid = ~np.isnan(y)
        self._global = float(np.mean(y[valid])) if valid.any() else float(np.mean(self.bounds))
        ent = np.asarray(train.entity)
        sums: dict[object, float] = {}
        counts: dict[object, int] = {}
        for e, yi in zip(ent[valid], y[valid]):
            sums[e] = sums.get(e, 0.0) + float(yi)
            counts[e] = counts.get(e, 0) + 1
        self._entity_mean = {e: sums[e] / counts[e] for e in sums}
        if train.prev_score is not None:
            prev = np.asarray(train.prev_score, dtype=float)[valid]
            self._sigma = _train_residual_sigma(y[valid] - prev)
        else:
            self._sigma = _train_residual_sigma(y[valid] - self._global)
        return self

    def _point(self, test: PanelData) -> np.ndarray:
        ent = np.asarray(test.entity)
        if test.prev_score is not None:
            prev = np.asarray(test.prev_score, dtype=float)
            fallback = np.array([self._entity_mean.get(e, self._global) for e in ent])
            return np.where(np.isnan(prev), fallback, prev)
        return np.array([self._entity_mean.get(e, self._global) for e in ent], dtype=float)


class RidgeBaseline(Baseline):
    name = "ridge"

    def __init__(self, bounds: tuple[float, float] = (0.0, 100.0), alpha: float = 1.0) -> None:
        super().__init__(bounds)
        self.alpha = alpha

    def fit(self, train: PanelData) -> RidgeBaseline:
        from sklearn.linear_model import Ridge

        y = np.asarray(train.y, dtype=float)
        valid = ~np.isnan(y)
        X = np.asarray(train.X, dtype=float)[valid]
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X, y[valid])
        self._sigma = _train_residual_sigma(y[valid] - self._model.predict(X))
        return self

    def _point(self, test: PanelData) -> np.ndarray:
        return self._model.predict(np.asarray(test.X, dtype=float))


class GBMBaseline(Baseline):
    name = "gbm"

    def fit(self, train: PanelData) -> GBMBaseline:
        from sklearn.ensemble import HistGradientBoostingRegressor

        y = np.asarray(train.y, dtype=float)
        valid = ~np.isnan(y)
        X = np.asarray(train.X, dtype=float)[valid]
        self._model = HistGradientBoostingRegressor(random_state=0)
        self._model.fit(X, y[valid])
        self._sigma = _train_residual_sigma(y[valid] - self._model.predict(X))
        return self

    def _point(self, test: PanelData) -> np.ndarray:
        return self._model.predict(np.asarray(test.X, dtype=float))


def build_default_baselines(bounds: tuple[float, float] = (0.0, 100.0)) -> list[Baseline]:
    """The full baseline panel, in table order."""
    return [
        GlobalMeanBaseline(bounds),
        EntityMeanBaseline(bounds),
        LastScoreBaseline(bounds),
        RidgeBaseline(bounds),
        GBMBaseline(bounds),
    ]


def _crps_from_samples(y_true: np.ndarray, samples: np.ndarray) -> float:
    from panelcast.evaluation.metrics import compute_crps

    return float(compute_crps(y_true, samples).mean_crps)


def score_prediction(
    model: str,
    split: str,
    y_true: np.ndarray,
    prediction: BaselinePrediction,
    levels: tuple[float, ...] = (0.80, 0.95),
    runtime_s: float = 0.0,
) -> BaselineScore:
    """Score a baseline prediction through the shared evaluation toolkit."""
    y_true = np.asarray(y_true, dtype=float)
    point = np.asarray(prediction.point, dtype=float)
    samples = np.asarray(prediction.samples, dtype=float)

    pm = compute_point_metrics(y_true, point)
    crps = _crps_from_samples(y_true, samples)

    coverage: dict[float, float] = {}
    width: dict[float, float] = {}
    for level in levels:
        cov = compute_coverage(y_true, samples, prob=level)
        coverage[level] = cov.empirical
        width[level] = cov.interval_width

    ppc = compute_ppc_statistics(y_true, samples)
    skew = next((s for s in ppc.statistics if s.name == "skewness"), None)

    return BaselineScore(
        model=model,
        split=split,
        n_obs=int(y_true.shape[0]),
        mae=pm.mae,
        rmse=pm.rmse,
        r2=pm.r2,
        median_ae=pm.median_ae,
        crps=crps,
        coverage=coverage,
        interval_width=width,
        ppc_skew_p=float(skew.bayesian_p_value) if skew is not None else float("nan"),
        runtime_s=runtime_s,
    )


def _drop_nan_targets(test: PanelData) -> tuple[PanelData, np.ndarray]:
    """Restrict a test panel to rows with known targets; return (panel, y)."""
    y = np.asarray(test.y, dtype=float)
    mask = ~np.isnan(y)
    if mask.all():
        return test, y
    return (
        PanelData(
            X=np.asarray(test.X)[mask],
            y=y[mask],
            entity=np.asarray(test.entity)[mask],
            prev_score=None if test.prev_score is None else np.asarray(test.prev_score)[mask],
            bounds=test.bounds,
        ),
        y[mask],
    )


def benchmark_baselines(
    train: PanelData,
    test: PanelData,
    split: str,
    baselines: list[Baseline] | None = None,
    levels: tuple[float, ...] = (0.80, 0.95),
    n_samples: int = 1000,
    seed: int = 0,
) -> list[BaselineScore]:
    """Fit and score every baseline on one split.

    Returns one :class:`BaselineScore` per baseline. Rows whose test target is
    NaN (masked held-out labels) are dropped before scoring.
    """
    baselines = baselines or build_default_baselines(train.bounds)
    test_eval, y_true = _drop_nan_targets(test)
    rng = np.random.default_rng(seed)
    scores: list[BaselineScore] = []
    for bl in baselines:
        start = time.perf_counter()
        bl.fit(train)
        prediction = bl.predict(test_eval, n_samples=n_samples, rng=rng)
        runtime = time.perf_counter() - start
        scores.append(
            score_prediction(bl.name, split, y_true, prediction, levels=levels, runtime_s=runtime)
        )
    return scores
