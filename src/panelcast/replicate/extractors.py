"""Named posterior extractors for replication claims (#272).

Each extractor turns a fitted run's artifacts into a 1-D array of posterior
draws for one scalar quantity, so claims grade against draws rather than
point estimates. Promoted from the per-domain scripts written twice in
panelcast-replications (baseball + chess).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from panelcast.replicate.spec import ClaimSpec

_REST_BUCKET = "__rest__"


@dataclass(frozen=True)
class ArtifactBundle:
    """A fitted run's posterior and its training-summary bookkeeping."""

    posterior: dict[str, np.ndarray]  # site -> (n_flat_draws, ...) arrays
    summary: dict

    @property
    def prefix(self) -> str:
        return str(self.summary["dataset"]["model_prefix"])

    def site(self, name: str) -> np.ndarray:
        key = f"{self.prefix}_{name}"
        if key not in self.posterior:
            raise ValueError(
                f"posterior has no site '{key}'. Available: "
                f"{sorted(self.posterior)[:12]}..."
            )
        return self.posterior[key]

    def feature_index(self, feature: str) -> int:
        cols = list(self.summary["feature_cols"])
        if feature not in cols:
            raise ValueError(
                f"unknown feature '{feature}'. Trained features: {cols}."
            )
        return cols.index(feature)

    def feature_moments(self, feature: str) -> tuple[float, float]:
        """(mean, std) the feature was standardized with; (0, 1) if unscaled."""
        scaler = self.summary.get("feature_scaler") or {}
        cols = list(scaler.get("feature_cols", []))
        if feature in cols:
            i = cols.index(feature)
            std = float(scaler["std"][i])
            return float(scaler["mean"][i]), std if std > 0 else 1.0
        return 0.0, 1.0

    def entity_indices(self, names: list[str]) -> np.ndarray:
        mapping = self.summary["artist_to_idx"]
        missing = [n for n in names if n not in mapping]
        if missing:
            raise ValueError(
                f"entities not in the trained panel: {missing}. "
                "Claim entity names must match the entity column exactly."
            )
        return np.asarray([int(mapping[n]) for n in names], dtype=np.int64)


def load_bundle(models_dir: Path | str) -> ArtifactBundle:
    """Load the newest .nc posterior + training summary from a models dir."""
    import arviz as az

    models_dir = Path(models_dir)
    summary_path = models_dir / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"no training_summary.json in {models_dir}.")
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    candidates = sorted(models_dir.glob("*.nc"))
    if not candidates:
        raise FileNotFoundError(f"no .nc posterior in {models_dir}.")
    idata = az.from_netcdf(str(candidates[-1]))
    posterior = {
        name: np.asarray(idata.posterior[name].values).reshape(
            -1, *idata.posterior[name].values.shape[2:]
        )
        for name in idata.posterior.data_vars
    }
    return ArtifactBundle(posterior=posterior, summary=summary)


@dataclass(frozen=True)
class ExtractedQuantity:
    """Posterior draws of one scalar claim quantity."""

    draws: np.ndarray  # (n_flat_draws,)
    shape_ok: bool  # the quantity is structurally well-defined
    detail: str  # human line for the verdict table


def _ordered_group_columns(bundle: ArtifactBundle, from_label: str | None) -> list[int]:
    group_to_idx: dict[str, int] = bundle.summary["group_to_idx"]
    labels = sorted(label for label in group_to_idx if label != _REST_BUCKET)
    if from_label is not None:
        if from_label not in labels:
            raise ValueError(
                f"expect.from='{from_label}' is not a trained group; "
                f"groups: {labels}."
            )
        labels = labels[labels.index(from_label) :]
    return [int(group_to_idx[label]) for label in labels]


def group_mean_trend(bundle: ArtifactBundle, claim: ClaimSpec) -> ExtractedQuantity:
    """Per-draw least-squares slope of the group offsets over ordered groups."""
    offsets = bundle.site("group_offset")
    columns = _ordered_group_columns(bundle, claim.expect.from_)
    if len(columns) < 2:
        return ExtractedQuantity(np.zeros(offsets.shape[0]), False, "fewer than 2 groups")
    series = offsets[:, columns]  # (draws, k)
    positions = np.arange(series.shape[1], dtype=float)
    centered = positions - positions.mean()
    slopes = (series * centered).sum(axis=1) / (centered**2).sum()
    return ExtractedQuantity(slopes, True, f"slope over {series.shape[1]} ordered groups")


def covariate_vertex(bundle: ArtifactBundle, claim: ClaimSpec) -> ExtractedQuantity:
    """Vertex of a quadratic covariate pair, on the raw feature scale.

    With standardized columns z_l=(x-m_l)/s_l and z_q=(x^2-m_q)/s_q the mean
    is extremal at x* = -(beta_l/s_l) * s_q / (2 beta_q): the standardization
    means shift the intercept only.
    """
    args = claim.extractor_args
    if len(args) != 2:
        raise ValueError(f"claim '{claim.name}': covariate_vertex(linear, quadratic).")
    lin, quad = args
    beta = bundle.site("beta")
    beta_l = beta[:, bundle.feature_index(lin)]
    beta_q = beta[:, bundle.feature_index(quad)]
    _, s_l = bundle.feature_moments(lin)
    _, s_q = bundle.feature_moments(quad)
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = -(beta_l / s_l) * s_q / (2.0 * beta_q)
    # A claimed peak needs downward curvature in most of the posterior.
    curvature_down = float(np.mean(beta_q < 0))
    shape_ok = bool(np.isfinite(vertex).all()) and curvature_down >= 0.5
    return ExtractedQuantity(
        vertex, shape_ok, f"P(curvature<0)={curvature_down:.2f}"
    )


def entity_contrast(bundle: ArtifactBundle, claim: ClaimSpec) -> ExtractedQuantity:
    """Difference in mean initial entity effects: group_a minus group_b."""
    if claim.entities is None:
        raise ValueError(f"claim '{claim.name}': entity_contrast needs an entities block.")
    effects = bundle.site("init_artist_effect")
    idx_a = bundle.entity_indices(claim.entities.group_a)
    if claim.entities.group_b == "rest":
        mask = np.ones(effects.shape[1], dtype=bool)
        mask[idx_a] = False
        idx_b = np.flatnonzero(mask)
    else:
        idx_b = bundle.entity_indices(claim.entities.group_b)
    draws = effects[:, idx_a].mean(axis=1) - effects[:, idx_b].mean(axis=1)
    return ExtractedQuantity(
        draws, True, f"{len(idx_a)} vs {len(idx_b)} entities"
    )


def entity_ranking(bundle: ArtifactBundle, claim: ClaimSpec) -> ExtractedQuantity:
    """Per-draw fraction of group_a entities inside that draw's top-K."""
    args = claim.extractor_args
    if len(args) != 1:
        raise ValueError(f"claim '{claim.name}': entity_ranking(top_k).")
    top_k = int(args[0])
    if claim.entities is None:
        raise ValueError(f"claim '{claim.name}': entity_ranking needs an entities block.")
    effects = bundle.site("init_artist_effect")
    idx_a = bundle.entity_indices(claim.entities.group_a)
    if top_k < 1 or top_k > effects.shape[1]:
        raise ValueError(
            f"claim '{claim.name}': top_k={top_k} outside [1, {effects.shape[1]}]."
        )
    order = np.argsort(-effects, axis=1)[:, :top_k]
    hits = np.isin(order, idx_a).sum(axis=1)
    return ExtractedQuantity(
        hits / float(len(idx_a)), True, f"top-{top_k} membership of {len(idx_a)} entities"
    )


def decline_between_ages(bundle: ArtifactBundle, claim: ClaimSpec) -> ExtractedQuantity:
    """Covariate-curve difference mu(b) - mu(a) for a quadratic pair.

    decline_between_ages(linear, quadratic, a, b): negative draws mean the
    curve declines from a to b.
    """
    args = claim.extractor_args
    if len(args) != 4:
        raise ValueError(
            f"claim '{claim.name}': decline_between_ages(linear, quadratic, a, b)."
        )
    lin, quad, raw_a, raw_b = args
    a, b = float(raw_a), float(raw_b)
    beta = bundle.site("beta")
    beta_l = beta[:, bundle.feature_index(lin)]
    beta_q = beta[:, bundle.feature_index(quad)]
    m_l, s_l = bundle.feature_moments(lin)
    m_q, s_q = bundle.feature_moments(quad)
    contribution_a = beta_l * (a - m_l) / s_l + beta_q * (a**2 - m_q) / s_q
    contribution_b = beta_l * (b - m_l) / s_l + beta_q * (b**2 - m_q) / s_q
    return ExtractedQuantity(
        contribution_b - contribution_a, True, f"curve change {a:g} -> {b:g}"
    )


EXTRACTORS = {
    "group_mean_trend": group_mean_trend,
    "covariate_vertex": covariate_vertex,
    "entity_contrast": entity_contrast,
    "entity_ranking": entity_ranking,
    "decline_between_ages": decline_between_ages,
}


def extract(bundle: ArtifactBundle, claim: ClaimSpec) -> ExtractedQuantity:
    extractor = EXTRACTORS.get(claim.extractor_name)
    if extractor is None:
        raise ValueError(
            f"claim '{claim.name}': unknown quantity '{claim.extractor_name}'. "
            f"Known: {sorted(EXTRACTORS)}."
        )
    return extractor(bundle, claim)
