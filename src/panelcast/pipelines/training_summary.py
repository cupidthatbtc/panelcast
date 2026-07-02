"""Typed training summary contract for models/training_summary.json.

The training summary is the hand-off artifact between the train stage and
every downstream consumer (evaluate, predict, reporting). This module gives it
an explicit, versioned schema:

- Fields are declared in the exact order the legacy dict literal produced, so
  serialized JSON keeps the historical key sequence as an ordered prefix and
  new keys (``schema_version``, ``dataset``) append at the end.
- ``extra="allow"`` keeps forward/backward compatibility with keys written by
  gated features (e.g. ``heteroscedastic_mode`` variants).
- :func:`load_training_summary` centralizes the legacy-upgrade path: summaries
  written before versioning (no ``schema_version``) are treated as v0 and
  upgraded in-memory with AOTY defaults, with a warning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from panelcast.config.descriptor import DEFAULT_DESCRIPTOR

log = structlog.get_logger()

SCHEMA_VERSION = 1


class DatasetSummaryBlock(BaseModel):
    """Provenance block describing the dataset the model was trained on."""

    model_config = ConfigDict(protected_namespaces=())

    name: str = "aoty"
    entity_col: str = "Artist"
    event_col: str = "Album"
    target_col: str = "User_Score"
    target_bounds: list[float] = Field(default_factory=lambda: [0.0, 100.0])
    model_prefix: str = "user"
    n_obs_col: str = "User_Ratings"
    secondary_target_col: str | None = "Critic_Score"
    secondary_prefix: str | None = "critic"
    descriptor_hash: str | None = None


class TrainingSummary(BaseModel):
    """Typed view of training_summary.json.

    Field declaration order intentionally mirrors the legacy dict literal in
    ``train_bayes.train_models`` so that ``model_dump()`` serializes the
    historical keys as an ordered prefix (regression-tested).
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    # --- legacy keys, in exact historical order --------------------------
    # All optional: consumers historically tolerated partial summaries
    # (and test fixtures construct minimal ones). Completeness of fresh
    # writes is guaranteed by train_models building the full dict.
    model_type: str | None = None
    model_path: str | None = None
    mcmc_config: dict[str, Any] | None = None
    convergence_thresholds: dict[str, Any] | None = None
    min_albums_filter: int | None = None
    n_artists_below_threshold: int | None = None
    priors: dict[str, Any] | None = None
    data_hash: str | None = None
    n_observations: int | None = None
    n_artists: int | None = None
    n_features: int | None = None
    feature_scaler: dict[str, Any] | None = None
    artist_to_idx: dict[str, int] | None = None
    max_seq: int | None = None
    max_albums: int | None = None
    global_mean_score: float | None = None
    feature_cols: list[str] | None = None
    n_exponent: float | None = None
    learn_n_exponent: bool | None = None
    n_exponent_prior: str | None = None
    likelihood_df: float | None = None
    n_ref: float | None = None
    n_reviews_stats: dict[str, Any] | None = None
    divergences: int | None = None
    divergence_rate: float | None = None
    runtime_seconds: float | None = None
    diagnostics: dict[str, Any] | None = None
    heteroscedastic_mode: dict[str, Any] | None = None

    # --- new keys append after every legacy key --------------------------
    schema_version: int = SCHEMA_VERSION
    dataset: DatasetSummaryBlock | None = None
    # Phase gates (None on legacy summaries -> consumers default to the
    # pre-gate behavior: identity transform, offset 0.5).
    target_transform: str | None = None
    logit_offset: float | None = None
    # Raw-scale AR(1) centering value (None on legacy summaries -> consumers
    # default to 0.0, the uncentered form the model was trained with).
    ar_center_value: float | None = None
    # Observation likelihood family (None on legacy summaries -> "studentt").
    likelihood_family: str | None = None
    # Whether the observation was interval-censored to integers (None/legacy -> False).
    discretize_observation: bool | None = None
    # Training-split score std on the model scale; the errors-in-variables path
    # derives prev_meas_sigma = global_std_score / sqrt(prev_n_reviews) at
    # predict/eval time (None on legacy summaries -> EIV unavailable downstream).
    global_std_score: float | None = None
    # Genre/group pooling gate (#41). group_to_idx maps group names to offset
    # indices (cold-start lookup); group_idx_by_artist is the per-entity index
    # vector the primary-split log-lik recompute conditions on. All None on
    # legacy / gate-off summaries.
    entity_group_pooling: bool | None = None
    entity_group_col: str | None = None
    group_to_idx: dict[str, int] | None = None
    group_idx_by_artist: list[int] | None = None
    n_groups: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize preserving declaration order.

        Declared keys that were never provided (unset, value None) are
        dropped so partial summaries round-trip without phantom null keys
        that would change ``summary.get(key, default)`` semantics downstream.
        """
        data = self.model_dump(mode="json")
        for name in type(self).model_fields:
            if name not in self.model_fields_set and data.get(name) is None:
                data.pop(name, None)
        return data


def load_training_summary(path: Path | str) -> TrainingSummary:
    """Load and validate a training summary, upgrading legacy files.

    Summaries written before versioning lack ``schema_version``; they are
    treated as v0 and upgraded with AOTY defaults (the only dataset that
    existed at the time), with a warning.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return upgrade_training_summary(raw, source=str(path))


def upgrade_training_summary(raw: dict[str, Any], source: str = "<dict>") -> TrainingSummary:
    """Validate a raw summary dict, applying the legacy v0 upgrade path."""
    if "schema_version" not in raw:
        log.warning(
            "training_summary_legacy_upgrade",
            source=source,
            from_version=0,
            to_version=SCHEMA_VERSION,
            message=(
                "training_summary.json predates schema versioning; assuming "
                "AOTY dataset defaults. Re-run the train stage to refresh."
            ),
        )
        raw = dict(raw)
        raw["schema_version"] = SCHEMA_VERSION
        if "dataset" not in raw:
            raw["dataset"] = DEFAULT_DESCRIPTOR.to_summary_block()
    return TrainingSummary(**raw)


def ar_center_on_model_scale(summary: dict[str, Any]) -> float:
    """Model-scale AR(1) centering value recorded in a training summary.

    Every prediction-time consumer must subtract the SAME center the model
    was trained with: ar_term = rho * (prev_score - center). Legacy
    summaries (no ``ar_center_value``) and ``ar_center="none"`` trainings
    resolve to 0.0, the uncentered form. The stored value is on the raw
    score scale; it is mapped through the training transform here. For
    "artist_running" trainings this is the global fallback value -- the
    per-observation running means are a training-time construct.
    """
    value = summary.get("ar_center_value")
    mode = (summary.get("priors") or {}).get("ar_center", "none")
    if value is None or mode == "none":
        return 0.0
    # Local import: keeps JAX out of the module-load path for light callers.
    from panelcast.models.bayes.transforms import get_transform

    block = summary.get("dataset") or {}
    transform = get_transform(
        summary.get("target_transform") or "identity",
        target_bounds=tuple(block.get("target_bounds", (0.0, 100.0))),
        offset=float(summary.get("logit_offset") or 0.5),
    )
    return float(transform.forward(float(value)))
