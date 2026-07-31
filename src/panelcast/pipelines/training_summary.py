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
import math
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
    invert_target_axis: bool = False
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
    # Per-group entity-effect variances (#271): "shared" | "per_group".
    group_variance: str | None = None
    # Period-effects gate (#269). period_to_idx maps str(period value) to
    # offset indices; evaluate looks held-out periods up the same way and
    # maps unseen ones to -1 (zero effect). All None on gate-off summaries.
    period_effects: bool | None = None
    period_col: str | None = None
    period_constraint: str | None = None
    period_to_idx: dict[str, int] | None = None
    n_periods: int | None = None
    # Expected-vs-actual fit resources (#78): estimator projection, measured
    # peak GPU memory, their ratio, and MCMC wall clock.
    resource_usage: dict[str, Any] | None = None
    # Train-fitted basis definitions bound to the exact model feature scaler.
    basis_curves: dict[str, Any] | None = None

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


DEFAULT_LOGIT_OFFSET = 0.5
# The pre-gate behavior a legacy summary with no recorded name means -- NOT
# the shipped config default, which resolves to offset_logit. Named apart from
# DEFAULT_LOGIT_OFFSET (which is the config default) so the two cannot be used
# interchangeably.
LEGACY_TARGET_TRANSFORM = "identity"
TARGET_TRANSFORMS = ("identity", "offset_logit")


def coerce_logit_offset(value: Any, *, context: str) -> float:
    """Validate one logit_offset and normalize it to a plain float.

    Shared by config validation and the summary resolver so the write path and
    the read path accept exactly the same domain: anything float-able except
    ``bool``, finite and non-negative. Zero is legal -- the plain logit, valid
    when observations sit strictly inside the bounds. Duck-typing through
    ``float`` also accepts numpy scalars, which an in-process summary can carry.

    Deliberately unbounded above: a huge offset flattens the target toward the
    middle of the bounds rather than leaving the transform's domain, so it is a
    modeling choice to be judged by the fit, not a malformed value.
    """
    if isinstance(value, bool):
        raise ValueError(f"Invalid logit_offset in {context}: {value!r}. Must be a number.")
    try:
        offset = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid logit_offset in {context}: {value!r}. Must be a number."
        ) from None
    if not math.isfinite(offset) or offset < 0.0:
        raise ValueError(f"Invalid logit_offset in {context}: {offset}. Must be finite and >= 0.")
    return offset


def coerce_target_transform(value: Any, *, context: str) -> str:
    """Validate a configured target_transform against the shipped names.

    The config side is deliberately narrower than the read side (see
    :func:`target_transform_from_summary`, which defers name checking to the
    live registry): a plugin transform is not registered until the model
    package imports, so a registry check at config-parse time would pass
    vacuously and fail later anyway. Widening the *configurable* set to the
    registry is the plugin work (#172), not a property of this contract.
    Both sides strip and return the bare name, so a padded one cannot be
    accepted by one end and rejected by the other.
    """
    name = value.strip() if isinstance(value, str) else value
    if name not in TARGET_TRANSFORMS:
        raise ValueError(
            f"Invalid target_transform in {context}: {value!r}. "
            f"Must be one of {', '.join(TARGET_TRANSFORMS)}."
        )
    return str(name)


def logit_offset_from_summary(summary: dict[str, Any]) -> float:
    """Offset-logit continuity offset the model was actually fit under.

    Only a missing key or an explicit ``null`` (legacy / pre-gate summaries)
    falls back to :data:`DEFAULT_LOGIT_OFFSET`. A recorded ``0.0`` is a real
    configuration and is propagated as zero, so evaluation, prediction and
    rollout apply the same forward transform, inverse and Jacobian the fit used.

    An out-of-range recorded offset is rejected here rather than downstream:
    every summary written before the config-side guard existed is unvalidated,
    and a negative or non-finite offset reaches the offset-logit map as a
    silent NaN log-likelihood instead of an error.
    """
    value = summary.get("logit_offset")
    if value is None:
        return DEFAULT_LOGIT_OFFSET
    return coerce_logit_offset(
        value, context="the training summary (re-run the train stage to rewrite it)"
    )


def target_transform_from_summary(summary: dict[str, Any]) -> str:
    """Target-transform name the model was actually fit under.

    Same null-versus-default confusion as the offset: the field is declared
    ``str | None`` and serializes as ``null`` on legacy summaries, so
    ``.get("target_transform", "identity")`` hands ``None`` to
    ``get_transform`` while ``.get(...) or "identity"`` resolves correctly.
    One resolver, so the two idioms cannot disagree.

    ``identity`` is the right fallback because the write path records a
    RESOLVED name: ``resolve_model_facts`` fills a null config value from the
    descriptor (else ``offset_logit``) before the stage context exists, so a
    null in a summary means the summary predates the transform gate, when
    identity was the only behavior. Only a null gets that fallback: an empty
    string is a recorded value, and rewriting it to a default is the same shape
    of bug as substituting a zero offset. Names themselves are checked by
    ``get_transform`` against the live registry, which stays extensible and
    already raises with the registered set.
    """
    value = summary.get("target_transform")
    if value is None:
        return LEGACY_TARGET_TRANSFORM
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Invalid target_transform in the training summary "
            f"(re-run the train stage to rewrite it): {value!r}. Must be a transform name."
        )
    return value.strip()


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
        target_transform_from_summary(summary),
        target_bounds=tuple(block.get("target_bounds", (0.0, 100.0))),
        offset=logit_offset_from_summary(summary),
    )
    return float(transform.forward(float(value)))
