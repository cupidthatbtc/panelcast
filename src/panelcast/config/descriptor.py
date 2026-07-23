"""Dataset descriptor: the single source of truth for domain-specific names.

A :class:`DatasetDescriptor` captures everything that ties the pipeline to a
particular dataset/domain — column names, target bounds, date formats,
posterior-site prefixes, feature-block composition. Every field default is the
exact literal the codebase used for the AOTY dataset, so ``DatasetDescriptor()``
reproduces today's behavior byte-for-byte ("default-equals-AOTY").

Retargeting the pipeline to a new domain means writing one YAML file (see
``configs/datasets/``) — no source changes. Bare names passed to
:func:`load_descriptor` resolve to ``configs/datasets/{name}.yaml``.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, PrivateAttr, model_validator

# Default raw->canonical column mapping (mirrors data.cleaning.RAW_TO_CANONICAL;
# kept as a literal here so the descriptor module has no panelcast imports and
# stays importable from anywhere without cycles).
_AOTY_RAW_COLUMN_MAP = {
    "Release Date": "Release_Date",
    "Critic Score": "Critic_Score",
    "User Score": "User_Score",
    "Avg Track Score": "Avg_Track_Score",
    "User Ratings": "User_Ratings",
    "Critic Reviews": "Critic_Reviews",
    "Tracks": "Num_Tracks",
    "Runtime (min)": "Runtime_Min",
    "Avg Track Runtime (min)": "Avg_Runtime",
    "Album URL": "Album_URL",
    "All Artists": "All_Artists",
    "Album Type": "Album_Type",
}

_AOTY_REQUIRED_RAW_COLUMNS = [
    "Artist",
    "Album",
    "Year",
    "Release Date",
    "Genres",
    "User Score",
    "User Ratings",
    "Tracks",
    "Runtime (min)",
    "Avg Track Runtime (min)",
    "Album Type",
    "All Artists",
]

_AOTY_OPTIONAL_RAW_COLUMNS = [
    "Critic Score",
    "Critic Reviews",
    "Avg Track Score",
    "Descriptors",
    "Label",
    "Album URL",
]


class FeatureBlockSpec(BaseModel):
    """One feature block to instantiate, by registry name, with params."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


def _default_feature_blocks() -> list[FeatureBlockSpec]:
    """The current AOTY block list, in dependency order."""
    return [
        FeatureBlockSpec(name="temporal"),
        FeatureBlockSpec(name="album_type"),
        FeatureBlockSpec(name="artist_history"),
        FeatureBlockSpec(name="genre", params={"min_genre_count": 20, "n_components": 10}),
        FeatureBlockSpec(name="collaboration"),
    ]


def _default_ablation_groups() -> dict[str, list[str]]:
    """Map CLI ablation flags to the block names they disable."""
    return {
        "genre": ["genre"],
        "artist": ["artist_history"],
        "temporal": ["temporal"],
    }


def _default_group_size_bins() -> dict[str, list[int | None]]:
    """Collaboration-size bins: label -> [min_count, max_count] (None = open)."""
    return {
        "solo": [1, 1],
        "duo": [2, 2],
        "small_group": [3, 4],
        "ensemble": [5, None],
    }


class DatasetDescriptor(BaseModel):
    """Declarative description of a dataset for the prediction pipeline.

    Field defaults are the AOTY literals they replaced; constructing with no
    arguments reproduces the original hard-coded behavior exactly.
    """

    _source_path: Path | None = PrivateAttr(default=None)
    _source_root: Path | None = PrivateAttr(default=None)

    # --- identity -------------------------------------------------------
    name: str = "aoty"

    # --- raw source -----------------------------------------------------
    raw_path_env: str = "AOTY_DATASET_PATH"
    raw_path_default: str = "data/raw/all_albums_full.csv"
    encoding: str = "utf-8-sig"
    raw_column_map: dict[str, str] = Field(default_factory=lambda: dict(_AOTY_RAW_COLUMN_MAP))
    required_raw_columns: list[str] = Field(
        default_factory=lambda: list(_AOTY_REQUIRED_RAW_COLUMNS)
    )
    optional_raw_columns: list[str] = Field(
        default_factory=lambda: list(_AOTY_OPTIONAL_RAW_COLUMNS)
    )

    # --- identity / sequencing ------------------------------------------
    entity_col: str = "Artist"
    event_col: str = "Album"
    # Optional per-event group column for the entity_group_pooling gate (an
    # entity's modal value over its training rows becomes its group). None
    # makes the gate unusable for the domain.
    entity_group_col: str | None = "primary_genre"
    date_col: str = "Release_Date"
    parsed_date_col: str = "Release_Date_Parsed"
    year_col: str = "Year"
    date_format: str = "%B %d, %Y"

    # --- targets ----------------------------------------------------------
    target_col: str = "User_Score"
    target_bounds: tuple[float, float] = (0.0, 100.0)
    invert_target_axis: bool = False
    model_prefix: str = "user"
    n_obs_col: str = "User_Ratings"
    # Whether n_obs_col counts independent raters whose mean IS the target (AOTY:
    # User_Ratings raters average to User_Score). Gates the beta_binomial family,
    # which only makes sense for a true aggregation count — not, e.g., a sensor
    # sample count that doesn't average to the performance score.
    n_obs_is_aggregation_count: bool = True
    # Secondary (dual-model) path; all None disables it for non-AOTY domains.
    secondary_target_col: str | None = "Critic_Score"
    secondary_prefix: str | None = "critic"
    secondary_n_obs_col: str | None = "Critic_Reviews"

    # --- cleaning semantics ----------------------------------------------
    multi_entity_col: str | None = "All_Artists"
    multi_entity_separator: str = " | "
    unknown_entity_sentinel: str | None = "[unknown artist]"
    group_size_bins: dict[str, list[int | None]] = Field(default_factory=_default_group_size_bins)
    min_year: int = 1950

    # --- dataset preparation ----------------------------------------------
    min_obs_thresholds: list[int] = Field(default_factory=lambda: [5, 10, 25])
    primary_min_obs: int = 10
    processed_name_template: str = "user_score_minratings_{min_ratings}"

    # --- features ----------------------------------------------------------
    feature_packs: list[str] = Field(default_factory=lambda: ["aoty"])
    feature_blocks: list[FeatureBlockSpec] = Field(default_factory=_default_feature_blocks)
    ablation_groups: dict[str, list[str]] = Field(default_factory=_default_ablation_groups)

    @model_validator(mode="after")
    def _validate(self) -> DatasetDescriptor:
        if self.primary_min_obs not in self.min_obs_thresholds:
            raise ValueError(
                f"primary_min_obs={self.primary_min_obs} is not one of "
                f"min_obs_thresholds={self.min_obs_thresholds}."
            )
        lo, hi = self.target_bounds
        if not lo < hi:
            raise ValueError(f"target_bounds must satisfy low < high, got {self.target_bounds}.")
        secondary_fields = (
            self.secondary_target_col,
            self.secondary_prefix,
            self.secondary_n_obs_col,
        )
        if any(f is not None for f in secondary_fields) and not all(
            f is not None for f in secondary_fields
        ):
            raise ValueError(
                "secondary_target_col, secondary_prefix and secondary_n_obs_col "
                "must be set together (or all None to disable the secondary model)."
            )
        if "{min_ratings}" not in self.processed_name_template:
            raise ValueError(
                "processed_name_template must contain the '{min_ratings}' placeholder."
            )
        return self

    # --- derived helpers ----------------------------------------------------

    def processed_name(self, min_obs: int | None = None) -> str:
        """Processed dataset name for a threshold (default: primary)."""
        value = self.primary_min_obs if min_obs is None else min_obs
        return self.processed_name_template.format(min_ratings=value)

    def resolve_raw_path(self) -> Path:
        """Resolve the environment, cwd, descriptor-local, then packaged raw path."""
        env_value = os.environ.get(self.raw_path_env)
        path = Path(env_value if env_value is not None else self.raw_path_default)
        if env_value is not None or path.is_absolute() or path.exists():
            return path
        roots = [self._source_path.parent if self._source_path else None, self._source_root]
        for root in roots:
            if root is not None and (candidate := root / path).exists():
                return candidate
        return path

    def descriptor_hash(self) -> str:
        """Stable fit/data hash; presentation-only fields do not invalidate runs."""
        payload = json.dumps(
            self.model_dump(mode="json", exclude={"invert_target_axis"}),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_summary_block(self) -> dict[str, Any]:
        """Compact provenance block embedded in training summaries."""
        return {
            "name": self.name,
            "entity_col": self.entity_col,
            "event_col": self.event_col,
            "target_col": self.target_col,
            "target_bounds": list(self.target_bounds),
            "invert_target_axis": self.invert_target_axis,
            "model_prefix": self.model_prefix,
            "n_obs_col": self.n_obs_col,
            "secondary_target_col": self.secondary_target_col,
            "secondary_prefix": self.secondary_prefix,
            "descriptor_hash": self.descriptor_hash(),
        }


DEFAULT_DESCRIPTOR = DatasetDescriptor()


def _packaged_data_root() -> Path:
    return Path(str(files("panelcast").joinpath("_data")))


def resolve_descriptor_path(ref: str | Path | None) -> Path | None:
    """Resolve a descriptor reference, preferring checkout files over package data."""
    if ref is None:
        return None
    path = Path(ref)
    if path.suffix in (".yaml", ".yml"):
        return path
    checkout_path = Path("configs/datasets") / f"{path.name}.yaml"
    if checkout_path.exists():
        return checkout_path
    packaged_path = _packaged_data_root() / "datasets" / f"{path.name}.yaml"
    return packaged_path if packaged_path.exists() else checkout_path


def resolve_demo_descriptor_path(ref: str | Path) -> Path:
    """Resolve the demo's bundled descriptor consistently in checkouts and wheels."""
    if str(ref) != "aero":
        resolved = resolve_descriptor_path(ref)
        assert resolved is not None
        return resolved

    checkout_path = Path("examples/aerospace/descriptor.yaml")
    if checkout_path.exists():
        return checkout_path
    return _packaged_data_root() / "examples" / "aerospace" / "descriptor.yaml"


def load_descriptor(ref: str | Path | None) -> DatasetDescriptor:
    """Resolve a descriptor reference to a :class:`DatasetDescriptor`.

    Args:
        ref: ``None`` (AOTY defaults), a bare name resolved to
            ``configs/datasets/{ref}.yaml``, or an explicit YAML path.

    Returns:
        The loaded descriptor. Omitted YAML keys keep their AOTY defaults.
    """
    if ref is None:
        return DatasetDescriptor()

    path = resolve_descriptor_path(ref)
    assert path is not None
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset descriptor not found: {path}. Bare names resolve to "
            "configs/datasets/{name}.yaml."
        )

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Descriptor file {path} must be a YAML mapping.")

    # Reuse the loader's env-var expansion so descriptor YAML supports
    # ${VAR} interpolation like the rest of the config layer.
    from panelcast.config.loader import _expand_env_vars

    data = _expand_env_vars(data)
    descriptor = DatasetDescriptor(**data)
    descriptor._source_path = path.resolve()
    package_root = _packaged_data_root().resolve()
    if descriptor._source_path.is_relative_to(package_root):
        descriptor._source_root = package_root
    else:
        for parent in descriptor._source_path.parents:
            if (parent / "pyproject.toml").is_file():
                descriptor._source_root = parent
                break
    return descriptor
