"""Raw data ingestion with validation and metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.data.validation import validate_raw_dataframe
from panelcast.io.readers import read_csv
from panelcast.utils.hashing import sha256_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataDimensions:
    """Data dimensions for memory estimation.

    Attributes:
        n_observations: Number of rows after filtering.
        n_artists: Number of unique artists.
        source: Description of data source (e.g., "from data: filename.csv").
    """

    n_observations: int
    n_artists: int
    source: str

    @classmethod
    def from_defaults(cls) -> DataDimensions:
        """Create with conservative defaults when data unavailable."""
        return cls(
            n_observations=1000,
            n_artists=100,
            source="defaults (data unavailable)",
        )


def extract_data_dimensions(
    csv_path: Path | str = "data/raw/all_albums_full.csv",
    min_ratings: int = 10,
    descriptor: DatasetDescriptor | None = None,
) -> DataDimensions:
    """Extract observation and entity counts from raw CSV.

    Loads only the entity and observation-count columns for performance
    (~0.2s vs ~30s full load).

    Args:
        csv_path: Path to raw CSV file.
        min_ratings: Minimum observation-count filter (matches CLI default).
        descriptor: Dataset descriptor supplying the entity / observation-count
            column names. ``None`` uses the AOTY defaults ("Artist",
            "User Ratings").

    Returns:
        DataDimensions with counts and source indicator.
        Returns conservative defaults if file not found or on error.
    """
    path = Path(csv_path)

    if not path.exists():
        return DataDimensions.from_defaults()

    # Resolve the *raw* (pre-rename) header names for the entity and
    # observation-count columns. The descriptor stores canonical names plus a
    # raw->canonical map; invert it so a domain whose raw header differs from
    # the canonical name (e.g. "Sensor Samples" -> "Sensor_Samples") still
    # loads the right columns. None -> AOTY raw defaults.
    if descriptor is None:
        entity_raw = "Artist"
        n_obs_raw = "User Ratings"
    else:
        canonical_to_raw = {v: k for k, v in descriptor.raw_column_map.items()}
        entity_raw = canonical_to_raw.get(descriptor.entity_col, descriptor.entity_col)
        n_obs_raw = canonical_to_raw.get(descriptor.n_obs_col, descriptor.n_obs_col)

    try:
        # Only load the two columns needed for counting.
        df = read_csv(path, usecols=[entity_raw, n_obs_raw])

        # Apply the same observation-count filter as the training pipeline.
        df = df[df[n_obs_raw] >= min_ratings]

        return DataDimensions(
            n_observations=len(df),
            n_artists=df[entity_raw].nunique(),
            source=f"from data: {path.name}",
        )
    except Exception as e:
        logger.warning(
            "Failed to extract dimensions from %s (min_ratings=%d): %s. Falling back to defaults.",
            path.name,
            min_ratings,
            e,
        )
        return DataDimensions.from_defaults()


@dataclass
class LoadMetadata:
    """Metadata about the loaded dataset."""

    file_path: str
    file_hash: str
    load_timestamp: str
    row_count: int
    column_count: int


def load_raw_albums(
    path: str | Path = "data/raw/all_albums_full.csv",
    validate: bool = True,
    descriptor: DatasetDescriptor | None = None,
) -> tuple[pd.DataFrame, LoadMetadata]:
    """
    Load raw album data with validation and metadata.

    Args:
        path: Path to raw CSV file
        validate: Whether to validate against schema (default True).
            Raw data may have quality issues (e.g., 5 rows with null Album)
            that are caught by validation. Set to False to skip schema checks.
        descriptor: Dataset descriptor supplying the file encoding (and, for
            validation, the raw-column expectations). None = AOTY defaults.

    Returns:
        Tuple of (DataFrame, LoadMetadata)
        - DataFrame has 'original_row_id' column preserving raw CSV row numbers
        - LoadMetadata contains file hash and load info for reproducibility

    Raises:
        FileNotFoundError: If path does not exist
        pa.errors.SchemaErrors: If validation fails
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    # Compute hash before loading for reproducibility
    file_hash = sha256_file(path)

    # Load with the descriptor's encoding (AOTY default handles a BOM)
    encoding = descriptor.encoding if descriptor is not None else "utf-8-sig"
    df = read_csv(path, encoding=encoding)

    # Preserve original row IDs for audit trail (before any filtering)
    df["original_row_id"] = df.index

    # Validate if requested
    if validate:
        df = validate_raw_dataframe(df, descriptor=descriptor)

    # Build metadata
    metadata = LoadMetadata(
        file_path=str(path.resolve()),
        file_hash=file_hash,
        load_timestamp=datetime.now().isoformat(),
        row_count=len(df),
        column_count=len(df.columns),
    )

    return df, metadata


def load_raw_dataset(path: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """
    Legacy function for backward compatibility.

    Use load_raw_albums() for new code - it returns metadata.
    """
    # Preserve legacy signature by honoring caller-provided encoding.
    df = read_csv(path, encoding=encoding)
    df["original_row_id"] = df.index
    return df
