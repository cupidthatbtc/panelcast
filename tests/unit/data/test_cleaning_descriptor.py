"""Descriptor-driven cleaning behavior on a non-AOTY (aero) domain.

The AOTY default path is covered by test_cleaning.py and the golden-hash
guard; these tests prove the same functions retarget to a different domain
purely through the descriptor.
"""

import pandas as pd
import pytest

from panelcast.config.descriptor import DEFAULT_DESCRIPTOR
from panelcast.data.cleaning import (
    CleaningConfig,
    clean_albums,
    filter_for_target_model,
)
from panelcast.data.lineage import AuditLogger
from panelcast.pipelines.prepare_dataset import PrepareConfig
from panelcast.pipelines.training_summary import TrainingSummary
from tests.helpers.aero_data import make_aero_descriptor as _aero_descriptor


def _make_aero_raw(**overrides) -> pd.DataFrame:
    data = {
        "Airframe": ["Falcon-X1", "Condor-7", "Raptor-M2"],
        "Flight ID": ["FX1-F01", "C7-F01", "RM2-F01"],
        "Campaign Year": [2021, 2022, 2022],
        "Flight Date": ["2021-03-15", "2022-07-01", None],
        "Perf Score": [7.5, 8.1, 6.2],
        "Sensor Samples": [120, 95, 60],
        "Test Crew": ["Falcon-X1", "Condor-7 + Chase-1", "Raptor-M2"],
        "original_row_id": [0, 1, 2],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _aero_cleaning_config() -> CleaningConfig:
    descriptor = _aero_descriptor()
    return CleaningConfig(min_year=descriptor.min_year, descriptor=descriptor)


class TestCleanAlbumsAeroDescriptor:
    """clean_albums retargeted by descriptor alone."""

    def test_renames_and_parses_iso_dates(self):
        cleaned = clean_albums(_make_aero_raw(), config=_aero_cleaning_config())
        assert "Perf_Score" in cleaned.columns
        assert "Flight_Date_Parsed" in cleaned.columns
        assert cleaned.loc[0, "Flight_Date_Parsed"] == pd.Timestamp("2021-03-15")
        assert cleaned.loc[0, "date_risk"] == "low"

    def test_year_fallback_uses_descriptor_year_col(self):
        cleaned = clean_albums(_make_aero_raw(), config=_aero_cleaning_config())
        # Row 2 has no Flight Date but a Campaign Year -> jan1 imputation.
        assert cleaned.loc[2, "date_risk"] == "medium"
        assert cleaned.loc[2, "Flight_Date_Parsed"] == pd.Timestamp("2022-01-01")
        assert not bool(cleaned.loc[2, "date_missing"])

    def test_event_col_shared_with_year_col_stays_numeric(self):
        descriptor = _aero_descriptor()
        descriptor = descriptor.model_copy(update={"event_col": descriptor.year_col})
        config = CleaningConfig(min_year=descriptor.min_year, descriptor=descriptor)

        cleaned = clean_albums(_make_aero_raw(), config=config)

        assert cleaned[descriptor.year_col].tolist() == [2021, 2022, 2022]
        assert not bool(cleaned["flag_future_year"].any())

    def test_no_genres_column_skips_primary_genre(self):
        cleaned = clean_albums(_make_aero_raw(), config=_aero_cleaning_config())
        assert "primary_genre" not in cleaned.columns

    def test_no_sentinel_skips_unknown_entity_flag(self):
        cleaned = clean_albums(_make_aero_raw(), config=_aero_cleaning_config())
        assert "is_unknown_artist" not in cleaned.columns

    def test_collaboration_uses_descriptor_separator(self):
        cleaned = clean_albums(_make_aero_raw(), config=_aero_cleaning_config())
        assert cleaned["num_artists"].tolist() == [1, 2, 1]
        assert cleaned["is_collaboration"].tolist() == [False, True, False]
        assert cleaned.loc[1, "collab_type"] == "duo"

    def test_no_multi_entity_col_skips_collaboration(self):
        descriptor = _aero_descriptor().model_copy(update={"multi_entity_col": None})
        config = CleaningConfig(min_year=descriptor.min_year, descriptor=descriptor)
        cleaned = clean_albums(_make_aero_raw(), config=config)
        assert "num_artists" not in cleaned.columns
        assert "is_collaboration" not in cleaned.columns


class TestNumericEntityIdCoercion:
    """All-digit catalog IDs (Gaia DR3 source_ids) must ingest as strings (#250).

    pandas reads numeric ID columns as int64; left numeric they reach
    ``artist_to_idx`` as int keys and only fail training-summary validation
    after the fit has run. Coercion at ingest keeps them strings end to end.
    """

    def _numeric_raw(self) -> pd.DataFrame:
        # Airframe / Flight ID as all-digit ints, the way pandas parses a
        # Gaia DR3 source_id column from CSV (int64, not object).
        return pd.DataFrame(
            {
                "Airframe": [4295806720, 4295806720, 34361129088],
                "Flight ID": [10001, 10002, 10003],
                "Campaign Year": [2021, 2021, 2022],
                "Flight Date": ["2021-03-15", "2021-05-01", "2022-07-01"],
                "Perf Score": [7.5, 8.1, 6.2],
                "Sensor Samples": [120, 95, 60],
                "Test Crew": ["4295806720", "4295806720", "34361129088"],
                "original_row_id": [0, 1, 2],
            }
        )

    def test_entity_and_event_columns_become_strings(self):
        cleaned = clean_albums(
            self._numeric_raw(),
            config=CleaningConfig(min_year=2015, descriptor=_aero_descriptor()),
        )
        assert cleaned["Airframe"].map(type).eq(str).all()
        assert cleaned["Flight_ID"].map(type).eq(str).all()
        # Exact catalog ID, no float artifacts.
        assert cleaned.loc[0, "Airframe"] == "4295806720"
        assert cleaned.loc[0, "Flight_ID"] == "10001"

    def test_artist_to_idx_keys_are_strings_end_to_end(self):
        descriptor = _aero_descriptor()
        cleaned = clean_albums(
            self._numeric_raw(),
            config=CleaningConfig(min_year=2015, descriptor=descriptor),
        )
        filtered = filter_for_target_model(cleaned, descriptor, 5)
        # Reproduce the train stage's artist_to_idx construction.
        artists = sorted(filtered[descriptor.entity_col].unique())
        artist_to_idx = {a: i for i, a in enumerate(artists)}
        assert all(isinstance(k, str) for k in artist_to_idx)
        # The typed summary requires str keys; numeric IDs would fail here.
        summary = TrainingSummary(artist_to_idx=artist_to_idx)
        assert set(summary.artist_to_idx) == set(artist_to_idx)

    def test_missing_entity_preserved_as_nan_not_string(self):
        raw = pd.DataFrame(
            {
                # None among ints -> pandas upcasts to float64; the coercion
                # must render clean integers (no ".0") and leave the gap as
                # NaN, never the string "nan".
                "Airframe": [4295806720, None, 34361129088],
                "Flight ID": [10001, 10002, 10003],
                "Campaign Year": [2021, 2021, 2022],
                "Flight Date": ["2021-03-15", "2021-05-01", "2022-07-01"],
                "Perf Score": [7.5, 8.1, 6.2],
                "Sensor Samples": [120, 95, 60],
                "Test Crew": ["a", "b", "c"],
                "original_row_id": [0, 1, 2],
            }
        )
        descriptor = _aero_descriptor()
        cleaned = clean_albums(
            raw, config=CleaningConfig(min_year=2015, descriptor=descriptor)
        )
        assert cleaned["Airframe"].isna().tolist() == [False, True, False]
        assert "nan" not in cleaned["Airframe"].dropna().tolist()
        # The missing-entity row is dropped by the existing identifier filter.
        filtered = filter_for_target_model(cleaned, descriptor, 5)
        assert filtered["Airframe"].tolist() == ["4295806720", "34361129088"]


class TestFilterForTargetModelAero:
    """Generic target filter driven by descriptor columns and bounds."""

    def _cleaned(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Airframe": ["A", "B", "C", "D", ""],
                "Flight_ID": ["F1", "F2", "F3", "F4", "F5"],
                "Perf_Score": [7.5, 11.0, None, 6.0, 5.0],
                "Sensor_Samples": [120, 95, 60, 3, 50],
                "original_row_id": [0, 1, 2, 3, 4],
            }
        )

    def test_bounds_and_threshold_filtering(self):
        descriptor = _aero_descriptor()
        filtered = filter_for_target_model(self._cleaned(), descriptor, 5)
        # Row B: 11.0 outside [0, 10]; row C: missing score; row D: 3 < 5
        # samples; row E: empty Airframe. Only A survives.
        assert filtered["Airframe"].tolist() == ["A"]

    def test_reason_strings_derive_from_descriptor(self, tmp_path):
        descriptor = _aero_descriptor()
        logger = AuditLogger(output_dir=tmp_path, run_id="aero")
        filter_for_target_model(self._cleaned(), descriptor, 5, logger=logger)
        assert [s.filter_name for s in logger.filter_stats] == [
            "missing_airframe_or_flight_id_identifier",
            "missing_perf_score",
            "invalid_perf_score_range",
            "below_min_sensor_samples_5",
        ]

    def test_secondary_target_requires_secondary_descriptor(self):
        descriptor = _aero_descriptor()
        with pytest.raises(ValueError, match="no secondary target"):
            filter_for_target_model(self._cleaned(), descriptor, 1, target="secondary")

    def test_invalid_target_name_raises(self):
        with pytest.raises(ValueError, match="primary"):
            filter_for_target_model(self._cleaned(), _aero_descriptor(), 1, target="tertiary")

    def test_aoty_default_reason_strings_unchanged(self, tmp_path):
        """The AOTY-default descriptor path must keep its historical audit reasons."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "Album": ["X", "Y"],
                "User_Score": [80.0, 85.0],
                "User_Ratings": [100, 5],
                "original_row_id": [0, 1],
            }
        )
        logger = AuditLogger(output_dir=tmp_path, run_id="aoty")
        filter_for_target_model(df, DEFAULT_DESCRIPTOR, 10, logger=logger)
        assert [s.filter_name for s in logger.filter_stats] == [
            "missing_artist_or_album_identifier",
            "missing_user_score",
            "invalid_user_score_range",
            "below_min_ratings_10",
        ]


class TestPrepareConfigDescriptorResolution:
    """PrepareConfig fields left as None resolve from the descriptor."""

    def test_raw_path_resolves_from_descriptor_env(self, monkeypatch):
        monkeypatch.setenv("AERO_DATASET_PATH", "custom/flights.csv")
        cfg = PrepareConfig(descriptor=_aero_descriptor())
        assert cfg.raw_path == "custom/flights.csv"

    def test_raw_path_falls_back_to_descriptor_default(self, monkeypatch):
        monkeypatch.delenv("AERO_DATASET_PATH", raising=False)
        cfg = PrepareConfig(descriptor=_aero_descriptor())
        assert cfg.raw_path == "data/raw/test_flights.csv"

    def test_thresholds_and_primary_resolve_from_descriptor(self):
        cfg = PrepareConfig(descriptor=_aero_descriptor())
        assert cfg.min_ratings_thresholds == [5, 10, 25]
        assert cfg.primary_min_ratings == 5

    def test_cleaning_carries_descriptor_and_min_year(self):
        descriptor = _aero_descriptor()
        cfg = PrepareConfig(descriptor=descriptor)
        assert cfg.cleaning is not None
        assert cfg.cleaning.descriptor is descriptor
        assert cfg.cleaning.min_year == 2015

    def test_explicit_fields_win_over_descriptor(self):
        cfg = PrepareConfig(
            descriptor=_aero_descriptor(),
            raw_path="explicit.csv",
            min_ratings_thresholds=[10],
            primary_min_ratings=10,
        )
        assert cfg.raw_path == "explicit.csv"
        assert cfg.min_ratings_thresholds == [10]
        assert cfg.primary_min_ratings == 10
