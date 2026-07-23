"""Synthetic aerospace test-flight dataset for domain-portability tests.

Deliberately different from AOTY in every domain-specific dimension:
ISO date format, [0, 10] target bounds, ``" + "`` multi-entity separator,
no genre/album-type analogues. A mild per-airframe random walk gives the
hierarchical model real signal to find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from panelcast.config.descriptor import DatasetDescriptor, FeatureBlockSpec


def make_aero_descriptor(**overrides) -> DatasetDescriptor:
    """Aero descriptor mirroring configs/datasets/aero.yaml."""
    fields: dict = dict(
        name="aero",
        raw_path_env="AERO_DATASET_PATH",
        raw_path_default="examples/aerospace/flights.csv",
        encoding="utf-8",
        raw_column_map={
            "Flight Date": "Flight_Date",
            "Perf Score": "Perf_Score",
            "Sensor Samples": "Sensor_Samples",
            "Test Crew": "Test_Crew",
            "Flight ID": "Flight_ID",
            "Campaign Year": "Year",
            "Thrust Margin": "Thrust_Margin",
            "Payload Fraction": "Payload_Fraction",
        },
        required_raw_columns=[
            "Airframe",
            "Flight ID",
            "Campaign Year",
            "Flight Date",
            "Perf Score",
            "Sensor Samples",
            "Test Crew",
        ],
        optional_raw_columns=[],
        entity_col="Airframe",
        event_col="Flight_ID",
        date_col="Flight_Date",
        parsed_date_col="Flight_Date_Parsed",
        year_col="Year",
        date_format="%Y-%m-%d",
        target_col="Perf_Score",
        target_bounds=(0.0, 10.0),
        model_prefix="perf",
        n_obs_col="Sensor_Samples",
        secondary_target_col=None,
        secondary_prefix=None,
        secondary_n_obs_col=None,
        multi_entity_col="Test_Crew",
        multi_entity_separator=" + ",
        unknown_entity_sentinel=None,
        min_year=2015,
        min_obs_thresholds=[5, 10, 25],
        primary_min_obs=5,
        processed_name_template="perf_minobs_{min_ratings}",
        feature_packs=[],
        feature_blocks=[
            FeatureBlockSpec(name="temporal"),
            FeatureBlockSpec(name="entity_history"),
            FeatureBlockSpec(
                name="core_numeric",
                params={"columns": ["Thrust_Margin", "Payload_Fraction"]},
            ),
        ],
        ablation_groups={"temporal": ["temporal"], "artist": ["entity_history"]},
    )
    fields.update(overrides)
    return DatasetDescriptor(**fields)


AIRFRAMES = [
    "Falcon-X1",
    "Condor-7",
    "Raptor-M2",
    "Albatross-3",
    "Kestrel-V",
    "Harrier-9",
    "Osprey-T4",
    "Swift-E2",
]


def make_aero_dataset(seed: int = 42) -> pd.DataFrame:
    """Generate ~44 sequential test flights across 8 airframes.

    Columns intentionally mirror the *shape* of a raw domain export, not
    AOTY's names: Airframe / Flight ID / Flight Date (ISO) / Campaign Year /
    Perf Score (0-10) / Sensor Samples / Test Crew / two numeric covariates.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for airframe_idx, airframe in enumerate(AIRFRAMES):
        n_flights = int(rng.integers(4, 8))
        # Per-airframe quality level + mild random-walk drift across flights.
        base_quality = float(rng.normal(6.0, 1.0))
        effect = 0.0
        start = pd.Timestamp("2021-01-15") + pd.Timedelta(days=int(rng.integers(0, 90)))

        for flight_num in range(1, n_flights + 1):
            effect += float(rng.normal(0.0, 0.25))
            date = start + pd.Timedelta(days=int(flight_num * rng.integers(20, 45)))
            thrust_margin = float(rng.normal(0.0, 1.0))
            payload_fraction = float(rng.uniform(0.2, 0.9))
            score = (
                base_quality
                + effect
                + 0.3 * thrust_margin
                - 0.5 * (payload_fraction - 0.5)
                + float(rng.normal(0.0, 0.4))
            )
            score = float(np.clip(score, 0.05, 9.95))

            solo = rng.random() < 0.7
            crew = airframe if solo else f"{airframe} + Chase-{int(rng.integers(1, 4))}"

            rows.append(
                {
                    "Airframe": airframe,
                    "Flight ID": f"{airframe}-F{flight_num:02d}",
                    "Flight Date": date.strftime("%Y-%m-%d"),
                    "Campaign Year": date.year,
                    "Perf Score": round(score, 2),
                    "Sensor Samples": int(rng.integers(8, 220)),
                    "Test Crew": crew,
                    "Thrust Margin": round(thrust_margin, 3),
                    "Payload Fraction": round(payload_fraction, 3),
                }
            )

    return pd.DataFrame(rows)
