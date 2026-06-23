"""The committed aerospace example must stay in sync with its generator.

`examples/aerospace/flights.csv` is committed so the quickstart / `panelcast
demo` need no test-suite imports. This guard fails if the generator drifts from
the fixture (regenerate with `python scripts/generate_aero_example.py`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.helpers.aero_data import make_aero_dataset

EXAMPLE_CSV = Path(__file__).resolve().parents[2] / "examples" / "aerospace" / "flights.csv"
DESCRIPTOR = EXAMPLE_CSV.parent / "descriptor.yaml"


def test_example_csv_exists():
    assert EXAMPLE_CSV.exists(), "examples/aerospace/flights.csv is missing"
    assert DESCRIPTOR.exists(), "examples/aerospace/descriptor.yaml is missing"


def test_example_csv_matches_generator():
    committed = pd.read_csv(EXAMPLE_CSV)
    generated = make_aero_dataset(seed=42)
    pd.testing.assert_frame_equal(
        committed.reset_index(drop=True),
        generated.reset_index(drop=True),
        check_dtype=False,
    )
