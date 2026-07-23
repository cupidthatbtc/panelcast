"""The committed aerospace example must stay in sync with its generator.

`examples/aerospace/flights.csv` is committed so the quickstart / `panelcast
demo` need no test-suite imports. This guard fails if the generator drifts from
the fixture (regenerate with `python scripts/generate_aero_example.py`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from panelcast.config.descriptor import load_descriptor, resolve_descriptor_path
from tests.helpers.aero_data import make_aero_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CSV = REPO_ROOT / "examples" / "aerospace" / "flights.csv"
DESCRIPTOR = EXAMPLE_CSV.parent / "descriptor.yaml"
PACKAGE_DATA = REPO_ROOT / "src" / "panelcast" / "_data"


def test_example_csv_exists():
    assert EXAMPLE_CSV.exists(), "examples/aerospace/flights.csv is missing"
    assert DESCRIPTOR.exists(), "examples/aerospace/descriptor.yaml is missing"


def test_packaged_data_matches_checkout_copies():
    pairs = [
        (DESCRIPTOR, PACKAGE_DATA / "examples" / "aerospace" / "descriptor.yaml"),
        (EXAMPLE_CSV, PACKAGE_DATA / "examples" / "aerospace" / "flights.csv"),
        (
            REPO_ROOT / "configs" / "datasets" / "aoty_full.yaml",
            PACKAGE_DATA / "datasets" / "aoty_full.yaml",
        ),
    ]
    for checkout, packaged in pairs:
        assert packaged.read_bytes() == checkout.read_bytes()

    checkout_aero = yaml.safe_load(
        (REPO_ROOT / "configs" / "datasets" / "aero.yaml").read_text(encoding="utf-8")
    )
    packaged_aero = yaml.safe_load(
        (PACKAGE_DATA / "datasets" / "aero.yaml").read_text(encoding="utf-8")
    )
    assert packaged_aero.pop("raw_path_default") == "examples/aerospace/flights.csv"
    checkout_aero.pop("raw_path_default")
    assert packaged_aero == checkout_aero


def test_bare_aero_descriptor_works_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    path = resolve_descriptor_path("aero")
    descriptor = load_descriptor("aero")

    assert path == PACKAGE_DATA / "datasets" / "aero.yaml"
    assert descriptor.resolve_raw_path() == PACKAGE_DATA / "examples" / "aerospace" / "flights.csv"


def test_example_csv_matches_generator():
    committed = pd.read_csv(EXAMPLE_CSV)
    generated = make_aero_dataset(seed=42)
    pd.testing.assert_frame_equal(
        committed.reset_index(drop=True),
        generated.reset_index(drop=True),
        check_dtype=False,
    )
