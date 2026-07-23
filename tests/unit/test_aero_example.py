"""The committed aerospace example must stay in sync with its generator.

`examples/aerospace/flights.csv` is committed so the quickstart / `panelcast
demo` need no test-suite imports. This guard fails if the generator drifts from
the fixture (regenerate with `python scripts/generate_aero_example.py`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from panelcast.config.descriptor import load_descriptor, resolve_descriptor_path
from panelcast.pipelines.prepare_dataset import PrepareConfig, prepare_datasets
from tests.helpers.aero_data import make_aero_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CSV = REPO_ROOT / "examples" / "aerospace" / "flights.csv"
DESCRIPTOR = EXAMPLE_CSV.parent / "descriptor.yaml"
PACKAGE_DATA = REPO_ROOT / "src" / "panelcast" / "_data"


def test_example_csv_exists():
    assert EXAMPLE_CSV.exists(), "examples/aerospace/flights.csv is missing"
    assert DESCRIPTOR.exists(), "examples/aerospace/descriptor.yaml is missing"


def test_packaged_data_matches_checkout_copies():
    packaged_descriptor = PACKAGE_DATA / "examples" / "aerospace" / "descriptor.yaml"
    aero_descriptors = [
        DESCRIPTOR,
        REPO_ROOT / "configs" / "datasets" / "aero.yaml",
        PACKAGE_DATA / "datasets" / "aero.yaml",
        packaged_descriptor,
    ]
    expected = DESCRIPTOR.read_bytes()
    assert all(path.read_bytes() == expected for path in aero_descriptors)
    assert (PACKAGE_DATA / "examples" / "aerospace" / "flights.csv").read_bytes() == (
        EXAMPLE_CSV.read_bytes()
    )
    assert (PACKAGE_DATA / "datasets" / "aoty_full.yaml").read_bytes() == (
        REPO_ROOT / "configs" / "datasets" / "aoty_full.yaml"
    ).read_bytes()

    hashes = {load_descriptor(path).descriptor_hash() for path in aero_descriptors}
    assert len(hashes) == 1


def test_bare_aero_descriptor_works_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    path = resolve_descriptor_path("aero")
    descriptor = load_descriptor("aero")

    assert path == PACKAGE_DATA / "datasets" / "aero.yaml"
    assert descriptor.resolve_raw_path() == PACKAGE_DATA / "examples" / "aerospace" / "flights.csv"


def test_repository_demo_descriptor_prepares_committed_data(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.delenv("AERO_DATASET_PATH", raising=False)
    descriptor = load_descriptor(Path("examples/aerospace/descriptor.yaml"))

    result = prepare_datasets(
        PrepareConfig(
            descriptor=descriptor,
            output_dir=str(tmp_path / "processed"),
            audit_dir=str(tmp_path / "audit"),
        )
    )

    assert result.load_metadata.row_count == len(pd.read_csv(EXAMPLE_CSV))
    assert result.datasets_created["perf_minobs_5"].exists()
    assert descriptor.resolve_raw_path().resolve() == EXAMPLE_CSV


def test_example_csv_matches_generator():
    committed = pd.read_csv(EXAMPLE_CSV)
    generated = make_aero_dataset(seed=42)
    pd.testing.assert_frame_equal(
        committed.reset_index(drop=True),
        generated.reset_index(drop=True),
        check_dtype=False,
    )
