"""Regenerate the committed aerospace example fixture.

Keeps ``examples/aerospace/flights.csv`` in sync with the synthetic generator
in ``tests/helpers/aero_data.py`` (the same generator the domain-portability
e2e test uses). The CSV is committed so the quickstart / ``panelcast demo`` need
no test-suite imports; run this script if the generator changes.

    python scripts/generate_aero_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so `tests.helpers` resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.helpers.aero_data import make_aero_dataset  # noqa: E402

EXAMPLE_DIR = Path("examples/aerospace")
SEED = 42


def main() -> None:
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    df = make_aero_dataset(seed=SEED)
    csv_path = EXAMPLE_DIR / "flights.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(df)} flights, {df['Airframe'].nunique()} airframes)")


if __name__ == "__main__":
    main()
