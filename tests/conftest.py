"""Pytest configuration and fixtures.

The sys.path manipulation below enables running tests directly via
`pixi run test` without requiring `pip install -e .`. This is simpler
for a research project where the package isn't distributed.

For production projects, prefer using `pip install -e .` and removing
the path manipulation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
