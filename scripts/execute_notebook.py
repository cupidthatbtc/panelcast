"""Execute a notebook in a scratch directory with a timeout (#174).

Keeps the checkout clean (the quickstart writes data/models/outputs under its
own cwd) and fails loudly on any errored cell, so the marketing artifact can
never rot.

Usage: python scripts/execute_notebook.py examples/quickstart.ipynb [timeout_s]
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    notebook = Path(sys.argv[1]).resolve()
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / notebook.name
        shutil.copy2(notebook, staged)
        nb = nbformat.read(staged, as_version=4)
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": scratch}},
        )
        client.execute()
    print(f"{notebook.name}: all {len(nb.cells)} cells executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
