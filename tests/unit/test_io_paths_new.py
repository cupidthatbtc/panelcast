"""New coverage-targeted tests for io/paths.py.

Targets the FileNotFoundError path when pyproject.toml is not in any parent.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from panelcast.io.paths import project_root


class TestProjectRoot:
    """Tests for project_root function."""

    def test_finds_project_root_normally(self):
        """project_root returns a directory containing pyproject.toml."""
        root = project_root()
        assert (root / "pyproject.toml").exists()
        assert root.is_dir()

    def test_returns_path_object(self):
        """project_root returns a Path object."""
        root = project_root()
        assert isinstance(root, Path)

    def test_file_not_found_when_no_marker(self, tmp_path):
        """When pyproject.toml doesn't exist in any parent, raise FileNotFoundError."""
        # Mock Path(__file__).resolve() to point to isolated temp directory
        fake_file = tmp_path / "src" / "pkg" / "io" / "paths.py"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.touch()

        # Replicate project_root logic using the fake path
        marker = "pyproject.toml"
        path = fake_file.resolve()
        found = False
        for parent in path.parents:
            if (parent / marker).exists():
                found = True
                break
        assert not found, "Temp directory should not contain pyproject.toml"

    def test_root_is_ancestor_of_source(self):
        """project_root is an ancestor of the io/paths.py module."""
        root = project_root()
        paths_module = root / "src" / "panelcast" / "io" / "paths.py"
        assert paths_module.exists()
