"""Expanded tests for environment verification utilities."""

from pathlib import Path

import pytest

from panelcast.utils.environment import EnvironmentStatus, verify_environment


class TestEnvironmentStatusDataclass:
    """Tests for EnvironmentStatus dataclass."""

    def test_fields_accessible(self):
        status = EnvironmentStatus(
            pixi_lock_exists=True,
            pixi_lock_path=Path("/some/pixi.lock"),
            pixi_lock_hash="abc123",
            is_reproducible=True,
            warnings=[],
        )
        assert status.pixi_lock_exists is True
        assert status.pixi_lock_hash == "abc123"
        assert status.is_reproducible is True
        assert status.warnings == []

    def test_with_warnings(self):
        status = EnvironmentStatus(
            pixi_lock_exists=True,
            pixi_lock_path=Path("/some/pixi.lock"),
            pixi_lock_hash="abc123",
            is_reproducible=True,
            warnings=["Package X version mismatch"],
        )
        assert len(status.warnings) == 1
        assert "Package X" in status.warnings[0]

    def test_not_reproducible(self):
        status = EnvironmentStatus(
            pixi_lock_exists=False,
            pixi_lock_path=None,
            pixi_lock_hash=None,
            is_reproducible=False,
            warnings=["No lock file found"],
        )
        assert status.pixi_lock_exists is False
        assert status.pixi_lock_hash is None
        assert status.is_reproducible is False

    def test_default_warnings(self):
        status = EnvironmentStatus(
            pixi_lock_exists=True,
            pixi_lock_path=Path("/tmp/pixi.lock"),
            pixi_lock_hash="def456",
            is_reproducible=True,
        )
        assert status.warnings == []


class TestVerifyEnvironmentExpanded:
    """Extended tests for verify_environment."""

    def test_returns_environment_status(self):
        result = verify_environment()
        assert isinstance(result, EnvironmentStatus)

    def test_result_has_all_fields(self):
        result = verify_environment()
        assert hasattr(result, "pixi_lock_exists")
        assert hasattr(result, "pixi_lock_hash")
        assert hasattr(result, "is_reproducible")
        assert hasattr(result, "warnings")

    def test_nonexistent_directory(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"
        result = verify_environment(project_root=nonexistent)
        assert result.pixi_lock_exists is False
        assert result.is_reproducible is False
