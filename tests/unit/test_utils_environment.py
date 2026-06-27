"""Tests for environment verification utilities."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from panelcast.utils.environment import (
    EnvironmentError,
    EnvironmentStatus,
    ensure_environment_locked,
    verify_environment,
)


class TestEnvironmentStatus:
    """Tests for EnvironmentStatus dataclass."""

    def test_create_with_pixi_lock(self):
        """EnvironmentStatus can be created when pixi.lock exists."""
        status = EnvironmentStatus(
            pixi_lock_exists=True,
            pixi_lock_path=Path("/project/pixi.lock"),
            pixi_lock_hash="abc123def456",
            is_reproducible=True,
            warnings=[],
        )
        assert status.pixi_lock_exists is True
        assert status.pixi_lock_path == Path("/project/pixi.lock")
        assert status.pixi_lock_hash == "abc123def456"
        assert status.is_reproducible is True
        assert status.warnings == []

    def test_create_without_pixi_lock(self):
        """EnvironmentStatus can be created when pixi.lock missing."""
        status = EnvironmentStatus(
            pixi_lock_exists=False,
            pixi_lock_path=None,
            pixi_lock_hash=None,
            is_reproducible=False,
            warnings=["pixi.lock not found"],
        )
        assert status.pixi_lock_exists is False
        assert status.pixi_lock_path is None
        assert status.pixi_lock_hash is None
        assert status.is_reproducible is False
        assert len(status.warnings) == 1

    def test_warnings_default_empty(self):
        """warnings defaults to empty list."""
        status = EnvironmentStatus(
            pixi_lock_exists=True,
            pixi_lock_path=Path("/project/pixi.lock"),
            pixi_lock_hash="abc123",
            is_reproducible=True,
        )
        assert status.warnings == []


class TestVerifyEnvironment:
    """Tests for verify_environment function."""

    def test_verify_when_pixi_lock_exists(self):
        """verify_environment returns success when pixi.lock exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            lock_file = tmppath / "pixi.lock"
            lock_file.write_text("test lock content")

            status = verify_environment(tmppath)

            assert status.pixi_lock_exists is True
            assert status.pixi_lock_path == lock_file
            assert status.is_reproducible is True
            assert len(status.warnings) == 0

    def test_verify_when_pixi_lock_missing(self):
        """verify_environment returns failure when pixi.lock missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            status = verify_environment(tmpdir)

            assert status.pixi_lock_exists is False
            assert status.pixi_lock_path is None
            assert status.pixi_lock_hash is None
            assert status.is_reproducible is False
            assert len(status.warnings) == 1
            assert "pixi.lock not found" in status.warnings[0]

    def test_verify_computes_hash(self):
        """verify_environment computes correct SHA256 hash of pixi.lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            lock_file = tmppath / "pixi.lock"
            content = "test lock content for hashing"
            lock_file.write_text(content)

            # Compute expected hash
            expected_hash = hashlib.sha256(content.encode()).hexdigest()

            status = verify_environment(tmppath)

            assert status.pixi_lock_hash == expected_hash

    def test_verify_accepts_string_path(self):
        """verify_environment accepts string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / "pixi.lock"
            lock_file.write_text("content")

            status = verify_environment(tmpdir)  # Pass string, not Path

            assert status.pixi_lock_exists is True

    def test_verify_accepts_path_object(self):
        """verify_environment accepts Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            lock_file = tmppath / "pixi.lock"
            lock_file.write_text("content")

            status = verify_environment(tmppath)

            assert status.pixi_lock_exists is True

    def test_verify_searches_parent_directories(self):
        """verify_environment finds pixi.lock in parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subdir = tmppath / "src" / "module"
            subdir.mkdir(parents=True)

            # Put pixi.lock in root, not subdir
            lock_file = tmppath / "pixi.lock"
            lock_file.write_text("content")

            # Search from subdir
            status = verify_environment(subdir)

            assert status.pixi_lock_exists is True
            assert status.pixi_lock_path == lock_file

    def test_verify_warning_message_helpful(self):
        """verify_environment warning includes helpful instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            status = verify_environment(tmpdir)

            warning = status.warnings[0]
            assert "pixi install" in warning.lower()
            assert "reproducible" in warning.lower()


class TestEnsureEnvironmentLocked:
    """Tests for ensure_environment_locked function."""

    def test_strict_mode_raises_when_not_locked(self):
        """ensure_environment_locked raises in strict mode when not locked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(EnvironmentError) as exc_info:
                ensure_environment_locked(tmpdir, strict=True)

            assert "not locked" in str(exc_info.value).lower()

    def test_strict_mode_no_raise_when_locked(self):
        """ensure_environment_locked doesn't raise when locked in strict mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / "pixi.lock"
            lock_file.write_text("content")

            # Should not raise
            ensure_environment_locked(tmpdir, strict=True)

    def test_non_strict_mode_logs_warning(self):
        """ensure_environment_locked logs warning in non-strict mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("panelcast.utils.environment.log") as mock_log:
                ensure_environment_locked(tmpdir, strict=False)

                mock_log.warning.assert_called_once()
                call_args = mock_log.warning.call_args
                assert call_args[0][0] == "environment_not_locked"

    def test_non_strict_mode_no_warning_when_locked(self):
        """ensure_environment_locked doesn't log when locked in non-strict mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / "pixi.lock"
            lock_file.write_text("content")

            with patch("panelcast.utils.environment.log") as mock_log:
                ensure_environment_locked(tmpdir, strict=False)

                mock_log.warning.assert_not_called()

    def test_default_is_non_strict(self):
        """ensure_environment_locked defaults to non-strict mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise, just log warning
            with patch("panelcast.utils.environment.log"):
                ensure_environment_locked(tmpdir)  # No strict argument

    def test_strict_error_message_helpful(self):
        """ensure_environment_locked error message includes helpful instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(EnvironmentError) as exc_info:
                ensure_environment_locked(tmpdir, strict=True)

            error_msg = str(exc_info.value)
            assert "pixi install" in error_msg.lower()


class TestEnvironmentError:
    """Tests for EnvironmentError exception class."""

    def test_environment_error_is_exception(self):
        """EnvironmentError inherits from Exception."""
        assert issubclass(EnvironmentError, Exception)

    def test_environment_error_message(self):
        """EnvironmentError can be created with message."""
        error = EnvironmentError("Test message")
        assert str(error) == "Test message"

    def test_environment_error_can_be_raised(self):
        """EnvironmentError can be raised and caught."""
        with pytest.raises(EnvironmentError):
            raise EnvironmentError("Test")


class TestHashComputation:
    """Tests for hash computation correctness."""

    def test_hash_is_sha256(self):
        """pixi_lock_hash is a valid 64-character SHA256 hex string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            lock_file = tmppath / "pixi.lock"
            lock_file.write_text("test content")

            status = verify_environment(tmppath)

            # SHA256 produces 64 hex characters
            assert len(status.pixi_lock_hash) == 64
            # All characters should be hex digits
            assert all(c in "0123456789abcdef" for c in status.pixi_lock_hash)

    def test_hash_changes_with_content(self):
        """pixi_lock_hash changes when file content changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            lock_file = tmppath / "pixi.lock"

            lock_file.write_text("content version 1")
            status1 = verify_environment(tmppath)

            lock_file.write_text("content version 2")
            status2 = verify_environment(tmppath)

            assert status1.pixi_lock_hash != status2.pixi_lock_hash

    def test_hash_deterministic(self):
        """Same content produces same hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            lock_file = tmppath / "pixi.lock"
            lock_file.write_text("deterministic content")

            status1 = verify_environment(tmppath)
            status2 = verify_environment(tmppath)

            assert status1.pixi_lock_hash == status2.pixi_lock_hash


# --- from unit/test_utils_environment_expanded.py ---


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
