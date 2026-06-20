"""Tests for platform detection module."""

from __future__ import annotations

from unittest import mock

import pytest

from panelcast.gpu_memory.platform import (
    PlatformInfo,
    PlatformType,
    detect_platform,
)


class TestPlatformType:
    """Tests for PlatformType enum."""

    def test_all_platform_types_exist(self):
        """All expected platform types are defined."""
        assert PlatformType.NATIVE_LINUX.value == "native_linux"
        assert PlatformType.WSL2.value == "wsl2"
        assert PlatformType.WSL1.value == "wsl1"
        assert PlatformType.MACOS.value == "macos"
        assert PlatformType.WINDOWS.value == "windows"
        assert PlatformType.UNKNOWN.value == "unknown"


class TestPlatformInfo:
    """Tests for PlatformInfo dataclass."""

    def test_frozen_dataclass(self):
        """PlatformInfo is immutable."""
        info = PlatformInfo(platform_type=PlatformType.NATIVE_LINUX)
        with pytest.raises(AttributeError):
            info.platform_type = PlatformType.WSL2  # type: ignore[misc]

    def test_is_wsl_true_for_wsl1(self):
        """is_wsl returns True for WSL1."""
        info = PlatformInfo(platform_type=PlatformType.WSL1)
        assert info.is_wsl is True

    def test_is_wsl_true_for_wsl2(self):
        """is_wsl returns True for WSL2."""
        info = PlatformInfo(platform_type=PlatformType.WSL2)
        assert info.is_wsl is True

    def test_is_wsl_false_for_native_linux(self):
        """is_wsl returns False for native Linux."""
        info = PlatformInfo(platform_type=PlatformType.NATIVE_LINUX)
        assert info.is_wsl is False

    def test_supports_gpu_native_linux(self):
        """supports_gpu returns True for native Linux."""
        info = PlatformInfo(platform_type=PlatformType.NATIVE_LINUX)
        assert info.supports_gpu is True

    def test_supports_gpu_wsl2(self):
        """supports_gpu returns True for WSL2."""
        info = PlatformInfo(platform_type=PlatformType.WSL2)
        assert info.supports_gpu is True

    def test_supports_gpu_wsl1_false(self):
        """supports_gpu returns False for WSL1 (no GPU passthrough)."""
        info = PlatformInfo(platform_type=PlatformType.WSL1)
        assert info.supports_gpu is False

    def test_supports_gpu_macos_false(self):
        """supports_gpu returns False for macOS (no NVIDIA CUDA)."""
        info = PlatformInfo(platform_type=PlatformType.MACOS)
        assert info.supports_gpu is False

    def test_supports_gpu_windows(self):
        """supports_gpu returns True for Windows (CUDA available)."""
        info = PlatformInfo(platform_type=PlatformType.WINDOWS)
        assert info.supports_gpu is True

    def test_optional_fields(self):
        """kernel_version and wsl_distro are optional."""
        info = PlatformInfo(
            platform_type=PlatformType.WSL2,
            kernel_version="5.15.0",
            wsl_distro="Ubuntu",
        )
        assert info.kernel_version == "5.15.0"
        assert info.wsl_distro == "Ubuntu"


class TestDetectPlatform:
    """Tests for detect_platform() function."""

    def test_detects_macos(self):
        """detect_platform returns MACOS on Darwin."""
        with mock.patch("panelcast.gpu_memory.platform.platform.system", return_value="Darwin"):
            result = detect_platform()
        assert result.platform_type == PlatformType.MACOS

    def test_detects_windows(self):
        """detect_platform returns WINDOWS on Windows."""
        with mock.patch("panelcast.gpu_memory.platform.platform.system", return_value="Windows"):
            result = detect_platform()
        assert result.platform_type == PlatformType.WINDOWS

    def test_detects_unknown_system(self):
        """detect_platform returns UNKNOWN for unrecognized OS."""
        with mock.patch("panelcast.gpu_memory.platform.platform.system", return_value="FreeBSD"):
            result = detect_platform()
        assert result.platform_type == PlatformType.UNKNOWN

    def test_detects_native_linux(self):
        """detect_platform returns NATIVE_LINUX when no WSL indicators."""
        mock_path = mock.MagicMock()
        mock_path.return_value.read_text.return_value = "Linux version 5.15.0"
        mock_path.return_value.exists.return_value = False

        with (
            mock.patch("panelcast.gpu_memory.platform.platform.system", return_value="Linux"),
            mock.patch("panelcast.gpu_memory.platform.Path", mock_path),
            mock.patch.dict("os.environ", {}, clear=True),
        ):
            result = detect_platform()
        assert result.platform_type == PlatformType.NATIVE_LINUX

    def test_detects_wsl2_by_proc_version(self):
        """detect_platform returns WSL2 when /proc/version contains 'microsoft'."""
        mock_path = mock.MagicMock()
        mock_path.return_value.read_text.return_value = (
            "linux version 5.15.0-microsoft-standard-wsl2"
        )
        mock_path.return_value.exists.return_value = True  # /run/WSL exists

        with (
            mock.patch("panelcast.gpu_memory.platform.platform.system", return_value="Linux"),
            mock.patch("panelcast.gpu_memory.platform.Path", mock_path),
            mock.patch.dict("os.environ", {"WSL_DISTRO_NAME": "Ubuntu"}, clear=True),
        ):
            result = detect_platform()
        assert result.platform_type == PlatformType.WSL2
        assert result.is_wsl is True

    def test_detects_wsl1_without_wsl2_indicators(self):
        """detect_platform returns WSL1 when WSL but no WSL2 indicators."""
        mock_path = mock.MagicMock()
        # WSL1 has 4.4.x kernel
        mock_path.return_value.read_text.return_value = "linux version 4.4.0-microsoft"
        mock_path.return_value.exists.return_value = False  # /run/WSL doesn't exist

        with (
            mock.patch("panelcast.gpu_memory.platform.platform.system", return_value="Linux"),
            mock.patch("panelcast.gpu_memory.platform.Path", mock_path),
            mock.patch.dict("os.environ", {"WSL_DISTRO_NAME": "Ubuntu"}, clear=True),
        ):
            result = detect_platform()
        assert result.platform_type == PlatformType.WSL1

    def test_handles_proc_version_not_found(self):
        """detect_platform handles missing /proc/version gracefully."""
        mock_path = mock.MagicMock()
        mock_path.return_value.read_text.side_effect = FileNotFoundError
        mock_path.return_value.exists.return_value = False  # /run/WSL doesn't exist

        with (
            mock.patch("panelcast.gpu_memory.platform.platform.system", return_value="Linux"),
            mock.patch("panelcast.gpu_memory.platform.Path", mock_path),
            mock.patch.dict("os.environ", {}, clear=True),
        ):
            result = detect_platform()
        # Should still work, defaulting to native Linux
        assert result.platform_type == PlatformType.NATIVE_LINUX
