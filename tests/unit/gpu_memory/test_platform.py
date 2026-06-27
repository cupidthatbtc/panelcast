"""Tests for platform detection module."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
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


# --- from unit/gpu_memory/test_platform_expanded.py ---


class TestPlatformTypeValues:
    """Tests for PlatformType enum values and membership."""

    def test_all_values_are_strings(self):
        for pt in PlatformType:
            assert isinstance(pt.value, str)

    def test_member_count(self):
        assert len(PlatformType) == 6

    def test_unique_values(self):
        values = [pt.value for pt in PlatformType]
        assert len(values) == len(set(values))


class TestPlatformInfoProperties:
    """Extended property tests for PlatformInfo."""

    def test_frozen(self):
        info = PlatformInfo(platform_type=PlatformType.NATIVE_LINUX)
        with pytest.raises(FrozenInstanceError):
            info.platform_type = PlatformType.WSL2

    def test_is_wsl_false_macos(self):
        info = PlatformInfo(platform_type=PlatformType.MACOS)
        assert info.is_wsl is False

    def test_is_wsl_false_windows(self):
        info = PlatformInfo(platform_type=PlatformType.WINDOWS)
        assert info.is_wsl is False

    def test_is_wsl_false_unknown(self):
        info = PlatformInfo(platform_type=PlatformType.UNKNOWN)
        assert info.is_wsl is False

    def test_supports_gpu_unknown_false(self):
        info = PlatformInfo(platform_type=PlatformType.UNKNOWN)
        assert info.supports_gpu is False

    def test_default_kernel_version_none(self):
        info = PlatformInfo(platform_type=PlatformType.NATIVE_LINUX)
        assert info.kernel_version is None

    def test_default_wsl_distro_none(self):
        info = PlatformInfo(platform_type=PlatformType.NATIVE_LINUX)
        assert info.wsl_distro is None

    def test_wsl2_with_full_metadata(self):
        info = PlatformInfo(
            platform_type=PlatformType.WSL2,
            kernel_version="5.15.133.1-microsoft-standard-WSL2",
            wsl_distro="Ubuntu-22.04",
        )
        assert info.is_wsl is True
        assert info.supports_gpu is True
        assert info.kernel_version == "5.15.133.1-microsoft-standard-WSL2"
        assert info.wsl_distro == "Ubuntu-22.04"

    def test_wsl1_no_gpu(self):
        info = PlatformInfo(
            platform_type=PlatformType.WSL1,
            kernel_version="4.4.0-microsoft",
            wsl_distro="Ubuntu",
        )
        assert info.is_wsl is True
        assert info.supports_gpu is False


class TestPlatformInfoEquality:
    """Tests for PlatformInfo equality."""

    def test_equal_same_type(self):
        a = PlatformInfo(platform_type=PlatformType.NATIVE_LINUX)
        b = PlatformInfo(platform_type=PlatformType.NATIVE_LINUX)
        assert a == b

    def test_unequal_different_type(self):
        a = PlatformInfo(platform_type=PlatformType.NATIVE_LINUX)
        b = PlatformInfo(platform_type=PlatformType.WSL2)
        assert a != b

    def test_unequal_different_metadata(self):
        a = PlatformInfo(platform_type=PlatformType.WSL2, kernel_version="5.15")
        b = PlatformInfo(platform_type=PlatformType.WSL2, kernel_version="5.10")
        assert a != b
