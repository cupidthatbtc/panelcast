"""Expanded tests for PlatformInfo and PlatformType."""

from dataclasses import FrozenInstanceError

import pytest

from panelcast.gpu_memory.platform import PlatformInfo, PlatformType


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
