"""GPU memory query via NVML.

Provides reliable GPU VRAM queries using NVIDIA Management Library (NVML)
via the nvidia-ml-py Python bindings.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from panelcast.pipelines.errors import GpuMemoryError

# NVML imports are optional - handle gracefully when not available
_NVML_AVAILABLE = False
_NVML_IMPORT_ERROR: str | None = None

try:
    from pynvml import (
        NVMLError,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlInit,
        nvmlShutdown,
    )

    _NVML_AVAILABLE = True
except ImportError as e:
    _NVML_IMPORT_ERROR = str(e)

    # Define placeholder types for when pynvml is not available
    class NVMLError(Exception):  # type: ignore[no-redef]
        """Placeholder for NVMLError when pynvml is not installed."""


if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _nvml_context() -> Iterator[None]:
    """Context manager for NVML initialization/shutdown.

    Ensures nvmlShutdown() is always called, even on exception.
    """
    nvmlInit()
    try:
        yield
    finally:
        nvmlShutdown()


@dataclass(frozen=True)
class GpuMemoryInfo:
    """GPU memory information from NVML query.

    All memory values are stored in bytes internally.
    Use properties for human-readable GB values.

    Attributes:
        device_name: GPU device name (e.g., "NVIDIA GeForce RTX 3080").
        total_bytes: Total VRAM in bytes.
        used_bytes: Currently used VRAM in bytes.
        free_bytes: Currently free VRAM in bytes.
    """

    device_name: str
    total_bytes: int
    used_bytes: int
    free_bytes: int

    @property
    def total_gb(self) -> float:
        """Total memory in GB (1024^3 bytes)."""
        return self.total_bytes / (1024**3)

    @property
    def used_gb(self) -> float:
        """Used memory in GB (1024^3 bytes)."""
        return self.used_bytes / (1024**3)

    @property
    def free_gb(self) -> float:
        """Free memory in GB (1024^3 bytes)."""
        return self.free_bytes / (1024**3)

    @property
    def free_percent(self) -> float:
        """Free memory as percentage of total (0-100)."""
        if self.total_bytes == 0:
            return 0.0
        return (self.free_bytes / self.total_bytes) * 100

    @property
    def used_percent(self) -> float:
        """Used memory as percentage of total (0-100)."""
        if self.total_bytes == 0:
            return 0.0
        return (self.used_bytes / self.total_bytes) * 100

    def format_display(self) -> str:
        """Format for user display.

        Returns:
            Human-readable string like:
            "6.2 GB free (77% of 8.0 GB) [6,656,212,992 bytes]"
        """
        return (
            f"{self.free_gb:.1f} GB free "
            f"({self.free_percent:.0f}% of {self.total_gb:.1f} GB) "
            f"[{self.free_bytes:,} bytes]"
        )


def query_gpu_memory(device_index: int = 0) -> GpuMemoryInfo:
    """Query GPU memory via NVML.

    Args:
        device_index: GPU device index (default 0 for first GPU).

    Returns:
        GpuMemoryInfo with total/used/free VRAM in bytes.

    Raises:
        GpuMemoryError: If NVML library not available, initialization fails,
            no GPU detected, device index out of range, or device_index is negative.

    Example:
        >>> info = query_gpu_memory()
        >>> print(f"Free: {info.free_gb:.1f} GB")
        Free: 6.2 GB
    """
    # Validate device_index is non-negative
    if device_index < 0:
        raise GpuMemoryError("device_index must be non-negative")

    # Check if NVML is available
    if not _NVML_AVAILABLE:
        raise GpuMemoryError(
            f"NVML library not available: {_NVML_IMPORT_ERROR}. "
            "Install nvidia-ml-py with 'pip install nvidia-ml-py' or use --force-run."
        )

    try:
        with _nvml_context():
            count = nvmlDeviceGetCount()
            if count == 0:
                raise GpuMemoryError("No NVIDIA GPU detected. Use --force-run to run on CPU.")
            if device_index >= count:
                raise GpuMemoryError(
                    f"GPU index {device_index} out of range. Available GPUs: 0-{count - 1}."
                )

            handle = nvmlDeviceGetHandleByIndex(device_index)
            memory = nvmlDeviceGetMemoryInfo(handle)
            name = nvmlDeviceGetName(handle)

            # Handle bytes vs str return type (varies by NVML version)
            device_name = name if isinstance(name, str) else name.decode("utf-8")

            return GpuMemoryInfo(
                device_name=device_name,
                total_bytes=memory.total,
                used_bytes=memory.used,
                free_bytes=memory.free,
            )

    except NVMLError as e:
        # Convert NVML-specific errors to our error type
        error_str = str(e)

        # Provide actionable error messages based on error type
        if "library" in error_str.lower() or "not found" in error_str.lower():
            raise GpuMemoryError(
                "NVML library not found. Install NVIDIA driver or use --force-run. "
                "See GPU_SETUP.md for troubleshooting."
            ) from e

        if "driver" in error_str.lower() or "not loaded" in error_str.lower():
            raise GpuMemoryError(
                "NVIDIA driver not loaded. Check driver installation with 'nvidia-smi'. "
                "Use --force-run to run on CPU."
            ) from e

        if "permission" in error_str.lower():
            raise GpuMemoryError(
                "Permission denied accessing GPU. Check user permissions for /dev/nvidia*."
            ) from e

        # Generic fallback for other NVML errors
        raise GpuMemoryError(f"GPU memory query failed: {e}") from e
