"""Preflight check module for GPU memory verification.

This module provides preflight checking capabilities to verify GPU memory
availability before starting long-running MCMC sampling jobs.

The preflight system can:
- Estimate memory requirements for a given model configuration
- Compare against available GPU memory
- Report pass/fail/warning status
- Suggest configuration adjustments

Example:
    >>> from panelcast.preflight import PreflightStatus, PreflightResult
    >>> # After running preflight check:
    >>> if result.status == PreflightStatus.FAIL:
    ...     raise SystemExit(result.exit_code)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from panelcast.gpu_memory import MemoryEstimate


class PreflightStatus(Enum):
    """Status of preflight check.

    Values:
        PASS: Sufficient memory available, safe to proceed.
        FAIL: Insufficient memory, MCMC run will likely OOM.
        WARNING: Memory is tight, may work but risky.
        CANNOT_CHECK: Unable to determine memory availability
            (e.g., no GPU, NVML not available).
    """

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    CANNOT_CHECK = "cannot_check"


def _exit_code_for_status(status: PreflightStatus) -> int:
    """Map PreflightStatus to CLI exit code."""
    match status:
        case PreflightStatus.PASS:
            return 0
        case PreflightStatus.FAIL:
            return 1
        case PreflightStatus.WARNING | PreflightStatus.CANNOT_CHECK:
            return 2
        case _:
            raise ValueError(f"Unexpected PreflightStatus: {status}")


@dataclass(frozen=True)
class PreflightResult:
    """Result of preflight memory check using ESTIMATED memory.

    Provides detailed information about memory availability and
    estimated requirements, with actionable suggestions if needed.

    This class uses formula-based memory estimation (quick preflight).
    For actual measured memory via mini-MCMC, see FullPreflightResult.

    Attributes:
        status: Overall pass/fail/warning status.
        estimate: Memory estimate breakdown.
        available_gb: Available (free) GPU memory in GB.
        total_gpu_gb: Total GPU memory in GB.
        headroom_percent: Remaining memory percentage after estimate.
            Negative if estimate exceeds available.
        message: Human-readable summary message.
        suggestions: Tuple of configuration adjustment suggestions (immutable).
        device_name: GPU device name (None if cannot check).
    """

    status: PreflightStatus
    estimate: MemoryEstimate | None
    available_gb: float
    total_gpu_gb: float
    headroom_percent: float
    message: str
    suggestions: tuple[str, ...]
    device_name: str | None = None

    @property
    def exit_code(self) -> int:
        """Exit code for CLI usage.

        Returns:
            0 for PASS (safe to proceed).
            1 for FAIL (do not proceed).
            2 for WARNING or CANNOT_CHECK (proceed with caution).
        """
        return _exit_code_for_status(self.status)


@dataclass(frozen=True)
class FullPreflightResult:
    """Result of full preflight check using MEASURED memory.

    Provides detailed information about memory availability using
    actual peak GPU memory measured from a mini-MCMC run.

    This class uses actual memory measurement via subprocess mini-run
    (full preflight, ~95% accuracy). For formula-based estimation
    (quick preflight, ~70-80% accuracy), see PreflightResult.

    The mini-run executes 1 chain, 10 warmup, 1 sample to capture
    JIT compilation overhead, which is typically the peak memory usage.

    Attributes:
        status: Overall pass/fail/warning status.
        measured_peak_gb: Actual peak GPU memory from mini-run (not estimate).
        available_gb: Available (free) GPU memory in GB.
        total_gpu_gb: Total GPU memory in GB.
        headroom_percent: Remaining memory percentage after measured peak.
            Negative if measured peak exceeds available.
        mini_run_seconds: Time taken for mini-run in seconds.
        message: Human-readable summary message.
        suggestions: Tuple of configuration adjustment suggestions (immutable).
        device_name: GPU device name (None if cannot check).
    """

    status: PreflightStatus
    measured_peak_gb: float
    available_gb: float
    total_gpu_gb: float
    headroom_percent: float
    mini_run_seconds: float
    message: str
    suggestions: tuple[str, ...]
    device_name: str | None = None

    @property
    def exit_code(self) -> int:
        """Exit code for CLI usage.

        Returns:
            0 for PASS (safe to proceed).
            1 for FAIL (do not proceed).
            2 for WARNING or CANNOT_CHECK (proceed with caution).
        """
        return _exit_code_for_status(self.status)


@dataclass(frozen=True)
class ExtrapolationResult:
    """Result of full preflight with extrapolation to target sample count.

    Provides memory projection from two-point calibration measurements
    to the target sample count, with uncertainty bounds.

    The calibration runs mini-MCMC at 10 and 50 samples to fit a linear
    model separating fixed JIT overhead from per-sample cost, then
    extrapolates to the target sample count.

    Attributes:
        status: Overall pass/fail/warning status based on projected memory.
        measured_peak_gb: Peak memory from calibration runs (max of both).
        projected_gb: Extrapolated memory for target samples.
        target_samples: Number of samples we're projecting to.
        calibration_samples: Total samples used in calibration (e.g., 60 for 10+50).
        uncertainty_percent: Uncertainty range for projection (default 10%).
        available_gb: Available (free) GPU memory in GB.
        total_gpu_gb: Total GPU memory in GB.
        headroom_percent: Remaining memory percentage after projected usage.
            Based on projected_gb, not measured_peak_gb.
        calibration: The CalibrationResult used for extrapolation.
        from_cache: Whether calibration was loaded from cache.
        message: Human-readable summary message.
        suggestions: Tuple of configuration adjustment suggestions (immutable).
        device_name: GPU device name (None if cannot check).
    """

    status: PreflightStatus
    measured_peak_gb: float
    projected_gb: float
    target_samples: int
    calibration_samples: int
    uncertainty_percent: float
    available_gb: float
    total_gpu_gb: float
    headroom_percent: float
    calibration: "CalibrationResult"
    from_cache: bool
    message: str
    suggestions: tuple[str, ...]
    device_name: str | None = None

    @property
    def exit_code(self) -> int:
        """Exit code for CLI usage.

        Returns:
            0 for PASS (safe to proceed).
            1 for FAIL (do not proceed).
            2 for WARNING or CANNOT_CHECK (proceed with caution).
        """
        return _exit_code_for_status(self.status)


# Imports placed after class definitions to avoid circular imports.
# These modules import PreflightResult/FullPreflightResult from this module.
from panelcast.preflight.cache import compute_config_hash
from panelcast.preflight.calibrate import CalibrationError, CalibrationResult
from panelcast.preflight.check import run_preflight_check
from panelcast.preflight.full_check import (
    run_extrapolated_preflight_check,
    run_full_preflight_check,
)
from panelcast.preflight.output import (
    render_extrapolation_result,
    render_full_preflight_result,
    render_preflight_result,
)

__all__ = [
    "CalibrationError",
    "CalibrationResult",
    "ExtrapolationResult",
    "FullPreflightResult",
    "PreflightResult",
    "PreflightStatus",
    "compute_config_hash",
    "render_extrapolation_result",
    "render_full_preflight_result",
    "render_preflight_result",
    "run_extrapolated_preflight_check",
    "run_full_preflight_check",
    "run_preflight_check",
]
