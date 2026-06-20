"""Pipeline exception hierarchy for error handling and exit codes.

This module defines a hierarchy of exceptions for pipeline failures, each
with a specific exit code for CLI integration. The exception classes
enable fail-fast semantics with descriptive error messages that include
stage context.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for pipeline failures.

    All pipeline-specific errors inherit from this class, enabling
    consistent error handling and exit codes.

    Attributes:
        message: Human-readable error description.
        stage: Name of the pipeline stage where error occurred (optional).
        exit_code: Process exit code for CLI integration.

    Example:
        >>> raise PipelineError("Something went wrong", stage="train")
        PipelineError: [train] Something went wrong
    """

    def __init__(
        self,
        message: str,
        stage: str = "",
        exit_code: int = 1,
    ) -> None:
        """Initialize pipeline error.

        Args:
            message: Error description.
            stage: Pipeline stage name (optional).
            exit_code: Exit code for CLI (default 1).
        """
        self.message = message
        self.stage = stage
        self.exit_code = exit_code
        super().__init__(str(self))

    def __str__(self) -> str:
        """Format error with stage prefix if available."""
        if self.stage:
            return f"[{self.stage}] {self.message}"
        return self.message


class ConvergenceError(PipelineError):
    """MCMC convergence failure (R-hat, ESS, divergences).

    Raised when model fitting fails convergence diagnostics:
    - R-hat exceeds threshold (default 1.01)
    - Effective sample size too low
    - Too many divergent transitions

    Exit code: 2
    """

    def __init__(self, message: str, stage: str = "") -> None:
        """Initialize convergence error with exit code 2."""
        super().__init__(message, stage=stage, exit_code=2)


class DataValidationError(PipelineError):
    """Input data validation failure.

    Raised when input data fails validation checks:
    - Missing required columns
    - Invalid data types
    - Out-of-range values
    - Schema violations

    Exit code: 3
    """

    def __init__(self, message: str, stage: str = "") -> None:
        """Initialize data validation error with exit code 3."""
        super().__init__(message, stage=stage, exit_code=3)


class StageError(PipelineError):
    """General stage execution error.

    Raised for stage-specific failures that don't fit other categories:
    - File I/O errors during stage
    - Missing dependencies
    - Computation failures

    Exit code: 4
    """

    def __init__(self, message: str, stage: str = "") -> None:
        """Initialize stage error with exit code 4."""
        super().__init__(message, stage=stage, exit_code=4)


class EnvironmentError(PipelineError):
    """Environment verification failure.

    Raised when environment reproducibility cannot be verified:
    - pixi.lock not found in strict mode
    - Environment hash mismatch
    - Missing required dependencies

    Exit code: 5
    """

    def __init__(self, message: str, stage: str = "") -> None:
        """Initialize environment error with exit code 5."""
        super().__init__(message, stage=stage, exit_code=5)


class GpuMemoryError(PipelineError):
    """GPU memory check failure.

    Raised when:
    - NVML initialization fails (no GPU, driver not loaded)
    - Insufficient GPU memory for planned operation
    - GPU memory query fails unexpectedly

    Exit code: 6
    """

    def __init__(self, message: str, stage: str = "gpu_check") -> None:
        """Initialize GPU memory error with exit code 6."""
        super().__init__(message, stage=stage, exit_code=6)


class StageSkipped(Exception):
    """Control flow exception for skipped stages.

    Not an error - used internally to signal that a stage was skipped
    due to unchanged inputs. Does not inherit from PipelineError since
    it's not a failure condition.

    Attributes:
        message: Explanation of why the stage was skipped.

    Example:
        >>> raise StageSkipped("Inputs unchanged since last run")
    """

    def __init__(self, message: str) -> None:
        """Initialize with skip reason.

        Args:
            message: Explanation of why stage was skipped.
        """
        self.message = message
        super().__init__(message)
