"""Environment verification utilities for reproducibility.

This module provides utilities to verify the environment is properly locked
for reproducible execution, including pixi.lock detection and hashing.
"""

from dataclasses import dataclass, field
from pathlib import Path

import structlog

from panelcast.utils.hashing import sha256_file

log = structlog.get_logger()


class EnvironmentError(Exception):
    """Raised when environment is not configured for reproducibility.

    This exception is raised by ensure_environment_locked in strict mode
    when pixi.lock is not found.
    """

    pass


@dataclass
class EnvironmentStatus:
    """Status of environment reproducibility verification.

    Attributes:
        pixi_lock_exists: Whether pixi.lock file was found
        pixi_lock_path: Path to pixi.lock file if found, None otherwise
        pixi_lock_hash: SHA256 hash of pixi.lock file if found, None otherwise
        is_reproducible: True only if pixi_lock_exists (environment is locked)
        warnings: List of warning messages about environment issues
    """

    pixi_lock_exists: bool
    pixi_lock_path: Path | None
    pixi_lock_hash: str | None
    is_reproducible: bool
    warnings: list[str] = field(default_factory=list)


def verify_environment(project_root: Path | str = Path(".")) -> EnvironmentStatus:
    """Verify environment is locked for reproducibility.

    Checks for pixi.lock in project_root and parent directories.
    Computes SHA256 hash of lock file for manifest tracking.

    Args:
        project_root: Path to start searching for pixi.lock.
            Defaults to current directory.

    Returns:
        EnvironmentStatus with verification results.

    Example:
        >>> status = verify_environment()
        >>> if not status.is_reproducible:
        ...     print("Warning:", status.warnings)
    """
    project_root = Path(project_root).resolve()
    warnings: list[str] = []

    # Search for pixi.lock in project_root and parent directories
    pixi_lock_path = _find_pixi_lock(project_root)

    if pixi_lock_path is None:
        warnings.append(
            "pixi.lock not found - environment may not be reproducible. "
            "Run 'pixi install' to generate pixi.lock for reproducible environments."
        )
        return EnvironmentStatus(
            pixi_lock_exists=False,
            pixi_lock_path=None,
            pixi_lock_hash=None,
            is_reproducible=False,
            warnings=warnings,
        )

    # Compute hash of pixi.lock
    pixi_lock_hash = sha256_file(pixi_lock_path)

    return EnvironmentStatus(
        pixi_lock_exists=True,
        pixi_lock_path=pixi_lock_path,
        pixi_lock_hash=pixi_lock_hash,
        is_reproducible=True,
        warnings=warnings,
    )


def _find_pixi_lock(start_path: Path) -> Path | None:
    """Search for pixi.lock in start_path and parent directories.

    Args:
        start_path: Directory to start searching from.

    Returns:
        Path to pixi.lock if found, None otherwise.
    """
    current = start_path.resolve()

    # Search up to filesystem root
    while current != current.parent:
        lock_path = current / "pixi.lock"
        if lock_path.exists():
            return lock_path
        current = current.parent

    # Check root as well
    lock_path = current / "pixi.lock"
    if lock_path.exists():
        return lock_path

    return None


def ensure_environment_locked(
    project_root: Path | str = Path("."),
    strict: bool = False,
) -> None:
    """Ensure environment is locked, optionally failing if not.

    Args:
        project_root: Path to start searching for pixi.lock.
        strict: If True, raise EnvironmentError if not locked.
            If False (default), log a warning.

    Raises:
        EnvironmentError: If strict=True and pixi.lock not found.

    Example:
        >>> ensure_environment_locked(strict=False)  # Logs warning if not locked
        >>> ensure_environment_locked(strict=True)   # Raises if not locked
    """
    status = verify_environment(project_root)

    if not status.is_reproducible:
        message = (
            "Environment is not locked for reproducibility. "
            "Run 'pixi install' to generate pixi.lock."
        )

        if strict:
            raise EnvironmentError(message)
        else:
            log.warning("environment_not_locked", message=message, warnings=status.warnings)
