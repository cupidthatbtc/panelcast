"""Logging configuration for pipeline execution.

This module provides utilities to configure structlog for pipeline runs
with console output at INFO level (human-readable) and optional file output
at DEBUG level (JSON for structured analysis).
"""

import logging
import sys
from pathlib import Path

import structlog


def setup_pipeline_logging(
    verbose: bool = False,
    log_file: Path | str | None = None,
) -> None:
    """Configure structlog for pipeline execution.

    Sets up structured logging with two output handlers:
    1. Console: Human-readable colored output at INFO level (DEBUG if verbose)
    2. File (optional): JSON-formatted DEBUG output for analysis

    Args:
        verbose: If True, console shows DEBUG level. If False (default),
            console shows INFO level.
        log_file: Optional path to JSON log file. If provided, all DEBUG
            messages are written in JSON format. If None (default), no
            file logging.

    Example:
        >>> # Basic setup: INFO to console
        >>> setup_pipeline_logging()
        >>>
        >>> # Verbose mode: DEBUG to console
        >>> setup_pipeline_logging(verbose=True)
        >>>
        >>> # With file logging
        >>> setup_pipeline_logging(log_file="outputs/run/pipeline.log.json")

    Note:
        This function reconfigures the root logger and structlog globally.
        Call it once at pipeline entry point, before any logging calls.
    """
    # Clear existing handlers from root logger
    root = logging.getLogger()
    root.handlers.clear()

    # Set root to DEBUG to capture all (handlers filter by level)
    root.setLevel(logging.DEBUG)

    # Configure structlog processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    structlog.configure(
        processors=shared_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Console handler: human-readable output
    console_level = logging.DEBUG if verbose else logging.INFO
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
        )
    )
    root.addHandler(console_handler)

    # Suppress JAX/jaxlib internal DEBUG spam (cache-key hashing, dispatch tracing).
    # These use stdlib logging and inherit from root (which is set to DEBUG).
    # Even when console is INFO-only, the root DEBUG level causes JAX to format
    # expensive debug messages. Set them to WARNING unconditionally.
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("jaxlib").setLevel(logging.WARNING)

    # File handler (optional): JSON output
    if log_file is not None:
        log_file = Path(log_file)
        # Ensure parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always DEBUG for file
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ],
            )
        )
        root.addHandler(file_handler)


def is_interactive() -> bool:
    """Check if running in an interactive terminal.

    Returns True if stdout is connected to a TTY (interactive terminal),
    False if output is being piped or running in CI environments.

    This is useful for:
    - Disabling progress bars in CI
    - Adjusting log formatting for non-interactive environments
    - Detecting when user input is not available

    Returns:
        True if stdout is a TTY, False otherwise.

    Example:
        >>> if is_interactive():
        ...     # Show progress bars
        ... else:
        ...     # Use simpler output for CI
    """
    return sys.stdout.isatty()


# Deprecated: kept for backwards compatibility
def setup_logging() -> None:
    """Deprecated: Use setup_pipeline_logging instead.

    Kept for backwards compatibility. New code should use
    setup_pipeline_logging(verbose=False) for equivalent behavior.
    """
    import warnings

    warnings.warn(
        "setup_logging() is deprecated, use setup_pipeline_logging() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    setup_pipeline_logging(verbose=False)
