"""Persistent JAX compilation cache configuration.

Every panelcast subprocess (pipeline runs, each ``panelcast select`` sweep
arm, preflight mini-runs) is a fresh Python process that would otherwise
recompile its XLA programs from scratch. Enabling JAX's on-disk compilation
cache lets later processes reuse the compiled executables directly.

Environment variables:
    PANELCAST_JAX_CACHE: set to ``"0"`` to disable the cache entirely.
    PANELCAST_JAX_CACHE_DIR: override the cache location. Defaults to
        ``~/.cache/panelcast/jax``.

Cached entries are the bit-identical compiled programs XLA would produce
anyway, so reuse has no effect on numerical outputs. JAX has no built-in
eviction, so the directory grows over time; it is safe to delete at any
point. Cache keys include the jaxlib/XLA versions and compile flags, so
entries written by a different toolchain are never reused.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

log = structlog.get_logger()


def enable_jax_compilation_cache() -> Path | None:
    """Enable JAX's on-disk compilation cache for this process.

    Idempotent; safe to call more than once.

    Returns:
        The cache directory in use, or None when disabled via
        ``PANELCAST_JAX_CACHE=0``.
    """
    if os.environ.get("PANELCAST_JAX_CACHE") == "0":
        return None

    env_dir = os.environ.get("PANELCAST_JAX_CACHE_DIR")
    cache_dir = Path(env_dir) if env_dir else Path.home() / ".cache" / "panelcast" / "jax"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Imported here so the kill switch never touches jax.
    import jax

    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    log.debug("jax_compilation_cache_enabled", cache_dir=str(cache_dir))
    return cache_dir
