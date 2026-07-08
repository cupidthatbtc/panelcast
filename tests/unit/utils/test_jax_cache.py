"""Tests for persistent JAX compilation cache setup.

jax.config is process-wide and shared across tests, so each test sets its
own environment and re-enables the cache rather than asserting pristine
global state.
"""

from pathlib import Path

import jax

from panelcast.utils.jax_cache import enable_jax_compilation_cache


class TestEnableJaxCompilationCache:
    """Tests for enable_jax_compilation_cache."""

    def test_default_enables_and_returns_default_dir(self, monkeypatch, tmp_path):
        """Without env overrides the cache lands under ~/.cache/panelcast/jax."""
        monkeypatch.delenv("PANELCAST_JAX_CACHE", raising=False)
        monkeypatch.delenv("PANELCAST_JAX_CACHE_DIR", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = enable_jax_compilation_cache()

        assert result == tmp_path / ".cache" / "panelcast" / "jax"
        assert result.is_dir()
        assert jax.config.jax_compilation_cache_dir == str(result)

    def test_kill_switch_returns_none_and_leaves_config_untouched(self, monkeypatch):
        """PANELCAST_JAX_CACHE=0 is a no-op that never touches jax.config."""
        monkeypatch.setenv("PANELCAST_JAX_CACHE", "0")
        before = jax.config.jax_compilation_cache_dir

        assert enable_jax_compilation_cache() is None
        assert jax.config.jax_compilation_cache_dir == before

    def test_dir_override_respected(self, monkeypatch, tmp_path):
        """PANELCAST_JAX_CACHE_DIR wins over the default location."""
        monkeypatch.delenv("PANELCAST_JAX_CACHE", raising=False)
        target = tmp_path / "custom" / "jax-cache"
        monkeypatch.setenv("PANELCAST_JAX_CACHE_DIR", str(target))

        result = enable_jax_compilation_cache()

        assert result == target
        assert target.is_dir()
        assert jax.config.jax_compilation_cache_dir == str(target)

    def test_config_reflects_cache_settings(self, monkeypatch, tmp_path):
        """Enabling sets the dir and persistence thresholds on jax.config."""
        monkeypatch.delenv("PANELCAST_JAX_CACHE", raising=False)
        monkeypatch.setenv("PANELCAST_JAX_CACHE_DIR", str(tmp_path / "jc"))

        result = enable_jax_compilation_cache()

        assert jax.config.jax_compilation_cache_dir == str(result)
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 1.0
        assert jax.config.jax_persistent_cache_min_entry_size_bytes == 0

    def test_idempotent(self, monkeypatch, tmp_path):
        """Calling twice returns the same dir and leaves config consistent."""
        monkeypatch.delenv("PANELCAST_JAX_CACHE", raising=False)
        monkeypatch.setenv("PANELCAST_JAX_CACHE_DIR", str(tmp_path / "jc"))

        first = enable_jax_compilation_cache()
        second = enable_jax_compilation_cache()

        assert first == second
        assert jax.config.jax_compilation_cache_dir == str(first)
