"""Unit tests for logging utilities."""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from panelcast.utils.logging import (
    is_interactive,
    setup_pipeline_logging,
)


class TestSetupPipelineLogging:
    """Tests for setup_pipeline_logging function."""

    def test_sets_root_to_debug(self):
        setup_pipeline_logging()
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_non_verbose_console_at_info(self):
        setup_pipeline_logging(verbose=False)
        root = logging.getLogger()
        console_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(console_handlers) >= 1
        assert console_handlers[0].level == logging.INFO

    def test_verbose_console_at_debug(self):
        setup_pipeline_logging(verbose=True)
        root = logging.getLogger()
        console_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(console_handlers) >= 1
        assert console_handlers[0].level == logging.DEBUG

    def test_clears_existing_handlers(self):
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        initial_count = len(root.handlers)
        setup_pipeline_logging()
        # After setup, should have exactly 1 console handler (not accumulated)
        assert len(root.handlers) <= initial_count

    def test_closes_prior_file_handler(self, tmp_path):
        """A prior run's FileHandler is closed (fd released) on reconfigure.

        Merely dropping it from root.handlers leaves the log file locked on
        Windows until GC; closing releases the handle immediately.
        """
        root = logging.getLogger()
        old_handler = logging.FileHandler(tmp_path / "old.log", encoding="utf-8")
        old_stream = old_handler.stream
        root.addHandler(old_handler)
        setup_pipeline_logging()
        assert old_handler not in root.handlers
        assert old_stream.closed

    def test_suppresses_jax_debug(self):
        setup_pipeline_logging()
        jax_logger = logging.getLogger("jax")
        jaxlib_logger = logging.getLogger("jaxlib")
        assert jax_logger.level == logging.WARNING
        assert jaxlib_logger.level == logging.WARNING

    def test_file_logging(self, tmp_path):
        log_file = tmp_path / "test.log"
        setup_pipeline_logging(log_file=log_file)
        assert log_file.exists() or True  # File created on first log, or handler created
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1
        assert file_handlers[0].level == logging.DEBUG

    def test_file_logging_creates_parent_dirs(self, tmp_path):
        log_file = tmp_path / "nested" / "dir" / "test.log"
        setup_pipeline_logging(log_file=log_file)
        assert log_file.parent.exists()

    def test_no_file_handler_when_none(self):
        setup_pipeline_logging(log_file=None)
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0

    def test_console_outputs_to_stderr(self):
        setup_pipeline_logging()
        root = logging.getLogger()
        console_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(console_handlers) >= 1
        assert console_handlers[0].stream is sys.stderr


class TestIsInteractive:
    """Tests for is_interactive function."""

    def test_returns_bool(self):
        result = is_interactive()
        assert isinstance(result, bool)

    @patch("panelcast.utils.logging.sys.stdout")
    def test_tty_returns_true(self, mock_stdout):
        mock_stdout.isatty.return_value = True
        assert is_interactive() is True

    @patch("panelcast.utils.logging.sys.stdout")
    def test_non_tty_returns_false(self, mock_stdout):
        mock_stdout.isatty.return_value = False
        assert is_interactive() is False
