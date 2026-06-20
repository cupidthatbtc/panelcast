"""Tests for pipeline error classes."""

from __future__ import annotations

import pytest

from panelcast.pipelines.errors import (
    ConvergenceError,
    DataValidationError,
    EnvironmentError,
    GpuMemoryError,
    PipelineError,
    StageError,
    StageSkipped,
)

# ============================================================================
# PipelineError Tests
# ============================================================================


class TestPipelineError:
    """Tests for PipelineError base class."""

    def test_default_exit_code(self):
        """PipelineError defaults to exit code 1."""
        error = PipelineError("something broke")
        assert error.exit_code == 1

    def test_message_stored(self):
        """PipelineError stores message."""
        error = PipelineError("test message")
        assert error.message == "test message"

    def test_stage_defaults_to_empty(self):
        """PipelineError stage defaults to empty string."""
        error = PipelineError("msg")
        assert error.stage == ""

    def test_str_with_stage(self):
        """String representation includes stage prefix."""
        error = PipelineError("broken", stage="train")
        assert str(error) == "[train] broken"

    def test_str_without_stage(self):
        """String representation without stage has no prefix."""
        error = PipelineError("broken")
        assert str(error) == "broken"

    def test_custom_exit_code(self):
        """PipelineError accepts custom exit code."""
        error = PipelineError("err", exit_code=99)
        assert error.exit_code == 99

    def test_inherits_from_exception(self):
        """PipelineError inherits from Exception."""
        error = PipelineError("test")
        assert isinstance(error, Exception)

    def test_can_be_caught_as_exception(self):
        """PipelineError can be caught as generic Exception."""
        with pytest.raises(Exception):
            raise PipelineError("test")

    def test_args_contain_string_representation(self):
        """Exception args contain formatted string."""
        error = PipelineError("msg", stage="data")
        assert "[data] msg" in str(error.args[0])


# ============================================================================
# ConvergenceError Tests
# ============================================================================


class TestConvergenceError:
    """Tests for ConvergenceError exception."""

    def test_exit_code_is_2(self):
        """ConvergenceError has exit code 2."""
        error = ConvergenceError("R-hat exceeded")
        assert error.exit_code == 2

    def test_inherits_from_pipeline_error(self):
        """ConvergenceError inherits from PipelineError."""
        error = ConvergenceError("test")
        assert isinstance(error, PipelineError)

    def test_default_stage_empty(self):
        """ConvergenceError defaults to empty stage."""
        error = ConvergenceError("test")
        assert error.stage == ""

    def test_custom_stage(self):
        """ConvergenceError accepts custom stage."""
        error = ConvergenceError("R-hat too high", stage="train")
        assert error.stage == "train"
        assert "[train]" in str(error)

    def test_message_preserved(self):
        """ConvergenceError preserves message."""
        error = ConvergenceError("R-hat=1.05")
        assert error.message == "R-hat=1.05"

    def test_can_be_caught_as_pipeline_error(self):
        """ConvergenceError caught by PipelineError handler."""
        with pytest.raises(PipelineError):
            raise ConvergenceError("convergence issue")


# ============================================================================
# DataValidationError Tests
# ============================================================================


class TestDataValidationError:
    """Tests for DataValidationError exception."""

    def test_exit_code_is_3(self):
        """DataValidationError has exit code 3."""
        error = DataValidationError("Missing columns")
        assert error.exit_code == 3

    def test_inherits_from_pipeline_error(self):
        """DataValidationError inherits from PipelineError."""
        error = DataValidationError("test")
        assert isinstance(error, PipelineError)

    def test_default_stage_empty(self):
        """DataValidationError defaults to empty stage."""
        error = DataValidationError("bad data")
        assert error.stage == ""

    def test_custom_stage(self):
        """DataValidationError accepts custom stage."""
        error = DataValidationError("schema mismatch", stage="data")
        assert error.stage == "data"

    def test_str_format(self):
        """String representation includes stage prefix."""
        error = DataValidationError("column X missing", stage="splits")
        assert str(error) == "[splits] column X missing"


# ============================================================================
# StageError Tests
# ============================================================================


class TestStageError:
    """Tests for StageError exception."""

    def test_exit_code_is_4(self):
        """StageError has exit code 4."""
        error = StageError("IO failure")
        assert error.exit_code == 4

    def test_inherits_from_pipeline_error(self):
        """StageError inherits from PipelineError."""
        error = StageError("test")
        assert isinstance(error, PipelineError)

    def test_custom_stage(self):
        """StageError accepts custom stage."""
        error = StageError("failed", stage="features")
        assert error.stage == "features"


# ============================================================================
# EnvironmentError Tests
# ============================================================================


class TestEnvironmentError:
    """Tests for EnvironmentError exception."""

    def test_exit_code_is_5(self):
        """EnvironmentError has exit code 5."""
        error = EnvironmentError("pixi.lock not found")
        assert error.exit_code == 5

    def test_inherits_from_pipeline_error(self):
        """EnvironmentError inherits from PipelineError."""
        error = EnvironmentError("test")
        assert isinstance(error, PipelineError)

    def test_default_stage_empty(self):
        """EnvironmentError defaults to empty stage."""
        error = EnvironmentError("env issue")
        assert error.stage == ""

    def test_custom_stage(self):
        """EnvironmentError accepts custom stage."""
        error = EnvironmentError("test", stage="setup")
        assert error.stage == "setup"


# ============================================================================
# GpuMemoryError Tests
# ============================================================================


class TestGpuMemoryError:
    """Tests for GpuMemoryError exception."""

    def test_exit_code_is_6(self):
        """GpuMemoryError has exit code 6."""
        error = GpuMemoryError("GPU check failed")
        assert error.exit_code == 6

    def test_inherits_from_pipeline_error(self):
        """GpuMemoryError inherits from PipelineError."""
        error = GpuMemoryError("test")
        assert isinstance(error, PipelineError)

    def test_default_stage_is_gpu_check(self):
        """GpuMemoryError defaults to stage='gpu_check'."""
        error = GpuMemoryError("test message")
        assert error.stage == "gpu_check"
        assert "[gpu_check]" in str(error)

    def test_custom_stage(self):
        """GpuMemoryError accepts custom stage."""
        error = GpuMemoryError("test", stage="preflight")
        assert error.stage == "preflight"

    def test_message_preserved(self):
        """GpuMemoryError preserves message."""
        error = GpuMemoryError("Insufficient GPU memory")
        assert error.message == "Insufficient GPU memory"

    def test_can_be_raised_and_caught(self):
        """GpuMemoryError can be raised and caught."""
        with pytest.raises(GpuMemoryError) as exc_info:
            raise GpuMemoryError("No GPU detected")
        assert exc_info.value.exit_code == 6


# ============================================================================
# StageSkipped Tests
# ============================================================================


class TestStageSkipped:
    """Tests for StageSkipped control flow exception."""

    def test_not_pipeline_error(self):
        """StageSkipped does not inherit from PipelineError."""
        error = StageSkipped("inputs unchanged")
        assert not isinstance(error, PipelineError)

    def test_inherits_from_exception(self):
        """StageSkipped inherits from Exception."""
        error = StageSkipped("test")
        assert isinstance(error, Exception)

    def test_message_stored(self):
        """StageSkipped stores message."""
        error = StageSkipped("no changes detected")
        assert error.message == "no changes detected"

    def test_str_representation(self):
        """StageSkipped str is the message."""
        error = StageSkipped("skipping stage")
        assert str(error) == "skipping stage"

    def test_can_be_caught_separately(self):
        """StageSkipped is not caught by PipelineError handler."""
        with pytest.raises(StageSkipped):
            raise StageSkipped("test")

        # And should NOT be caught by PipelineError
        try:
            raise StageSkipped("test")
        except PipelineError:
            pytest.fail("StageSkipped should not be caught as PipelineError")
        except StageSkipped:
            pass


# ============================================================================
# Error Hierarchy Tests
# ============================================================================


class TestErrorHierarchy:
    """Tests for the error class hierarchy."""

    @pytest.mark.parametrize(
        "error_cls,expected_exit_code",
        [
            (PipelineError, 1),
            (ConvergenceError, 2),
            (DataValidationError, 3),
            (StageError, 4),
            (EnvironmentError, 5),
            (GpuMemoryError, 6),
        ],
    )
    def test_exit_codes_are_unique_and_correct(self, error_cls, expected_exit_code):
        """Each error class has the correct unique exit code."""
        if error_cls == PipelineError:
            error = error_cls("test")
        elif error_cls == GpuMemoryError:
            error = error_cls("test")
        else:
            error = error_cls("test")
        assert error.exit_code == expected_exit_code

    @pytest.mark.parametrize(
        "error_cls",
        [
            ConvergenceError,
            DataValidationError,
            StageError,
            EnvironmentError,
            GpuMemoryError,
        ],
    )
    def test_all_subclasses_inherit_from_pipeline_error(self, error_cls):
        """All error subclasses inherit from PipelineError."""
        assert issubclass(error_cls, PipelineError)

    @pytest.mark.parametrize(
        "error_cls",
        [
            ConvergenceError,
            DataValidationError,
            StageError,
            EnvironmentError,
            GpuMemoryError,
        ],
    )
    def test_all_catchable_as_pipeline_error(self, error_cls):
        """All error subclasses can be caught as PipelineError."""
        with pytest.raises(PipelineError):
            if error_cls == GpuMemoryError:
                raise error_cls("test")
            else:
                raise error_cls("test")
