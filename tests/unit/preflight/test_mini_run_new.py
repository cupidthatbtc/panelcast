"""New coverage-targeted tests for preflight/mini_run.py.

Focuses on the __main__ block (lines 220-254) which involves:
- Logging configuration to stderr
- _parse_args ValueError -> JSON error output + sys.exit(1)
- run_and_measure success -> JSON output
- run_and_measure exception -> JSON error output + sys.exit(1)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from panelcast.preflight.mini_run import _parse_args, run_and_measure


def _run_mini(*args: str, timeout: int = 120) -> tuple[int, dict]:
    """Invoke ``python -m panelcast.preflight.mini_run`` and return (rc, json).

    The module's contract is "JSON on stdout, logs/warnings on stderr", but a
    fresh subprocess can still pick up an environment line on stdout (e.g. an
    editable-install or platform notice in CI). Parse the last JSON object on
    stdout so the test asserts on the module's payload rather than on stdout
    being byte-pure, and surface stdout+stderr on failure for diagnosability.
    """
    result = subprocess.run(
        [sys.executable, "-m", "panelcast.preflight.mini_run", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    payload = None
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if payload is None:
        raise AssertionError(
            "mini_run produced no JSON object on stdout "
            f"(returncode={result.returncode}).\n"
            f"--- stdout ---\n{result.stdout!r}\n"
            f"--- stderr (tail) ---\n{result.stderr[-3000:]!r}"
        )
    return result.returncode, payload


class TestMainBlockViaSubprocess:
    """Test the __main__ block by actually invoking it as a subprocess.

    This covers the real code path: logging config, _parse_args,
    run_and_measure, and error handling - all within the if __name__ == '__main__'
    block.
    """

    def test_no_args_outputs_error_json(self):
        """Invoking with no args should produce JSON error with usage message."""
        rc, output = _run_mini()
        assert rc == 1
        assert output["success"] is False
        assert output["exit_code"] == 1
        assert "Usage" in output["error"]
        assert output["peak_memory_bytes"] == 0
        assert output["runtime_seconds"] == 0.0

    def test_unknown_arg_outputs_error_json(self):
        """Invoking with unknown arg should produce JSON error."""
        rc, output = _run_mini("dummy.json", "--bad-flag")
        assert rc == 1
        assert output["success"] is False
        assert "Unknown argument" in output["error"]

    def test_missing_warmup_value_outputs_error_json(self):
        """--num-warmup without value produces JSON error."""
        rc, output = _run_mini("dummy.json", "--num-warmup")
        assert rc == 1
        assert output["success"] is False
        assert "--num-warmup requires" in output["error"]

    def test_missing_samples_value_outputs_error_json(self):
        """--num-samples without value produces JSON error."""
        rc, output = _run_mini("dummy.json", "--num-samples")
        assert rc == 1
        assert output["success"] is False
        assert "--num-samples requires" in output["error"]

    def test_zero_warmup_outputs_error_json(self):
        """--num-warmup 0 produces JSON error about positive value."""
        rc, output = _run_mini("dummy.json", "--num-warmup", "0")
        assert rc == 1
        assert output["success"] is False
        assert "positive" in output["error"]

    def test_nonexistent_file_outputs_error_json(self):
        """Passing a nonexistent JSON file should produce JSON error from exception handler."""
        rc, output = _run_mini("/nonexistent/model_args.json")
        assert rc == 1
        assert output["success"] is False
        assert output["exit_code"] == 1
        assert output["peak_memory_bytes"] == 0

    def test_invalid_json_file_outputs_error_json(self):
        """Passing an invalid JSON file triggers the run_and_measure exception path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            temp_path = f.name

        try:
            rc, output = _run_mini(temp_path)
            assert rc == 1
            assert output["success"] is False
            assert output["exit_code"] == 1
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_missing_keys_file_outputs_error_json(self):
        """JSON file with missing required keys triggers run_and_measure ValueError path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"artist_idx": [0]}, f)
            temp_path = f.name

        try:
            rc, output = _run_mini(temp_path)
            assert rc == 1
            assert output["success"] is False
            assert "Missing required keys" in output["error"]
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestParseArgsEdgeCases:
    """Additional edge cases for _parse_args not covered by existing tests."""

    def test_non_integer_warmup_float_str(self):
        """Non-integer warmup with float string raises ValueError."""
        with pytest.raises(ValueError, match="--num-warmup requires an integer"):
            _parse_args(["args.json", "--num-warmup", "1.5"])

    def test_non_integer_samples_text(self):
        """Non-integer samples with text raises ValueError."""
        with pytest.raises(ValueError, match="--num-samples requires an integer"):
            _parse_args(["args.json", "--num-samples", "many"])

    def test_both_args_with_path(self):
        """Both warmup and samples along with path return correct tuple."""
        path, warmup, samples, _, _, _ = _parse_args(
            ["/tmp/model.json", "--num-warmup", "5", "--num-samples", "10"]
        )
        assert path == Path("/tmp/model.json")
        assert warmup == 5
        assert samples == 10

    def test_samples_before_warmup(self):
        """--num-samples can appear before --num-warmup."""
        _, warmup, samples, _, _, _ = _parse_args(
            ["a.json", "--num-samples", "20", "--num-warmup", "15"]
        )
        assert warmup == 15
        assert samples == 20


class TestRunAndMeasureEdgeCases:
    """Additional edge cases for run_and_measure."""

    @mock.patch("panelcast.gpu_memory.measure.get_jax_memory_stats")
    @mock.patch("numpyro.infer.MCMC")
    @mock.patch("numpyro.infer.NUTS")
    @mock.patch("panelcast.models.bayes.model.make_score_model")
    def test_custom_warmup_samples_passed(
        self,
        mock_make_model,
        mock_nuts,
        mock_mcmc_class,
        mock_get_stats,
    ):
        """Custom num_warmup and num_samples reach the MCMC constructor."""
        mock_make_model.return_value = "model"
        mock_nuts.return_value = "kernel"
        mock_mcmc = mock.Mock()
        mock_mcmc_class.return_value = mock_mcmc

        mock_stats = mock.Mock()
        mock_stats.peak_bytes_in_use = 500
        mock_stats.peak_gb = 0.0
        mock_get_stats.return_value = mock_stats

        model_args = {
            "artist_idx": [0],
            "album_seq": [1],
            "prev_score": [0.0],
            "X": [[1.0]],
            "y": [70.0],
            "n_artists": 1,
            "max_seq": 1,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_args, f)
            temp_path = Path(f.name)

        try:
            result = run_and_measure(temp_path, num_warmup=50, num_samples=100)
            call_kwargs = mock_mcmc_class.call_args[1]
            assert call_kwargs["num_warmup"] == 50
            assert call_kwargs["num_samples"] == 100
            assert result["success"] is True
            assert result["peak_memory_bytes"] == 500
        finally:
            temp_path.unlink()

    def test_empty_json_file_raises(self):
        """Empty JSON object (no keys) raises ValueError for missing keys."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Missing required keys"):
                run_and_measure(temp_path)
        finally:
            Path(temp_path).unlink()

    @mock.patch("panelcast.gpu_memory.measure.get_jax_memory_stats")
    @mock.patch("numpyro.infer.MCMC")
    @mock.patch("numpyro.infer.NUTS")
    @mock.patch("panelcast.models.bayes.model.make_score_model")
    def test_only_n_reviews_optional_param(
        self,
        mock_make_model,
        mock_nuts,
        mock_mcmc_class,
        mock_get_stats,
    ):
        """Only n_reviews optional param set (not n_exponent or learn_n_exponent)."""
        mock_make_model.return_value = "model"
        mock_nuts.return_value = "kernel"
        mock_mcmc = mock.Mock()
        mock_mcmc_class.return_value = mock_mcmc

        mock_stats = mock.Mock()
        mock_stats.peak_bytes_in_use = 1024
        mock_stats.peak_gb = 0.001
        mock_get_stats.return_value = mock_stats

        model_args = {
            "artist_idx": [0],
            "album_seq": [1],
            "prev_score": [0.0],
            "X": [[1.0]],
            "y": [70.0],
            "n_artists": 1,
            "max_seq": 1,
            "n_reviews": [5.0],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_args, f)
            temp_path = Path(f.name)

        try:
            result = run_and_measure(temp_path)
            call_kwargs = mock_mcmc.run.call_args[1]
            assert "n_reviews" in call_kwargs
            assert "n_exponent" not in call_kwargs
            assert "learn_n_exponent" not in call_kwargs
        finally:
            Path(temp_path).unlink()
