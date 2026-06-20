"""Tests for mini-MCMC run module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture
def minimal_model_args():
    """Minimal valid model arguments for testing."""
    return {
        "artist_idx": [0],
        "album_seq": [1],
        "prev_score": [0.0],
        "X": [[1.0]],
        "y": [70.0],
        "n_artists": 1,
        "max_seq": 1,
    }


class TestMiniRunJsonOutputFormat:
    """Tests for run_and_measure JSON output format."""

    @mock.patch("panelcast.gpu_memory.measure.get_jax_memory_stats")
    @mock.patch("numpyro.infer.MCMC")
    @mock.patch("numpyro.infer.NUTS")
    @mock.patch("panelcast.models.bayes.model.make_score_model")
    def test_run_and_measure_returns_expected_keys(
        self,
        mock_make_model,
        mock_nuts,
        mock_mcmc_class,
        mock_get_stats,
    ):
        """run_and_measure returns dict with success, peak_memory_bytes, runtime_seconds."""
        from panelcast.preflight.mini_run import run_and_measure

        # Set up mocks
        mock_make_model.return_value = "model"
        mock_nuts.return_value = "nuts_kernel"

        mock_mcmc = mock.Mock()
        mock_mcmc_class.return_value = mock_mcmc

        mock_stats = mock.Mock()
        mock_stats.peak_bytes_in_use = 4 * 1024**3
        mock_stats.peak_gb = 4.0
        mock_get_stats.return_value = mock_stats

        # Create temp JSON file with model args
        model_args = {
            "artist_idx": [0, 1, 0],
            "album_seq": [1, 1, 2],
            "prev_score": [0.0, 0.0, 75.0],
            "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "y": [70.0, 80.0, 75.0],
            "n_artists": 2,
            "max_seq": 2,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_args, f)
            temp_path = Path(f.name)

        try:
            result = run_and_measure(temp_path)

            # Verify returned dict has expected keys
            assert "success" in result
            assert "peak_memory_bytes" in result
            assert "runtime_seconds" in result

            # Verify values
            assert result["success"] is True
            assert result["peak_memory_bytes"] == 4 * 1024**3
            assert isinstance(result["runtime_seconds"], float)
            assert result["runtime_seconds"] >= 0

        finally:
            temp_path.unlink()

    @mock.patch("panelcast.gpu_memory.measure.get_jax_memory_stats")
    @mock.patch("numpyro.infer.MCMC")
    @mock.patch("numpyro.infer.NUTS")
    @mock.patch("panelcast.models.bayes.model.make_score_model")
    def test_run_and_measure_handles_optional_params(
        self,
        mock_make_model,
        mock_nuts,
        mock_mcmc_class,
        mock_get_stats,
    ):
        """run_and_measure handles optional heteroscedastic parameters."""
        from panelcast.preflight.mini_run import run_and_measure

        # Set up mocks
        mock_make_model.return_value = "model"
        mock_nuts.return_value = "nuts_kernel"

        mock_mcmc = mock.Mock()
        mock_mcmc_class.return_value = mock_mcmc

        mock_stats = mock.Mock()
        mock_stats.peak_bytes_in_use = 2 * 1024**3
        mock_stats.peak_gb = 2.0
        mock_get_stats.return_value = mock_stats

        # Create temp JSON file with model args including optional params
        model_args = {
            "artist_idx": [0, 1],
            "album_seq": [1, 1],
            "prev_score": [0.0, 0.0],
            "X": [[1.0], [2.0]],
            "y": [70.0, 80.0],
            "n_artists": 2,
            "max_seq": 1,
            # Optional heteroscedastic params
            "n_reviews": [10.0, 20.0],
            "n_exponent": 0.33,
            "learn_n_exponent": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_args, f)
            temp_path = Path(f.name)

        try:
            result = run_and_measure(temp_path)

            assert result["success"] is True

            # Verify MCMC.run was called with the optional params
            call_kwargs = mock_mcmc.run.call_args[1]
            assert "n_reviews" in call_kwargs
            assert call_kwargs["n_exponent"] == 0.33
            assert call_kwargs["learn_n_exponent"] is False

        finally:
            temp_path.unlink()


class TestMiniRunExceptionHandling:
    """Tests for error handling in mini-run."""

    @mock.patch("numpyro.infer.MCMC")
    @mock.patch("numpyro.infer.NUTS")
    @mock.patch("panelcast.models.bayes.model.make_score_model")
    def test_run_and_measure_mcmc_exception_propagates(
        self,
        mock_make_model,
        mock_nuts,
        mock_mcmc_class,
        minimal_model_args,
    ):
        """Exception from MCMC.run propagates (caught by main block)."""
        from panelcast.preflight.mini_run import run_and_measure

        # Set up mocks
        mock_make_model.return_value = "model"
        mock_nuts.return_value = "nuts_kernel"

        mock_mcmc = mock.Mock()
        mock_mcmc.run.side_effect = RuntimeError("CUDA out of memory")
        mock_mcmc_class.return_value = mock_mcmc

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(minimal_model_args, f)
            temp_path = Path(f.name)

        try:
            # run_and_measure propagates the exception (main block catches it)
            with pytest.raises(RuntimeError) as exc_info:
                run_and_measure(temp_path)

            assert "CUDA out of memory" in str(exc_info.value)

        finally:
            temp_path.unlink()

    def test_mini_run_invalid_json_file(self):
        """run_and_measure raises on invalid JSON file."""
        from panelcast.preflight.mini_run import run_and_measure

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            temp_path = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                run_and_measure(temp_path)
        finally:
            temp_path.unlink()

    def test_mini_run_missing_file(self):
        """run_and_measure raises FileNotFoundError for missing file."""
        from panelcast.preflight.mini_run import run_and_measure

        with pytest.raises(FileNotFoundError):
            run_and_measure(Path("/nonexistent/path/model_args.json"))


class TestMiniRunMCMCConfiguration:
    """Tests for MCMC configuration in mini-run."""

    @mock.patch("panelcast.gpu_memory.measure.get_jax_memory_stats")
    @mock.patch("numpyro.infer.MCMC")
    @mock.patch("numpyro.infer.NUTS")
    @mock.patch("panelcast.models.bayes.model.make_score_model")
    def test_mcmc_configured_for_mini_run(
        self,
        mock_make_model,
        mock_nuts,
        mock_mcmc_class,
        mock_get_stats,
        minimal_model_args,
    ):
        """MCMC is configured with 1 chain, 10 warmup, 1 sample."""
        from panelcast.preflight.mini_run import run_and_measure

        # Set up mocks
        mock_make_model.return_value = "model"
        mock_nuts.return_value = "nuts_kernel"

        mock_mcmc = mock.Mock()
        mock_mcmc_class.return_value = mock_mcmc

        mock_stats = mock.Mock()
        mock_stats.peak_bytes_in_use = 1 * 1024**3
        mock_stats.peak_gb = 1.0
        mock_get_stats.return_value = mock_stats

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(minimal_model_args, f)
            temp_path = Path(f.name)

        try:
            run_and_measure(temp_path)

            # Verify MCMC was constructed with mini-run params
            call_kwargs = mock_mcmc_class.call_args[1]
            assert call_kwargs["num_warmup"] == 10
            assert call_kwargs["num_samples"] == 1
            assert call_kwargs["num_chains"] == 1
            assert call_kwargs["progress_bar"] is False

        finally:
            temp_path.unlink()
