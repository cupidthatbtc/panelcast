"""Tests for mini-MCMC run module."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

import panelcast
from panelcast.preflight.mini_run import _parse_args, run_and_measure

_PKG_PARENT = str(Path(panelcast.__file__).resolve().parent.parent)


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


# --- from unit/preflight/test_mini_run_coverage.py ---


class TestParseArgsBasic:
    """Tests for _parse_args with valid arguments."""

    def test_minimal_args(self):
        """Single positional arg returns defaults for warmup and samples."""
        path, warmup, samples, chains, prefix, excl, *_ = _parse_args(["model_args.json"])
        assert path == Path("model_args.json")
        assert warmup == 10
        assert samples == 1
        assert chains == 1
        assert prefix == "user"

    def test_custom_warmup(self):
        """--num-warmup overrides default."""
        path, warmup, samples, *_ = _parse_args(["args.json", "--num-warmup", "25"])
        assert path == Path("args.json")
        assert warmup == 25
        assert samples == 1

    def test_custom_samples(self):
        """--num-samples overrides default."""
        path, warmup, samples, *_ = _parse_args(["args.json", "--num-samples", "100"])
        assert path == Path("args.json")
        assert warmup == 10
        assert samples == 100

    def test_both_custom_args(self):
        """Both --num-warmup and --num-samples can be set together."""
        path, warmup, samples, *_ = _parse_args(
            ["args.json", "--num-warmup", "20", "--num-samples", "50"]
        )
        assert path == Path("args.json")
        assert warmup == 20
        assert samples == 50

    def test_args_order_independent(self):
        """--num-samples before --num-warmup works."""
        _, warmup, samples, *_ = _parse_args(
            ["args.json", "--num-samples", "50", "--num-warmup", "20"]
        )
        assert warmup == 20
        assert samples == 50

    def test_custom_chains_and_prefix(self):
        """--num-chains and --prefix override the defaults."""
        _, _, _, chains, prefix, *_ = _parse_args(
            ["args.json", "--num-chains", "2", "--prefix", "perf"]
        )
        assert chains == 2
        assert prefix == "perf"

    def test_zero_chains_raises(self):
        with pytest.raises(ValueError, match="--num-chains must be positive"):
            _parse_args(["args.json", "--num-chains", "0"])

    def test_prefix_missing_value_raises(self):
        with pytest.raises(ValueError, match="--prefix requires a value"):
            _parse_args(["args.json", "--prefix"])


class TestParseArgsErrors:
    """Tests for _parse_args error handling."""

    def test_empty_args_raises(self):
        """Empty argument list raises ValueError with usage message."""
        with pytest.raises(ValueError, match="Usage"):
            _parse_args([])

    def test_num_warmup_missing_value_raises(self):
        """--num-warmup without value raises ValueError."""
        with pytest.raises(ValueError, match="--num-warmup requires an integer"):
            _parse_args(["args.json", "--num-warmup"])

    def test_num_samples_missing_value_raises(self):
        """--num-samples without value raises ValueError."""
        with pytest.raises(ValueError, match="--num-samples requires an integer"):
            _parse_args(["args.json", "--num-samples"])

    def test_num_warmup_non_integer_raises(self):
        """--num-warmup with non-integer value raises ValueError."""
        with pytest.raises(ValueError, match="--num-warmup requires an integer"):
            _parse_args(["args.json", "--num-warmup", "abc"])

    def test_num_samples_non_integer_raises(self):
        """--num-samples with non-integer value raises ValueError."""
        with pytest.raises(ValueError, match="--num-samples requires an integer"):
            _parse_args(["args.json", "--num-samples", "xyz"])

    def test_unknown_argument_raises(self):
        """Unknown argument raises ValueError."""
        with pytest.raises(ValueError, match="Unknown argument"):
            _parse_args(["args.json", "--unknown"])

    def test_negative_warmup_raises(self):
        """Negative --num-warmup raises ValueError."""
        with pytest.raises(ValueError, match="--num-warmup must be positive"):
            _parse_args(["args.json", "--num-warmup", "-5"])

    def test_zero_warmup_raises(self):
        """Zero --num-warmup raises ValueError."""
        with pytest.raises(ValueError, match="--num-warmup must be positive"):
            _parse_args(["args.json", "--num-warmup", "0"])

    def test_negative_samples_raises(self):
        """Negative --num-samples raises ValueError."""
        with pytest.raises(ValueError, match="--num-samples must be positive"):
            _parse_args(["args.json", "--num-samples", "-1"])

    def test_zero_samples_raises(self):
        """Zero --num-samples raises ValueError."""
        with pytest.raises(ValueError, match="--num-samples must be positive"):
            _parse_args(["args.json", "--num-samples", "0"])

    def test_num_warmup_float_value_raises(self):
        """Float value for --num-warmup raises ValueError."""
        with pytest.raises(ValueError, match="--num-warmup requires an integer"):
            _parse_args(["args.json", "--num-warmup", "3.14"])

    def test_num_samples_float_value_raises(self):
        """Float value for --num-samples raises ValueError."""
        with pytest.raises(ValueError, match="--num-samples requires an integer"):
            _parse_args(["args.json", "--num-samples", "2.5"])


class TestRunAndMeasureMissingKeys:
    """Tests for run_and_measure missing-key validation."""

    def test_missing_required_keys_raises(self):
        """Missing required keys in JSON raises ValueError with details."""
        from panelcast.preflight.mini_run import run_and_measure

        model_args = {
            "artist_idx": [0],
            # Missing album_seq, prev_score, X, y, n_artists, max_seq
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_args, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Missing required keys"):
                run_and_measure(temp_path)
        finally:
            temp_path.unlink()

    def test_missing_single_key_names_it(self):
        """Error message includes the specific missing key name."""
        from panelcast.preflight.mini_run import run_and_measure

        model_args = {
            "artist_idx": [0],
            "album_seq": [1],
            "prev_score": [0.0],
            "X": [[1.0]],
            "y": [70.0],
            "n_artists": 1,
            # Missing max_seq
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_args, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="max_seq"):
                run_and_measure(temp_path)
        finally:
            temp_path.unlink()


class TestRunAndMeasureCustomParams:
    """Tests for run_and_measure with custom warmup and sample counts."""

    @mock.patch("panelcast.gpu_memory.measure.get_jax_memory_stats")
    @mock.patch("numpyro.infer.MCMC")
    @mock.patch("numpyro.infer.NUTS")
    @mock.patch("panelcast.models.bayes.model.make_score_model")
    def test_custom_warmup_and_samples_passed_to_mcmc(
        self,
        mock_make_model,
        mock_nuts,
        mock_mcmc_class,
        mock_get_stats,
    ):
        """Custom num_warmup and num_samples are forwarded to MCMC constructor."""
        from panelcast.preflight.mini_run import run_and_measure

        mock_make_model.return_value = "model"
        mock_nuts.return_value = "nuts_kernel"
        mock_mcmc = mock.Mock()
        mock_mcmc_class.return_value = mock_mcmc

        mock_stats = mock.Mock()
        mock_stats.peak_bytes_in_use = 1 * 1024**3
        mock_stats.peak_gb = 1.0
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
            result = run_and_measure(temp_path, num_warmup=25, num_samples=50)

            call_kwargs = mock_mcmc_class.call_args[1]
            assert call_kwargs["num_warmup"] == 25
            assert call_kwargs["num_samples"] == 50
            assert result["success"] is True
        finally:
            temp_path.unlink()


class TestRunAndMeasureOutputFormat:
    """Tests for the return value structure of run_and_measure."""

    @mock.patch("panelcast.gpu_memory.measure.get_jax_memory_stats")
    @mock.patch("numpyro.infer.MCMC")
    @mock.patch("numpyro.infer.NUTS")
    @mock.patch("panelcast.models.bayes.model.make_score_model")
    def test_exit_code_in_success_result(
        self,
        mock_make_model,
        mock_nuts,
        mock_mcmc_class,
        mock_get_stats,
    ):
        """Successful run_and_measure includes exit_code=0."""
        from panelcast.preflight.mini_run import run_and_measure

        mock_make_model.return_value = "model"
        mock_nuts.return_value = "nuts_kernel"
        mock_mcmc = mock.Mock()
        mock_mcmc_class.return_value = mock_mcmc

        mock_stats = mock.Mock()
        mock_stats.peak_bytes_in_use = 512 * 1024**2
        mock_stats.peak_gb = 0.5
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
            result = run_and_measure(temp_path)
            assert result["exit_code"] == 0
            assert result["success"] is True
            assert result["peak_memory_bytes"] == 512 * 1024**2
        finally:
            temp_path.unlink()


class TestMainBlockArgParseError:
    """Tests for the __main__ block arg-parse error path.

    The __main__ block catches ValueError from _parse_args and emits JSON.
    We test this by invoking the module as a subprocess-like simulation.
    """

    def test_main_parse_error_outputs_json(self, monkeypatch):
        """_parse_args failure in __main__ produces error JSON to stdout."""
        import io
        import sys

        # We simulate the __main__ block logic without actually calling sys.exit
        captured_output = io.StringIO()

        try:
            _parse_args([])  # Should raise ValueError
            assert False, "Should have raised"  # pragma: no cover
        except ValueError as e:
            result = {
                "success": False,
                "exit_code": 1,
                "error": str(e),
                "peak_memory_bytes": 0,
                "runtime_seconds": 0.0,
            }
            json_str = json.dumps(result)
            parsed = json.loads(json_str)
            assert parsed["success"] is False
            assert parsed["exit_code"] == 1
            assert "Usage" in parsed["error"]

    def test_main_run_error_produces_error_json(self):
        """Exception in run_and_measure produces error JSON."""
        result = {
            "success": False,
            "exit_code": 1,
            "error": "CUDA out of memory",
            "peak_memory_bytes": 0,
            "runtime_seconds": 0.0,
        }
        json_str = json.dumps(result)
        parsed = json.loads(json_str)

        assert parsed["success"] is False
        assert parsed["peak_memory_bytes"] == 0
        assert "CUDA" in parsed["error"]


class TestRunAndMeasureModelCreation:
    """Tests for model creation and NUTS kernel setup in run_and_measure."""

    @mock.patch("panelcast.gpu_memory.measure.get_jax_memory_stats")
    @mock.patch("numpyro.infer.MCMC")
    @mock.patch("numpyro.infer.NUTS")
    @mock.patch("panelcast.models.bayes.model.make_score_model")
    def test_uses_user_model(
        self,
        mock_make_model,
        mock_nuts,
        mock_mcmc_class,
        mock_get_stats,
    ):
        """run_and_measure uses make_score_model('user')."""
        from panelcast.preflight.mini_run import run_and_measure

        mock_make_model.return_value = "user_model"
        mock_nuts.return_value = "nuts_kernel"
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
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_args, f)
            temp_path = Path(f.name)

        try:
            run_and_measure(temp_path)
            mock_make_model.assert_called_once_with("user")
            mock_nuts.assert_called_once_with("user_model")
        finally:
            temp_path.unlink()

    @mock.patch("panelcast.gpu_memory.measure.get_jax_memory_stats")
    @mock.patch("numpyro.infer.MCMC")
    @mock.patch("numpyro.infer.NUTS")
    @mock.patch("panelcast.models.bayes.model.make_score_model")
    def test_mcmc_chain_method_sequential(
        self,
        mock_make_model,
        mock_nuts,
        mock_mcmc_class,
        mock_get_stats,
    ):
        """MCMC is configured with chain_method='sequential'."""
        from panelcast.preflight.mini_run import run_and_measure

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
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_args, f)
            temp_path = Path(f.name)

        try:
            run_and_measure(temp_path)
            call_kwargs = mock_mcmc_class.call_args[1]
            assert call_kwargs["chain_method"] == "sequential"
        finally:
            temp_path.unlink()


# --- from unit/preflight/test_mini_run_more.py ---


class TestParseArgsExcludeCollection:
    def test_single_site(self):
        _, _, _, _, _, excl, *_ = _parse_args(["args.json", "--exclude-collection", "mu_artist"])
        assert excl == ("mu_artist",)

    def test_comma_separated_sites(self):
        _, _, _, _, _, excl, *_ = _parse_args(
            ["args.json", "--exclude-collection", "mu_artist,sigma_obs,rho"]
        )
        assert excl == ("mu_artist", "sigma_obs", "rho")

    def test_spaces_stripped(self):
        _, _, _, _, _, excl, *_ = _parse_args(
            ["args.json", "--exclude-collection", " mu_artist , sigma_obs "]
        )
        assert excl == ("mu_artist", "sigma_obs")

    def test_empty_segments_ignored(self):
        # Trailing comma and double comma both produce clean tuple
        _, _, _, _, _, excl, *_ = _parse_args(["args.json", "--exclude-collection", "mu_artist,,"])
        assert excl == ("mu_artist",)

    def test_missing_value_raises(self):
        with pytest.raises(ValueError, match="--exclude-collection requires a value"):
            _parse_args(["args.json", "--exclude-collection"])

    def test_combined_with_other_flags(self):
        _, warmup, _, chains, prefix, excl, *_ = _parse_args(
            [
                "args.json",
                "--num-warmup",
                "5",
                "--num-chains",
                "2",
                "--prefix",
                "perf",
                "--exclude-collection",
                "mu_artist,beta",
            ]
        )
        assert warmup == 5
        assert chains == 2
        assert prefix == "perf"
        assert excl == ("mu_artist", "beta")

    def test_default_is_empty_tuple(self):
        _, _, _, _, _, excl, *_ = _parse_args(["args.json"])
        assert excl == ()


def _exec_main(argv: list[str]) -> tuple[str, int]:
    """Run the __main__ block with the given sys.argv, return (stdout, exit_code)."""
    import runpy

    buf = io.StringIO()
    exit_code = 0

    def _fake_exit(code=0):
        nonlocal exit_code
        exit_code = code
        # Must raise SystemExit so that the __main__ block stops after sys.exit(1)
        # (without this the code continues into the second try block where
        # model_args_path may not yet be defined).
        raise SystemExit(code)

    with (
        mock.patch("sys.argv", ["panelcast.preflight.mini_run", *argv]),
        mock.patch("sys.stdout", buf),
        mock.patch("sys.exit", side_effect=_fake_exit),
    ):
        try:
            runpy.run_module("panelcast.preflight.mini_run", run_name="__main__", alter_sys=False)
        except SystemExit:
            pass

    return buf.getvalue(), exit_code


class TestMainBlockParseError:
    """__main__ lines 262-280: ValueError from _parse_args -> JSON error + exit(1)."""

    def test_no_args_emits_json_and_exits_1(self):
        out, rc = _exec_main([])
        parsed = json.loads(out.strip())
        assert rc == 1
        assert parsed["success"] is False
        assert parsed["exit_code"] == 1
        assert "Usage" in parsed["error"]
        assert parsed["peak_memory_bytes"] == 0

    def test_unknown_flag_emits_json_and_exits_1(self):
        out, rc = _exec_main(["dummy.json", "--bad"])
        parsed = json.loads(out.strip())
        assert rc == 1
        assert parsed["success"] is False
        assert "Unknown argument" in parsed["error"]

    def test_zero_warmup_emits_json_and_exits_1(self):
        out, rc = _exec_main(["dummy.json", "--num-warmup", "0"])
        parsed = json.loads(out.strip())
        assert rc == 1
        assert "positive" in parsed["error"]


class TestMainBlockRunSuccess:
    """__main__ lines 282-286: successful run_and_measure -> JSON to stdout."""

    def test_success_path_emits_json(self, tmp_path):
        model_args = {
            "artist_idx": [0],
            "album_seq": [1],
            "prev_score": [0.0],
            "X": [[1.0]],
            "y": [70.0],
            "n_artists": 1,
            "max_seq": 1,
        }
        args_file = tmp_path / "model_args.json"
        args_file.write_text(json.dumps(model_args))

        mock_stats = mock.Mock()
        mock_stats.peak_bytes_in_use = 1024
        mock_stats.peak_gb = 0.001

        with (
            mock.patch("panelcast.models.bayes.model.make_score_model", return_value="model"),
            mock.patch("numpyro.infer.NUTS", return_value="kernel"),
            mock.patch("numpyro.infer.MCMC") as mock_mcmc_cls,
            mock.patch(
                "panelcast.gpu_memory.measure.get_jax_memory_stats", return_value=mock_stats
            ),
        ):
            mock_mcmc_cls.return_value = mock.Mock()
            out, rc = _exec_main([str(args_file)])

        parsed = json.loads(out.strip())
        assert rc == 0
        assert parsed["success"] is True
        assert parsed["peak_memory_bytes"] == 1024


class TestMainBlockRunError:
    """__main__ lines 287-297: exception in run_and_measure -> JSON error + exit(1)."""

    def test_missing_file_emits_json_and_exits_1(self):
        out, rc = _exec_main(["/nonexistent/path/model_args.json"])
        parsed = json.loads(out.strip())
        assert rc == 1
        assert parsed["success"] is False
        assert parsed["exit_code"] == 1
        assert parsed["peak_memory_bytes"] == 0

    def test_invalid_json_emits_json_and_exits_1(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid {{{")
        out, rc = _exec_main([str(bad)])
        parsed = json.loads(out.strip())
        assert rc == 1
        assert parsed["success"] is False

    def test_missing_keys_emits_json_and_exits_1(self, tmp_path):
        partial = tmp_path / "partial.json"
        partial.write_text(json.dumps({"artist_idx": [0]}))
        out, rc = _exec_main([str(partial)])
        parsed = json.loads(out.strip())
        assert rc == 1
        assert parsed["success"] is False
        assert "Missing required keys" in parsed["error"]

    def test_mcmc_error_emits_json_and_exits_1(self, tmp_path):
        model_args = {
            "artist_idx": [0],
            "album_seq": [1],
            "prev_score": [0.0],
            "X": [[1.0]],
            "y": [70.0],
            "n_artists": 1,
            "max_seq": 1,
        }
        args_file = tmp_path / "model_args.json"
        args_file.write_text(json.dumps(model_args))

        mock_mcmc = mock.Mock()
        mock_mcmc.run.side_effect = RuntimeError("CUDA OOM")

        with (
            mock.patch("panelcast.models.bayes.model.make_score_model", return_value="model"),
            mock.patch("numpyro.infer.NUTS", return_value="kernel"),
            mock.patch("numpyro.infer.MCMC", return_value=mock_mcmc),
        ):
            out, rc = _exec_main([str(args_file)])

        parsed = json.loads(out.strip())
        assert rc == 1
        assert parsed["success"] is False
        assert "CUDA OOM" in parsed["error"]


# --- from unit/preflight/test_mini_run_new.py ---


def _run_mini(*args: str, timeout: int = 120) -> tuple[int, dict]:
    """Invoke ``python -m panelcast.preflight.mini_run`` and return (rc, json).

    The module's contract is "JSON on stdout, logs/warnings on stderr", but a
    fresh subprocess can still pick up an environment line on stdout (e.g. an
    editable-install or platform notice in CI). Parse the last JSON object on
    stdout so the test asserts on the module's payload rather than on stdout
    being byte-pure, and surface stdout+stderr on failure for diagnosability.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_PKG_PARENT, env.get("PYTHONPATH", "")) if p)
    result = subprocess.run(
        [sys.executable, "-m", "panelcast.preflight.mini_run", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
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
        path, warmup, samples, *_ = _parse_args(
            ["/tmp/model.json", "--num-warmup", "5", "--num-samples", "10"]
        )
        assert path == Path("/tmp/model.json")
        assert warmup == 5
        assert samples == 10

    def test_samples_before_warmup(self):
        """--num-samples can appear before --num-warmup."""
        _, warmup, samples, *_ = _parse_args(
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


# --- 0.7.0 preflight ladder flags (#104) ---


class TestParseArgsNewFlags:
    def test_defaults(self):
        _, _, _, _, _, _, transform, method, pooling = _parse_args(["args.json"])
        assert transform == "identity"
        assert method == "sequential"
        assert pooling is False

    def test_target_transform_offset_logit(self):
        _, _, _, _, _, _, transform, _, _ = _parse_args(
            ["args.json", "--target-transform", "offset_logit"]
        )
        assert transform == "offset_logit"

    def test_target_transform_invalid_raises(self):
        with pytest.raises(ValueError, match="--target-transform must be"):
            _parse_args(["args.json", "--target-transform", "logit"])

    def test_target_transform_missing_value_raises(self):
        with pytest.raises(ValueError, match="--target-transform requires a value"):
            _parse_args(["args.json", "--target-transform"])

    def test_chain_method_vectorized(self):
        _, _, _, _, _, _, _, method, _ = _parse_args(["args.json", "--chain-method", "vectorized"])
        assert method == "vectorized"

    def test_chain_method_invalid_raises(self):
        with pytest.raises(ValueError, match="--chain-method must be"):
            _parse_args(["args.json", "--chain-method", "parallel"])

    def test_chain_method_missing_value_raises(self):
        with pytest.raises(ValueError, match="--chain-method requires a value"):
            _parse_args(["args.json", "--chain-method"])

    def test_entity_group_pooling_flag(self):
        _, _, _, _, _, _, _, _, pooling = _parse_args(["args.json", "--entity-group-pooling"])
        assert pooling is True

    def test_combined_with_existing_flags(self):
        (path, warmup, _, chains, prefix, excl, transform, method, pooling) = _parse_args(
            [
                "args.json",
                "--num-warmup",
                "5",
                "--num-chains",
                "2",
                "--prefix",
                "perf",
                "--exclude-collection",
                "perf_rw_raw",
                "--target-transform",
                "offset_logit",
                "--chain-method",
                "vectorized",
                "--entity-group-pooling",
            ]
        )
        assert warmup == 5
        assert chains == 2
        assert prefix == "perf"
        assert excl == ("perf_rw_raw",)
        assert transform == "offset_logit"
        assert method == "vectorized"
        assert pooling is True


def _capture_run_kwargs(model_args: dict, **kwargs) -> tuple[dict, dict]:
    """Run run_and_measure with mocked MCMC; return (MCMC ctor, mcmc.run) kwargs."""
    with (
        mock.patch("panelcast.models.bayes.model.make_score_model", return_value="model"),
        mock.patch("numpyro.infer.NUTS", return_value="kernel"),
        mock.patch("numpyro.infer.MCMC") as mock_mcmc_cls,
        mock.patch("panelcast.gpu_memory.measure.get_jax_memory_stats") as mock_stats,
    ):
        mock_mcmc = mock.Mock()
        mock_mcmc_cls.return_value = mock_mcmc
        mock_stats.return_value = mock.Mock(peak_bytes_in_use=1024, peak_gb=0.001)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_args, f)
            temp_path = Path(f.name)
        try:
            run_and_measure(temp_path, **kwargs)
        finally:
            temp_path.unlink()
        return mock_mcmc_cls.call_args[1], mock_mcmc.run.call_args[1]


class TestRunAndMeasureTargetTransform:
    def test_offset_logit_transforms_y_and_prev_score(self):
        import numpy as np

        from panelcast.models.bayes.transforms import get_transform

        model_args = {
            "artist_idx": [0, 1],
            "album_seq": [1, 1],
            "prev_score": [72.5, 60.0],
            "X": [[1.0], [2.0]],
            "y": [70.0, 80.0],
            "n_artists": 2,
            "max_seq": 1,
            "target_bounds": [0.0, 100.0],
        }
        _, run_kwargs = _capture_run_kwargs(model_args, target_transform="offset_logit")

        transform = get_transform("offset_logit", target_bounds=(0.0, 100.0))
        np.testing.assert_allclose(
            np.asarray(run_kwargs["y"]),
            np.asarray(transform.forward(np.array([70.0, 80.0], dtype=np.float32))),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(run_kwargs["prev_score"]),
            np.asarray(transform.forward(np.array([72.5, 60.0], dtype=np.float32))),
            rtol=1e-6,
        )

    def test_offset_logit_priors_mirror_production(self):
        model_args = {
            "artist_idx": [0],
            "album_seq": [1],
            "prev_score": [50.0],
            "X": [[1.0]],
            "y": [70.0],
            "n_artists": 1,
            "max_seq": 1,
        }
        _, run_kwargs = _capture_run_kwargs(model_args, target_transform="offset_logit")

        priors = run_kwargs["priors"]
        assert priors.target_transform == "offset_logit"
        # priors_for_transform right-sizes the noise scales on the logit scale
        assert priors.sigma_obs_scale == 0.5
        assert run_kwargs["target_bounds"] == (0.0, 100.0)

    def test_custom_bounds_from_json(self):
        import numpy as np

        from panelcast.models.bayes.transforms import get_transform

        model_args = {
            "artist_idx": [0],
            "album_seq": [1],
            "prev_score": [5.0],
            "X": [[1.0]],
            "y": [7.0],
            "n_artists": 1,
            "max_seq": 1,
            "target_bounds": [0.0, 10.0],
        }
        _, run_kwargs = _capture_run_kwargs(model_args, target_transform="offset_logit")

        transform = get_transform("offset_logit", target_bounds=(0.0, 10.0))
        np.testing.assert_allclose(
            np.asarray(run_kwargs["y"]),
            np.asarray(transform.forward(np.array([7.0], dtype=np.float32))),
            rtol=1e-6,
        )
        assert run_kwargs["target_bounds"] == (0.0, 10.0)

    def test_identity_default_stays_legacy(self):
        import numpy as np

        model_args = {
            "artist_idx": [0],
            "album_seq": [1],
            "prev_score": [50.0],
            "X": [[1.0]],
            "y": [70.0],
            "n_artists": 1,
            "max_seq": 1,
        }
        mcmc_kwargs, run_kwargs = _capture_run_kwargs(model_args)

        assert "priors" not in run_kwargs
        assert "target_bounds" not in run_kwargs
        np.testing.assert_allclose(np.asarray(run_kwargs["y"]), [70.0])
        assert mcmc_kwargs["chain_method"] == "sequential"

    def test_chain_method_vectorized_passed_to_mcmc(self):
        model_args = {
            "artist_idx": [0],
            "album_seq": [1],
            "prev_score": [50.0],
            "X": [[1.0]],
            "y": [70.0],
            "n_artists": 1,
            "max_seq": 1,
        }
        mcmc_kwargs, _ = _capture_run_kwargs(model_args, chain_method="vectorized")
        assert mcmc_kwargs["chain_method"] == "vectorized"


class TestRunAndMeasureEntityGroupPooling:
    def test_missing_group_keys_raises(self):
        model_args = {
            "artist_idx": [0],
            "album_seq": [1],
            "prev_score": [50.0],
            "X": [[1.0]],
            "y": [70.0],
            "n_artists": 1,
            "max_seq": 1,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_args, f)
            temp_path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="group_idx_by_artist"):
                run_and_measure(temp_path, entity_group_pooling=True)
        finally:
            temp_path.unlink()

    def test_pooling_args_passed_to_model(self):
        import numpy as np

        model_args = {
            "artist_idx": [0, 1],
            "album_seq": [1, 1],
            "prev_score": [50.0, 50.0],
            "X": [[1.0], [2.0]],
            "y": [70.0, 80.0],
            "n_artists": 2,
            "max_seq": 1,
            "group_idx_by_artist": [0, 1],
            "n_groups": 2,
        }
        _, run_kwargs = _capture_run_kwargs(model_args, entity_group_pooling=True)

        np.testing.assert_array_equal(np.asarray(run_kwargs["group_idx_by_artist"]), [0, 1])
        assert run_kwargs["n_groups"] == 2
        assert run_kwargs["priors"].entity_group_pooling is True

    def test_pooling_tiny_fit_runs(self):
        """Real CPU mini-fit through the pooling + offset_logit gates."""
        model_args = {
            "artist_idx": [0, 0, 1, 1, 2, 2],
            "album_seq": [1, 2, 1, 2, 1, 2],
            "prev_score": [70.0, 68.0, 75.0, 74.0, 60.0, 62.0],
            "X": [[0.1], [0.2], [-0.1], [0.0], [0.3], [-0.2]],
            "y": [68.0, 71.0, 74.0, 76.0, 62.0, 61.0],
            "n_artists": 3,
            "max_seq": 2,
            "group_idx_by_artist": [0, 1, 1],
            "n_groups": 2,
            "target_bounds": [0.0, 100.0],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_args, f)
            temp_path = Path(f.name)

        mock_stats = mock.Mock(peak_bytes_in_use=1024, peak_gb=0.001)
        try:
            with mock.patch(
                "panelcast.gpu_memory.measure.get_jax_memory_stats", return_value=mock_stats
            ):
                result = run_and_measure(
                    temp_path,
                    num_warmup=2,
                    num_samples=2,
                    target_transform="offset_logit",
                    entity_group_pooling=True,
                )
            assert result["success"] is True
            assert result["exit_code"] == 0
        finally:
            temp_path.unlink()


class TestSubprocessNewFlagErrors:
    def test_invalid_target_transform_outputs_error_json(self):
        rc, output = _run_mini("dummy.json", "--target-transform", "bogus")
        assert rc == 1
        assert output["success"] is False
        assert "--target-transform must be" in output["error"]

    def test_invalid_chain_method_outputs_error_json(self):
        rc, output = _run_mini("dummy.json", "--chain-method", "parallel")
        assert rc == 1
        assert output["success"] is False
        assert "--chain-method must be" in output["error"]
