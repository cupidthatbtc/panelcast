"""Unit tests for model save/load and manifest management in io.py.

Tests cover:
- generate_model_filename: timestamp format
- get_git_commit: subprocess handling
- ModelManifest: serialization/deserialization
- ModelsManifest: history management
- save_manifest/load_manifest: file operations
- save_model/load_model: NetCDF persistence

Uses tmp_path for file isolation.
"""

import json
import re
from unittest.mock import MagicMock, patch

import arviz as az
import numpy as np
import pytest
import xarray as xr

from panelcast.models.bayes.io import (
    ModelManifest,
    ModelsManifest,
    _to_python_native,
    generate_model_filename,
    get_git_commit,
    load_manifest,
    load_model,
    save_manifest,
    save_model,
)
from panelcast.models.bayes.priors import PriorConfig

# =============================================================================
# Tests for generate_model_filename
# =============================================================================


class TestGenerateModelFilename:
    """Tests for generate_model_filename function."""

    def test_returns_correct_format(self):
        """Should return format {model_type}_{timestamp}.nc."""
        filename = generate_model_filename("user_score")

        # Should match pattern: user_score_YYYYMMDD_HHMMSS.nc
        pattern = r"^user_score_\d{8}_\d{6}\.nc$"
        assert re.match(pattern, filename), f"Filename {filename} doesn't match expected format"

    def test_timestamp_format(self):
        """Timestamp should be in YYYYMMDD_HHMMSS format."""
        filename = generate_model_filename("test_model")

        # Extract timestamp
        match = re.search(r"_(\d{8}_\d{6})\.nc$", filename)
        assert match, f"Could not extract timestamp from {filename}"

        timestamp = match.group(1)
        # Verify it's a valid date format
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS = 8+1+6 = 15

    def test_different_model_types(self):
        """Different model types should produce different prefixes."""
        user_filename = generate_model_filename("user_score")
        critic_filename = generate_model_filename("critic_score")

        assert user_filename.startswith("user_score_")
        assert critic_filename.startswith("critic_score_")

    def test_custom_model_type(self):
        """Should work with custom model type names."""
        filename = generate_model_filename("custom_model")
        assert filename.startswith("custom_model_")
        assert filename.endswith(".nc")


# =============================================================================
# Tests for get_git_commit
# =============================================================================


class TestGetGitCommit:
    """Tests for get_git_commit function."""

    def test_returns_string(self):
        """Should return a string."""
        result = get_git_commit()
        assert isinstance(result, str)

    def test_in_git_repo_returns_40_char_hex(self):
        """In git repo, should return 40-char hex commit hash."""
        result = get_git_commit()

        # Should be either "unknown" or a 40-char hex string
        if result != "unknown":
            assert len(result) == 40
            assert all(c in "0123456789abcdef" for c in result)

    @patch("panelcast.models.bayes.io.subprocess.run")
    def test_git_unavailable_returns_unknown(self, mock_run):
        """Should return 'unknown' when git is unavailable."""
        mock_run.side_effect = FileNotFoundError("git not found")

        result = get_git_commit()
        assert result == "unknown"

    @patch("panelcast.models.bayes.io.subprocess.run")
    def test_git_timeout_returns_unknown(self, mock_run):
        """Should return 'unknown' on timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("git", 5)

        result = get_git_commit()
        assert result == "unknown"

    @patch("panelcast.models.bayes.io.subprocess.run")
    def test_git_nonzero_exit_returns_unknown(self, mock_run):
        """Should return 'unknown' when git returns non-zero exit code."""
        mock_result = MagicMock()
        mock_result.returncode = 128  # Not a git repo
        mock_run.return_value = mock_result

        result = get_git_commit()
        assert result == "unknown"


# =============================================================================
# Tests for ModelManifest
# =============================================================================


class TestModelManifest:
    """Tests for ModelManifest dataclass."""

    @pytest.fixture
    def sample_manifest(self):
        """Create a sample ModelManifest."""
        return ModelManifest(
            version="1.0",
            created_at="2026-01-26T12:00:00Z",
            model_type="user_score",
            filename="user_score_20260126_120000.nc",
            mcmc_config={"num_warmup": 1000, "num_samples": 1000},
            priors={"mu_artist_loc": 70.0},
            data_hash="abc123",
            git_commit="0123456789abcdef0123456789abcdef01234567",
            gpu_info="NVIDIA Test GPU",
            runtime_seconds=100.5,
            divergences=0,
        )

    def test_to_dict_serialization(self, sample_manifest):
        """to_dict should serialize all fields."""
        d = sample_manifest.to_dict()

        assert isinstance(d, dict)
        assert d["version"] == "1.0"
        assert d["model_type"] == "user_score"
        assert d["filename"] == "user_score_20260126_120000.nc"
        assert d["mcmc_config"]["num_warmup"] == 1000
        assert d["priors"]["mu_artist_loc"] == 70.0
        assert d["data_hash"] == "abc123"
        assert d["divergences"] == 0

    def test_from_dict_deserialization(self):
        """from_dict should create valid ModelManifest."""
        d = {
            "version": "1.0",
            "created_at": "2026-01-26T12:00:00Z",
            "model_type": "critic_score",
            "filename": "critic_score_20260126_120000.nc",
            "mcmc_config": {"num_chains": 4},
            "priors": {},
            "data_hash": "xyz789",
            "git_commit": "abc",
            "gpu_info": "CPU only",
            "runtime_seconds": 50.0,
            "divergences": 5,
        }

        manifest = ModelManifest.from_dict(d)

        assert manifest.model_type == "critic_score"
        assert manifest.divergences == 5

    def test_roundtrip(self, sample_manifest):
        """to_dict -> from_dict should preserve data."""
        d = sample_manifest.to_dict()
        restored = ModelManifest.from_dict(d)

        assert restored == sample_manifest

    def test_frozen(self, sample_manifest):
        """ModelManifest should be frozen (immutable)."""
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            sample_manifest.version = "2.0"


# =============================================================================
# Tests for ModelsManifest
# =============================================================================


class TestModelsManifest:
    """Tests for ModelsManifest dataclass."""

    def test_empty_manifest_creation(self):
        """Should create empty manifest with defaults."""
        manifest = ModelsManifest()

        assert manifest.version == "1.0"
        assert manifest.current == {}
        assert manifest.history == []

    def test_to_dict_with_history(self):
        """to_dict should include history entries."""
        model_manifest = ModelManifest(
            version="1.0",
            created_at="2026-01-26T12:00:00Z",
            model_type="user_score",
            filename="user_score_test.nc",
            mcmc_config={},
            priors={},
            data_hash="hash",
            git_commit="commit",
            gpu_info="GPU",
            runtime_seconds=1.0,
            divergences=0,
        )

        manifest = ModelsManifest(
            version="1.0",
            current={"user_score": "user_score_test.nc"},
            history=[model_manifest],
        )

        d = manifest.to_dict()

        assert d["version"] == "1.0"
        assert d["current"]["user_score"] == "user_score_test.nc"
        assert len(d["history"]) == 1
        assert d["history"][0]["filename"] == "user_score_test.nc"

    def test_from_dict_with_history(self):
        """from_dict should restore history as ModelManifest objects."""
        d = {
            "version": "1.0",
            "current": {"user_score": "test.nc"},
            "history": [
                {
                    "version": "1.0",
                    "created_at": "2026-01-26T12:00:00Z",
                    "model_type": "user_score",
                    "filename": "test.nc",
                    "mcmc_config": {},
                    "priors": {},
                    "data_hash": "hash",
                    "git_commit": "commit",
                    "gpu_info": "GPU",
                    "runtime_seconds": 1.0,
                    "divergences": 0,
                }
            ],
        }

        manifest = ModelsManifest.from_dict(d)

        assert len(manifest.history) == 1
        assert isinstance(manifest.history[0], ModelManifest)
        assert manifest.history[0].filename == "test.nc"

    def test_roundtrip(self):
        """to_dict -> from_dict should preserve data."""
        model_manifest = ModelManifest(
            version="1.0",
            created_at="2026-01-26T12:00:00Z",
            model_type="user_score",
            filename="test.nc",
            mcmc_config={"seed": 42},
            priors={"alpha": 2.0},
            data_hash="hash123",
            git_commit="commit456",
            gpu_info="NVIDIA GPU",
            runtime_seconds=99.9,
            divergences=3,
        )

        original = ModelsManifest(
            version="1.0",
            current={"user_score": "test.nc"},
            history=[model_manifest],
        )

        d = original.to_dict()
        restored = ModelsManifest.from_dict(d)

        assert restored.version == original.version
        assert restored.current == original.current
        assert len(restored.history) == len(original.history)
        assert restored.history[0].filename == original.history[0].filename

    def test_history_list_management(self):
        """Should be able to modify history list."""
        manifest = ModelsManifest()

        model = ModelManifest(
            version="1.0",
            created_at="2026-01-26T12:00:00Z",
            model_type="user_score",
            filename="test.nc",
            mcmc_config={},
            priors={},
            data_hash="hash",
            git_commit="commit",
            gpu_info="GPU",
            runtime_seconds=1.0,
            divergences=0,
        )

        # Should be able to append to history
        manifest.history.insert(0, model)
        assert len(manifest.history) == 1

        # Should be able to update current
        manifest.current["user_score"] = "test.nc"
        assert manifest.current["user_score"] == "test.nc"


# =============================================================================
# Tests for save_manifest and load_manifest
# =============================================================================


class TestManifestFileOperations:
    """Tests for save_manifest and load_manifest functions."""

    def test_save_creates_manifest_json(self, tmp_path):
        """save_manifest should create manifest.json in output_dir."""
        manifest = ModelsManifest(version="1.0", current={}, history=[])

        result_path = save_manifest(manifest, tmp_path)

        assert result_path.exists()
        assert result_path.name == "manifest.json"
        assert result_path.parent == tmp_path

    def test_load_returns_none_for_nonexistent(self, tmp_path):
        """load_manifest should return None if file doesn't exist."""
        result = load_manifest(tmp_path)
        assert result is None

    def test_load_handles_malformed_json(self, tmp_path):
        """load_manifest should handle malformed JSON gracefully."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{ invalid json }", encoding="utf-8")

        result = load_manifest(tmp_path)
        assert result is None

    def test_roundtrip(self, tmp_path):
        """save -> load should preserve manifest data."""
        model = ModelManifest(
            version="1.0",
            created_at="2026-01-26T12:00:00Z",
            model_type="user_score",
            filename="test.nc",
            mcmc_config={"num_warmup": 500},
            priors={"sigma": 1.0},
            data_hash="hash",
            git_commit="commit",
            gpu_info="GPU",
            runtime_seconds=50.0,
            divergences=1,
        )

        original = ModelsManifest(
            version="1.0",
            current={"user_score": "test.nc"},
            history=[model],
        )

        save_manifest(original, tmp_path)
        loaded = load_manifest(tmp_path)

        assert loaded is not None
        assert loaded.version == original.version
        assert loaded.current == original.current
        assert len(loaded.history) == 1
        assert loaded.history[0].filename == "test.nc"

    def test_save_creates_directory_if_missing(self, tmp_path):
        """save_manifest should create output_dir if it doesn't exist."""
        nested_dir = tmp_path / "models" / "nested"
        manifest = ModelsManifest()

        save_manifest(manifest, nested_dir)

        assert (nested_dir / "manifest.json").exists()

    def test_manifest_json_is_valid_json(self, tmp_path):
        """Saved manifest.json should be valid JSON."""
        manifest = ModelsManifest(current={"test": "value"})
        save_manifest(manifest, tmp_path)

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["current"]["test"] == "value"


# =============================================================================
# Tests for save_model
# =============================================================================


class TestSaveModel:
    """Tests for save_model function."""

    @pytest.fixture
    def mock_fit_result(self):
        """Create a mock FitResult with mock idata."""
        # Create minimal InferenceData
        posterior = xr.Dataset(
            {"param": xr.DataArray(np.random.randn(2, 10), dims=["chain", "draw"])}
        )
        idata = az.InferenceData(posterior=posterior)

        mock_result = MagicMock()
        mock_result.idata = idata
        mock_result.divergences = 2
        mock_result.runtime_seconds = 45.5
        mock_result.gpu_info = "Mock GPU"
        # Mock MCMC object
        mock_result.mcmc = MagicMock()
        mock_result.mcmc._kernel_params = {"max_tree_depth": 10}

        return mock_result

    def test_creates_nc_file(self, tmp_path, mock_fit_result):
        """save_model should create .nc file in output_dir."""
        priors = PriorConfig()

        path, manifest = save_model(
            fit_result=mock_fit_result,
            model_type="user_score",
            priors=priors,
            data_hash="test_hash",
            output_dir=tmp_path,
        )

        assert path.exists()
        assert path.suffix == ".nc"
        assert path.parent == tmp_path

    def test_updates_manifest(self, tmp_path, mock_fit_result):
        """save_model should update manifest.json with new entry."""
        priors = PriorConfig()

        path, manifest = save_model(
            fit_result=mock_fit_result,
            model_type="user_score",
            priors=priors,
            data_hash="test_hash",
            output_dir=tmp_path,
        )

        # Load manifest and verify
        loaded_manifest = load_manifest(tmp_path)
        assert loaded_manifest is not None
        assert "user_score" in loaded_manifest.current
        assert len(loaded_manifest.history) == 1

    def test_returns_path_and_manifest(self, tmp_path, mock_fit_result):
        """save_model should return (Path, ModelManifest) tuple."""
        priors = PriorConfig()

        result = save_model(
            fit_result=mock_fit_result,
            model_type="user_score",
            priors=priors,
            data_hash="test_hash",
            output_dir=tmp_path,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

        path, manifest = result
        from pathlib import Path

        assert isinstance(path, Path)
        assert isinstance(manifest, ModelManifest)

    def test_manifest_contains_correct_metadata(self, tmp_path, mock_fit_result):
        """Returned manifest should have correct metadata."""
        priors = PriorConfig()

        path, manifest = save_model(
            fit_result=mock_fit_result,
            model_type="user_score",
            priors=priors,
            data_hash="my_data_hash",
            output_dir=tmp_path,
        )

        assert manifest.model_type == "user_score"
        assert manifest.data_hash == "my_data_hash"
        assert manifest.divergences == 2
        assert manifest.runtime_seconds == 45.5
        assert manifest.gpu_info == "Mock GPU"


# =============================================================================
# Tests for load_model
# =============================================================================


class TestLoadModel:
    """Tests for load_model function."""

    def test_loads_netcdf_correctly(self, tmp_path):
        """load_model should return InferenceData from NetCDF."""
        # Create and save minimal InferenceData
        posterior = xr.Dataset(
            {"beta": xr.DataArray(np.random.randn(2, 10), dims=["chain", "draw"])}
        )
        idata = az.InferenceData(posterior=posterior)

        filepath = tmp_path / "test_model.nc"
        idata.to_netcdf(filepath)

        # Load and verify
        loaded = load_model(filepath)

        assert isinstance(loaded, az.InferenceData)
        assert "posterior" in loaded.groups()
        assert "beta" in loaded.posterior

    def test_returns_inference_data(self, tmp_path):
        """load_model should return az.InferenceData object."""
        # Create minimal file
        posterior = xr.Dataset({"param": xr.DataArray(np.zeros((1, 5)), dims=["chain", "draw"])})
        idata = az.InferenceData(posterior=posterior)

        filepath = tmp_path / "model.nc"
        idata.to_netcdf(filepath)

        result = load_model(filepath)

        assert isinstance(result, az.InferenceData)

    def test_preserves_all_groups(self, tmp_path):
        """load_model should preserve all InferenceData groups."""
        # Create InferenceData with multiple groups
        posterior = xr.Dataset({"mu": xr.DataArray(np.random.randn(2, 10), dims=["chain", "draw"])})
        observed = xr.Dataset({"y": xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["obs"])})
        idata = az.InferenceData(posterior=posterior, observed_data=observed)

        filepath = tmp_path / "full_model.nc"
        idata.to_netcdf(filepath)

        loaded = load_model(filepath)

        assert "posterior" in loaded.groups()
        assert "observed_data" in loaded.groups()
        assert "mu" in loaded.posterior
        assert "y" in loaded.observed_data


# =============================================================================
# Tests for _to_python_native
# =============================================================================


class TestToPythonNative:
    """Tests for _to_python_native helper."""

    def test_converts_numpy_scalar(self):
        """Should convert numpy scalar to Python float."""
        result = _to_python_native(np.float64(3.14))
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_converts_numpy_int(self):
        """Should convert numpy int to Python int."""
        result = _to_python_native(np.int32(42))
        assert isinstance(result, int)
        assert result == 42

    def test_converts_numpy_array(self):
        """Should convert numpy array to Python list."""
        result = _to_python_native(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_converts_nested_dict(self):
        """Should recursively convert nested dicts."""
        data = {"a": np.float64(1.0), "b": {"c": np.int32(2)}}
        result = _to_python_native(data)
        assert isinstance(result["a"], float)
        assert isinstance(result["b"]["c"], int)

    def test_passes_through_python_types(self):
        """Should pass through native Python types unchanged."""
        assert _to_python_native(42) == 42
        assert _to_python_native("hello") == "hello"
        assert _to_python_native(3.14) == 3.14
        assert _to_python_native(None) is None

    def test_converts_list_of_numpy(self):
        """Should convert lists containing numpy values."""
        result = _to_python_native([np.float64(1.0), np.int32(2)])
        assert isinstance(result, list)
        assert isinstance(result[0], float)
        assert isinstance(result[1], int)

    def test_save_model_with_jax_kernel_params(self, tmp_path):
        """save_model should serialize mcmc_config with numpy/JAX values."""
        posterior = xr.Dataset(
            {"param": xr.DataArray(np.random.randn(2, 10), dims=["chain", "draw"])}
        )
        idata = az.InferenceData(posterior=posterior)

        mock_result = MagicMock()
        mock_result.idata = idata
        mock_result.divergences = 0
        mock_result.runtime_seconds = 1.0
        mock_result.gpu_info = "CPU"
        mock_result.mcmc = MagicMock(spec=[])  # empty spec avoids auto-creating attrs
        mock_result.mcmc._kernel_params = {
            "max_tree_depth": np.int32(10),
            "target_accept": np.float64(0.9),
        }

        priors = PriorConfig()
        path, manifest = save_model(
            fit_result=mock_result,
            model_type="user_score",
            priors=priors,
            data_hash="test",
            output_dir=tmp_path,
        )

        # Verify the mcmc_config values are native Python types (JSON-serializable)
        assert isinstance(manifest.mcmc_config["max_tree_depth"], int)
        assert isinstance(manifest.mcmc_config["target_accept"], float)

        # Verify the manifest was actually saved as valid JSON
        loaded = load_manifest(tmp_path)
        assert loaded is not None
        assert loaded.history[0].mcmc_config["max_tree_depth"] == 10
