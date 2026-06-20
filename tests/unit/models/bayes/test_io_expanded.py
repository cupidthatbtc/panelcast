"""Expanded tests for model I/O: generate_model_filename, get_git_commit, _to_python_native."""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from panelcast.models.bayes.io import (
    ModelManifest,
    ModelsManifest,
    _to_python_native,
    generate_model_filename,
    get_git_commit,
)
from panelcast.models.bayes.priors import PriorConfig


class TestGenerateModelFilenameExpanded:
    """Extended tests for generate_model_filename."""

    def test_user_score_prefix(self):
        filename = generate_model_filename("user_score")
        assert filename.startswith("user_score_")

    def test_critic_score_prefix(self):
        filename = generate_model_filename("critic_score")
        assert filename.startswith("critic_score_")

    def test_custom_prefix(self):
        filename = generate_model_filename("custom_model")
        assert filename.startswith("custom_model_")

    def test_ends_with_nc(self):
        filename = generate_model_filename("test")
        assert filename.endswith(".nc")

    def test_unique_per_call(self):
        """Two calls should produce different filenames (different timestamps)."""
        f1 = generate_model_filename("test")
        f2 = generate_model_filename("test")
        # May be the same if called within the same second, so just check format
        pattern = r"^test_\d{8}_\d{6}\.nc$"
        assert re.match(pattern, f1)
        assert re.match(pattern, f2)

    def test_empty_model_type(self):
        filename = generate_model_filename("")
        assert filename.startswith("_")
        assert filename.endswith(".nc")


class TestGetGitCommitExpanded:
    """Extended tests for get_git_commit."""

    @patch("panelcast.models.bayes.io.subprocess.run")
    def test_returns_commit_hash(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123def456")
        result = get_git_commit()
        assert result == "abc123def456"

    @patch("panelcast.models.bayes.io.subprocess.run")
    def test_strips_whitespace(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="  abc123  \n")
        result = get_git_commit()
        assert result == "abc123"

    @patch("panelcast.models.bayes.io.subprocess.run")
    def test_fallback_on_error(self, mock_run):
        mock_run.side_effect = FileNotFoundError("git not found")
        result = get_git_commit()
        assert result == "unknown"

    @patch("panelcast.models.bayes.io.subprocess.run")
    def test_fallback_on_nonzero_returncode(self, mock_run):
        mock_run.return_value = MagicMock(returncode=128, stdout="")
        result = get_git_commit()
        assert result == "unknown"


class TestToPythonNative:
    """Tests for _to_python_native conversion."""

    def test_numpy_int(self):
        result = _to_python_native(np.int64(42))
        assert isinstance(result, int)
        assert result == 42

    def test_numpy_float(self):
        result = _to_python_native(np.float64(3.14))
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_numpy_bool(self):
        result = _to_python_native(np.bool_(True))
        assert isinstance(result, bool)
        assert result is True

    def test_python_int(self):
        result = _to_python_native(42)
        assert isinstance(result, int)
        assert result == 42

    def test_python_float(self):
        result = _to_python_native(3.14)
        assert isinstance(result, float)

    def test_string(self):
        result = _to_python_native("hello")
        assert result == "hello"

    def test_none(self):
        result = _to_python_native(None)
        assert result is None

    def test_list_of_numpy(self):
        result = _to_python_native([np.int64(1), np.int64(2)])
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_dict_of_numpy(self):
        result = _to_python_native({"a": np.float64(1.0), "b": np.int64(2)})
        assert isinstance(result, dict)
        assert isinstance(result["a"], float)
        assert isinstance(result["b"], int)

    def test_nested_structure(self):
        data = {"list": [np.int64(1), np.float64(2.0)], "val": np.bool_(False)}
        result = _to_python_native(data)
        assert result == {"list": [1, 2.0], "val": False}


class TestModelManifestExpanded:
    """Extended tests for ModelManifest."""

    @pytest.fixture
    def sample_manifest(self):
        return ModelManifest(
            version="1.0",
            created_at="2026-01-15T10:00:00",
            model_type="user_score",
            filename="user_score_20260115_100000.nc",
            mcmc_config={"num_warmup": 1000, "num_samples": 1000, "num_chains": 4},
            priors={"mu_artist_loc": 0.0, "mu_artist_scale": 1.0},
            data_hash="abc123def456",
            git_commit="deadbeef",
            gpu_info="NVIDIA RTX 3080, 10240 MiB",
            runtime_seconds=120.5,
            divergences=0,
        )

    def test_fields_accessible(self, sample_manifest):
        assert sample_manifest.model_type == "user_score"
        assert sample_manifest.filename.endswith(".nc")
        assert sample_manifest.divergences == 0
        assert sample_manifest.runtime_seconds == 120.5

    def test_mcmc_config_is_dict(self, sample_manifest):
        assert isinstance(sample_manifest.mcmc_config, dict)
        assert sample_manifest.mcmc_config["num_chains"] == 4

    def test_priors_is_dict(self, sample_manifest):
        assert isinstance(sample_manifest.priors, dict)
        assert sample_manifest.priors["mu_artist_loc"] == 0.0

    def test_to_dict_returns_dict(self, sample_manifest):
        d = sample_manifest.to_dict()
        assert isinstance(d, dict)
        assert d["model_type"] == "user_score"
        assert d["version"] == "1.0"


class TestModelsManifestExpanded:
    """Extended tests for ModelsManifest."""

    def test_empty_manifest(self):
        manifest = ModelsManifest()
        assert len(manifest.history) == 0
        assert manifest.current == {}

    def test_default_version(self):
        manifest = ModelsManifest()
        assert manifest.version == "1.0"

    def test_add_to_history(self):
        manifest = ModelsManifest()
        model = ModelManifest(
            version="1.0",
            created_at="2026-01-15T10:00:00",
            model_type="user_score",
            filename="test.nc",
            mcmc_config={},
            priors={},
            data_hash="abc",
            git_commit="def",
            gpu_info="CPU",
            runtime_seconds=1.0,
            divergences=0,
        )
        manifest.history.append(model)
        assert len(manifest.history) == 1

    def test_to_dict(self):
        manifest = ModelsManifest()
        d = manifest.to_dict()
        assert isinstance(d, dict)
        assert "version" in d
        assert "current" in d
        assert "history" in d
