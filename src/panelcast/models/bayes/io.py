"""Model save/load and manifest management for reproducibility.

This module provides infrastructure for persisting fitted models and tracking
them via a manifest file. Key features:
- NetCDF persistence via ArviZ InferenceData.to_netcdf()
- ModelManifest captures all metadata for reproducibility
- ModelsManifest tracks current models and full history
- Git commit hash captured for version control integration
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import arviz as az
import structlog

from panelcast.models.bayes.priors import PriorConfig

if TYPE_CHECKING:
    from panelcast.models.bayes.fit import FitResult

logger = structlog.get_logger(__name__)

__all__ = [
    "generate_model_filename",
    "get_git_commit",
    "load_manifest",
    "load_model",
    "ModelManifest",
    "ModelsManifest",
    "save_manifest",
    "save_model",
]


@dataclass(frozen=True)
class ModelManifest:
    """Manifest for a single fitted model.

    Captures all metadata needed to reproduce and understand a model fit.
    Frozen to prevent accidental modification.

    Attributes:
        version: Manifest schema version (e.g., "1.0").
        created_at: ISO timestamp when model was fitted.
        model_type: Type of model ("user_score" or "critic_score").
        filename: Name of the NetCDF file (e.g., "user_score_20260119_143052.nc").
        mcmc_config: MCMC configuration as dictionary.
        priors: Prior configuration as dictionary.
        data_hash: SHA256 hash of training data for verification.
        git_commit: Git commit hash at time of fitting.
        gpu_info: String describing GPU used.
        runtime_seconds: Wall-clock time for fitting.
        divergences: Number of divergent transitions.
    """

    version: str
    created_at: str
    model_type: str
    filename: str
    mcmc_config: dict[str, Any]
    priors: dict[str, Any]
    data_hash: str
    git_commit: str
    gpu_info: str
    runtime_seconds: float
    divergences: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelManifest":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class ModelsManifest:
    """Manifest tracking all fitted models.

    Maintains pointers to current models by type and full history
    of all fitted models for audit trail.

    Attributes:
        version: Manifest schema version (e.g., "1.0").
        current: Mapping of model_type -> filename for current active models.
        history: List of all ModelManifest entries (most recent first).
    """

    version: str = "1.0"
    current: dict[str, str] = field(default_factory=dict)
    history: list[ModelManifest] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "current": self.current,
            "history": [m.to_dict() for m in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelsManifest":
        """Create from dictionary."""
        return cls(
            version=d.get("version", "1.0"),
            current=d.get("current", {}),
            history=[ModelManifest.from_dict(m) for m in d.get("history", [])],
        )


def _to_python_native(obj: Any) -> Any:
    """Recursively convert JAX/numpy values to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_python_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_python_native(v) for v in obj)
    # Handle numpy/JAX arrays and scalars via tolist (works for both)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def generate_model_filename(model_type: str) -> str:
    """Generate timestamped filename for model.

    Format: {model_type}_{timestamp}.nc
    Example: user_score_20260119_143052.nc

    Parameters
    ----------
    model_type : str
        Type of model ("user_score" or "critic_score").

    Returns
    -------
    str
        Filename with timestamp in format YYYYMMDD_HHMMSS.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{model_type}_{timestamp}.nc"


def get_git_commit() -> str:
    """Get current git commit hash.

    Returns "unknown" if not in a git repository or if git is unavailable.

    Returns
    -------
    str
        Git commit hash (40 characters) or "unknown".
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            logger.debug(
                "git_commit_lookup_failed",
                returncode=result.returncode,
                output=result.stderr.strip(),
                reason="git_nonzero_exit",
            )
    except FileNotFoundError as e:
        logger.debug("git_commit_lookup_failed", error=str(e), reason="git_not_found")
    except subprocess.TimeoutExpired as e:
        logger.debug("git_commit_lookup_failed", error=str(e), reason="timeout")
    except OSError as e:
        logger.debug("git_commit_lookup_failed", error=str(e), reason="os_error")
    return "unknown"


def save_model(
    fit_result: FitResult,
    model_type: str,
    priors: PriorConfig,
    data_hash: str,
    output_dir: Path = Path("models"),
) -> tuple[Path, ModelManifest]:
    """Save fitted model to NetCDF and update manifest.

    Persists the InferenceData to a NetCDF file and updates the
    models/manifest.json with metadata for reproducibility.

    Parameters
    ----------
    fit_result : FitResult
        Result from fit_model() containing idata, divergences, etc.
    model_type : str
        Type of model ("user_score" or "critic_score").
    priors : PriorConfig
        Prior configuration used for fitting.
    data_hash : str
        SHA256 hash of training data for verification.
    output_dir : Path, default Path("models")
        Directory to save model files.

    Returns
    -------
    tuple[Path, ModelManifest]
        Path to saved NetCDF file and the created ModelManifest.

    Example
    -------
    >>> from panelcast.models.bayes import fit_model, save_model, user_score_model
    >>> from panelcast.models.bayes.priors import get_default_priors
    >>> from panelcast.utils.hashing import hash_dataframe
    >>>
    >>> result = fit_model(user_score_model, model_args)
    >>> path, manifest = save_model(
    ...     result,
    ...     model_type="user_score",
    ...     priors=get_default_priors(),
    ...     data_hash=hash_dataframe(train_df),
    ... )
    >>> print(f"Saved to: {path}")
    """
    # Create output directory if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    filename = generate_model_filename(model_type)
    filepath = output_dir / filename

    # Save InferenceData to NetCDF
    fit_result.idata.to_netcdf(filepath)

    # Extract MCMC kernel config for manifest metadata.
    # Try _kernel_params first (available in current NumPyro versions),
    # then fall back to empty dict if the attribute doesn't exist or
    # contains non-serializable values.
    mcmc_config: dict = {}
    if hasattr(fit_result.mcmc, "_kernel_params"):
        raw = fit_result.mcmc._kernel_params
        if isinstance(raw, dict):
            mcmc_config = _to_python_native(raw)
    manifest = ModelManifest(
        version="1.0",
        created_at=datetime.now(timezone.utc).isoformat(),
        model_type=model_type,
        filename=filename,
        mcmc_config=mcmc_config,
        priors=asdict(priors),
        data_hash=data_hash,
        git_commit=get_git_commit(),
        gpu_info=fit_result.gpu_info,
        runtime_seconds=fit_result.runtime_seconds,
        divergences=fit_result.divergences,
    )

    # Update models manifest
    models_manifest = load_manifest(output_dir)
    if models_manifest is None:
        models_manifest = ModelsManifest()

    # Update current pointer for this model type
    models_manifest.current[model_type] = filename

    # Add to history (most recent first)
    models_manifest.history.insert(0, manifest)

    # Save updated manifest
    save_manifest(models_manifest, output_dir)

    return filepath, manifest


def load_model(filepath: Path) -> az.InferenceData:
    """Load model from NetCDF file.

    Simple wrapper around az.from_netcdf for consistency.

    Parameters
    ----------
    filepath : Path
        Path to the NetCDF file.

    Returns
    -------
    az.InferenceData
        Loaded InferenceData with posterior, observed_data, etc.

    Example
    -------
    >>> idata = load_model(Path("models/user_score_20260119_143052.nc"))
    >>> print(idata.groups())
    """
    return az.from_netcdf(filepath)


def load_manifest(output_dir: Path = Path("models")) -> ModelsManifest | None:
    """Load models manifest from JSON file.

    Parameters
    ----------
    output_dir : Path, default Path("models")
        Directory containing manifest.json.

    Returns
    -------
    ModelsManifest or None
        Loaded manifest, or None if file doesn't exist.
    """
    manifest_path = Path(output_dir) / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("manifest_parse_failed", path=str(manifest_path), error=str(e))
        return None
    return ModelsManifest.from_dict(data)


def save_manifest(manifest: ModelsManifest, output_dir: Path = Path("models")) -> Path:
    """Save models manifest to JSON file.

    Parameters
    ----------
    manifest : ModelsManifest
        Manifest to save.
    output_dir : Path, default Path("models")
        Directory to save manifest.json.

    Returns
    -------
    Path
        Path to saved manifest file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return manifest_path
