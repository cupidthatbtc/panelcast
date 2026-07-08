"""Run manifest schema and I/O for pipeline reproducibility.

This module provides the RunManifest schema that captures everything needed
to reproduce a pipeline run: command, flags, git state, environment (including
pixi.lock hash), input hashes, and stage execution metadata.
"""

import importlib
import platform
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from panelcast.utils.environment import verify_environment
from panelcast.utils.git_state import GitState


class EnvironmentInfo(BaseModel):
    """Software environment information for reproducibility.

    Captures Python and key package versions, platform info, and
    the pixi.lock hash for exact environment reproducibility.

    Attributes:
        python_version: Python version string (e.g., "3.11.5")
        jax_version: JAX version string (e.g., "0.4.26")
        numpyro_version: NumPyro version if installed, None otherwise
        arviz_version: ArviZ version if installed, None otherwise
        platform: Platform description (e.g., "Windows 11")
        pixi_lock_hash: SHA256 hash of pixi.lock file, None if not found
    """

    python_version: str
    jax_version: str
    numpyro_version: str | None
    arviz_version: str | None
    platform: str
    pixi_lock_hash: str | None


def _get_version(module_name: str) -> str | None:
    """Get version of installed package.

    Args:
        module_name: Name of the module to check.

    Returns:
        Version string if installed, None otherwise.
    """
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


def capture_environment() -> EnvironmentInfo:
    """Capture current software environment.

    Gathers Python version, key package versions (JAX, NumPyro, ArviZ),
    platform information, and pixi.lock hash for reproducibility tracking.

    Returns:
        EnvironmentInfo with current environment details.

    Example:
        >>> env = capture_environment()
        >>> env.python_version  # e.g., "3.11.5"
        >>> env.pixi_lock_hash  # SHA256 hash or None
    """
    # Get pixi.lock hash from environment verification
    env_status = verify_environment()
    pixi_lock_hash = env_status.pixi_lock_hash

    return EnvironmentInfo(
        python_version=sys.version.split()[0],
        jax_version=_get_version("jax") or "not installed",
        numpyro_version=_get_version("numpyro"),
        arviz_version=_get_version("arviz"),
        platform=f"{platform.system()} {platform.release()}",
        pixi_lock_hash=pixi_lock_hash,
    )


class GitStateModel(BaseModel):
    """Pydantic model wrapper for GitState dataclass.

    Enables JSON serialization of GitState within RunManifest.
    """

    commit: str
    branch: str
    dirty: bool
    untracked_count: int

    @classmethod
    def from_git_state(cls, git_state: GitState) -> "GitStateModel":
        """Create from GitState dataclass."""
        return cls(
            commit=git_state.commit,
            branch=git_state.branch,
            dirty=git_state.dirty,
            untracked_count=git_state.untracked_count,
        )

    def to_git_state(self) -> GitState:
        """Convert back to GitState dataclass."""
        return GitState(
            commit=self.commit,
            branch=self.branch,
            dirty=self.dirty,
            untracked_count=self.untracked_count,
        )


class RunManifest(BaseModel):
    """Complete manifest for pipeline run reproducibility.

    Captures everything needed to reproduce a run: command invocation,
    parsed flags, random seed, git state, environment (with pixi.lock hash),
    input file hashes, per-stage execution metadata, and outputs.

    Attributes:
        run_id: Timestamp-based identifier (e.g., "2026-01-19_143052_123456_a3f9")
        created_at: ISO 8601 timestamp when run started
        command: Full CLI invocation string
        flags: Parsed flag values (seed, skip_existing, etc.)
        seed: Random seed used for reproducibility
        git: Git repository state at run time
        environment: Software environment details (including pixi_lock_hash)
        input_hashes: Mapping of input file paths to SHA256 hashes
        stage_hashes: Mapping of stage names to input hashes when executed
        stages_completed: List of stage names that completed successfully
        stages_skipped: List of stage names that were skipped (unchanged inputs)
        outputs: Mapping of artifact names to output paths
        success: Whether the run completed successfully
        error: Error message if run failed, None otherwise
        duration_seconds: Total run duration in seconds
        version: panelcast package version that produced the run, None on
            manifests written by older versions
        tag: Optional free-form label (``panelcast run --tag``), None if unset
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    created_at: str
    command: str
    flags: dict[str, Any]
    seed: int
    git: GitStateModel
    environment: EnvironmentInfo
    input_hashes: dict[str, str]
    stage_hashes: dict[str, str]
    stages_completed: list[str]
    stages_skipped: list[str]
    outputs: dict[str, str]
    success: bool
    error: str | None = None
    duration_seconds: float = 0.0
    # Cross-run provenance for `runs history`: the package version that
    # produced the run and an optional user label. None on legacy manifests.
    version: str | None = None
    tag: str | None = None
    # Data-root stamps as observed by this run (stage name -> stamp payload);
    # consumer stages verify these against the on-disk stamps to fail fast on
    # artifacts regenerated by another run mid-flight.
    data_stamps: dict[str, dict[str, Any]] = {}
    # Per-stage wall-clock seconds and expected-vs-actual resource telemetry
    # (#78): every fit becomes a calibration datapoint for the estimator.
    stage_durations: dict[str, float] = {}
    resources: dict[str, dict[str, Any]] = {}


def generate_run_id() -> str:
    """Generate a sortable, collision-resistant run identifier.

    The second-resolution timestamp alone let two runs started in the same
    second silently share a run directory; microseconds plus a random suffix
    keep ids unique while lexicographic order still follows creation time.

    Returns:
        Run ID in format "YYYY-MM-DD_HHMMSS_ffffff_xxxx"
        (e.g., "2026-01-19_143052_123456_a3f9").

    Example:
        >>> run_id = generate_run_id()
        >>> len(run_id) == 29
        True
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    return f"{stamp}_{secrets.token_hex(2)}"


def save_run_manifest(manifest: RunManifest, run_dir: Path) -> Path:
    """Save run manifest to JSON file.

    Args:
        manifest: RunManifest to save.
        run_dir: Directory to save manifest in.

    Returns:
        Path to saved manifest file (run_dir/manifest.json).

    Example:
        >>> path = save_run_manifest(manifest, Path("outputs/2026-01-19_143052"))
        >>> path.name
        'manifest.json'
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    return manifest_path


def load_run_manifest(path: Path) -> RunManifest:
    """Load run manifest from JSON file.

    Args:
        path: Path to manifest JSON file.

    Returns:
        RunManifest object.

    Raises:
        FileNotFoundError: If manifest file doesn't exist.
        pydantic.ValidationError: If JSON doesn't match RunManifest schema.

    Example:
        >>> manifest = load_run_manifest(Path("outputs/2026-01-19_143052/manifest.json"))
        >>> manifest.run_id
        '2026-01-19_143052'
    """
    path = Path(path)
    json_content = path.read_text(encoding="utf-8")
    return RunManifest.model_validate_json(json_content)
