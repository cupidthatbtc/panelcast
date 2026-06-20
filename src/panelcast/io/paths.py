"""Path helpers for project directory resolution."""

from pathlib import Path


def project_root() -> Path:
    """Find project root using marker-based detection.

    Searches parent directories for pyproject.toml to reliably locate
    the project root regardless of where the code is invoked from.

    Returns
    -------
    Path
        Absolute path to project root directory.

    Raises
    ------
    FileNotFoundError
        If pyproject.toml is not found in any parent directory.
    """
    marker = "pyproject.toml"
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root not found - no {marker} in parent directories of {path}")
