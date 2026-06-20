"""Hashing utilities for reproducibility verification."""

import hashlib
from pathlib import Path

import pandas as pd


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Generate deterministic SHA256 hash of DataFrame contents.

    Uses pandas internal hashing (handles NaN, dtypes consistently)
    then combines into single digest.

    Args:
        df: DataFrame to hash

    Returns:
        Hexadecimal SHA256 digest (64 characters)

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3]})
        >>> h = hash_dataframe(df)
        >>> len(h)
        64
    """
    # Use a stable text serialization rather than pandas internal hashing,
    # which can change across pandas versions.
    normalized = df.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    csv_bytes = normalized.to_csv(
        index=True,
        sep="\t",
        na_rep="<NA>",
        lineterminator="\n",
        float_format="%.17g",
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    ).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def sha256_file(path: Path | str, block_size: int = 65536) -> str:
    """
    Compute SHA256 hash of file in memory-efficient chunks.

    Args:
        path: Path to the file to hash
        block_size: Size of chunks to read (default 64KB for memory efficiency)

    Returns:
        Hexadecimal SHA256 digest (64 characters)

    Example:
        >>> h = sha256_file("data/raw/all_albums_full.csv")
        >>> len(h)
        64
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha256.update(block)
    return sha256.hexdigest()


def sha256_directory(path: Path | str) -> str:
    """Compute deterministic SHA256 hash for a directory tree.

    The hash includes relative file paths and each file's SHA256 digest.
    """
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    digest = hashlib.sha256()
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_path(path: Path | str) -> str:
    """Compute SHA256 hash for either a file or directory."""
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    if p.is_dir():
        return sha256_directory(p)
    raise FileNotFoundError(f"Path does not exist: {p}")
