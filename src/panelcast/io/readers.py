"""Input readers for data loading."""

from pathlib import Path

import pandas as pd


def read_csv(path: str | Path, encoding: str = "utf-8-sig", **kwargs) -> pd.DataFrame:
    """
    Read CSV with robust encoding handling.

    Args:
        path: Path to CSV file
        encoding: Encoding to use (default utf-8-sig handles BOM)
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        DataFrame with data from CSV
    """
    return pd.read_csv(path, encoding=encoding, **kwargs)
