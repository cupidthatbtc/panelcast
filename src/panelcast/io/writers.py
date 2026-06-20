"""Output writers."""

from pathlib import Path

import pandas as pd


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False)
