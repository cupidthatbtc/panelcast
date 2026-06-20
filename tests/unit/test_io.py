"""Unit tests for io module (readers, writers, paths).

Tests cover:
- read_csv with basic content, BOM handling, kwargs
- read_csv error handling for missing files
- write_csv with basic content, no index
- project_root path functions
"""

from pathlib import Path

import pandas as pd
import pytest

from panelcast.io.paths import project_root
from panelcast.io.readers import read_csv
from panelcast.io.writers import write_csv

# =============================================================================
# Test Class: TestReaders
# =============================================================================


class TestReaders:
    """Tests for io.readers module."""

    def test_read_csv_basic(self, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")
        df = read_csv(csv_path)
        assert list(df.columns) == ["a", "b", "c"]
        assert len(df) == 2
        assert df["a"].tolist() == [1, 4]

    def test_read_csv_with_bom(self, tmp_path: Path):
        csv_path = tmp_path / "bom.csv"
        bom_content = "\ufeffa,b,c\n1,2,3\n"
        csv_path.write_text(bom_content, encoding="utf-8")
        df = read_csv(csv_path)
        assert df.columns[0] == "a"
        assert "\ufeff" not in df.columns[0]

    def test_read_csv_with_kwargs(self, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b,c,d\n1,2,3,4\n5,6,7,8\n")
        df = read_csv(csv_path, usecols=["a", "c"])
        assert list(df.columns) == ["a", "c"]
        assert len(df) == 2

    def test_read_csv_file_not_found(self, tmp_path: Path):
        nonexistent = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            read_csv(nonexistent)

    def test_read_csv_with_different_encoding(self, tmp_path: Path):
        csv_path = tmp_path / "utf8.csv"
        csv_path.write_text("name,value\ntest,42\n", encoding="utf-8")
        df = read_csv(csv_path, encoding="utf-8")
        assert df["name"].iloc[0] == "test"

    def test_read_csv_with_nrows(self, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n")
        df = read_csv(csv_path, nrows=2)
        assert len(df) == 2

    def test_read_csv_empty_file(self, tmp_path: Path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("a,b,c\n")
        df = read_csv(csv_path)
        assert len(df) == 0
        assert list(df.columns) == ["a", "b", "c"]

    def test_read_csv_with_path_object(self, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("x,y\n1,2\n")
        df = read_csv(csv_path)
        assert len(df) == 1

    def test_read_csv_with_string_path(self, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("x,y\n1,2\n")
        df = read_csv(str(csv_path))
        assert len(df) == 1

    def test_read_csv_with_special_characters(self, tmp_path: Path):
        csv_path = tmp_path / "special.csv"
        csv_path.write_text('name,value\n"hello, world",42\n', encoding="utf-8")
        df = read_csv(csv_path, encoding="utf-8")
        assert df["name"].iloc[0] == "hello, world"

    def test_read_csv_with_skiprows(self, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n")
        df = read_csv(csv_path, skiprows=[1])
        assert len(df) == 2
        assert df.iloc[0]["a"] == 3


# =============================================================================
# Test Class: TestWriters
# =============================================================================


class TestWriters:
    """Tests for io.writers module."""

    def test_write_csv_basic(self, tmp_path: Path):
        csv_path = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        write_csv(df, str(csv_path))
        assert csv_path.exists()
        result = pd.read_csv(csv_path)
        assert list(result.columns) == ["x", "y"]
        assert result["x"].tolist() == [1, 2, 3]

    def test_write_csv_no_index(self, tmp_path: Path):
        csv_path = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]}, index=["row1", "row2"])
        write_csv(df, str(csv_path))
        result = pd.read_csv(csv_path)
        assert "Unnamed: 0" not in result.columns

    def test_write_csv_roundtrip(self, tmp_path: Path):
        csv_path = tmp_path / "roundtrip.csv"
        original = pd.DataFrame({"name": ["alice", "bob"], "score": [85.5, 92.0]})
        write_csv(original, str(csv_path))
        recovered = read_csv(csv_path)
        pd.testing.assert_frame_equal(original, recovered)

    def test_write_csv_overwrites_existing(self, tmp_path: Path):
        csv_path = tmp_path / "output.csv"
        write_csv(pd.DataFrame({"a": [1, 2]}), str(csv_path))
        write_csv(pd.DataFrame({"b": [3, 4, 5]}), str(csv_path))
        result = pd.read_csv(csv_path)
        assert list(result.columns) == ["b"]
        assert len(result) == 3

    def test_write_csv_empty_dataframe(self, tmp_path: Path):
        csv_path = tmp_path / "empty.csv"
        df = pd.DataFrame({"a": [], "b": []})
        write_csv(df, str(csv_path))
        result = pd.read_csv(csv_path)
        assert len(result) == 0
        assert list(result.columns) == ["a", "b"]

    def test_write_csv_with_path_object(self, tmp_path: Path):
        csv_path = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1]})
        write_csv(df, csv_path)
        assert csv_path.exists()

    def test_write_csv_with_nan_values(self, tmp_path: Path):
        csv_path = tmp_path / "nan.csv"
        df = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
        write_csv(df, str(csv_path))
        result = pd.read_csv(csv_path)
        assert pd.isna(result["a"].iloc[1])

    def test_write_csv_with_string_data(self, tmp_path: Path):
        csv_path = tmp_path / "strings.csv"
        df = pd.DataFrame({"name": ["hello", "world"], "id": [1, 2]})
        write_csv(df, str(csv_path))
        result = pd.read_csv(csv_path)
        assert result["name"].tolist() == ["hello", "world"]


# =============================================================================
# Test Class: TestPaths
# =============================================================================


class TestPaths:
    """Tests for io.paths module."""

    def test_project_root_is_directory(self):
        root = project_root()
        assert isinstance(root, Path)
        assert root.is_dir()

    def test_project_root_contains_expected_files(self):
        root = project_root()
        has_pyproject = (root / "pyproject.toml").exists()
        has_src = (root / "src").is_dir()
        assert has_pyproject or has_src

    def test_project_root_consistent(self):
        root1 = project_root()
        root2 = project_root()
        assert root1 == root2

    def test_project_root_is_absolute(self):
        root = project_root()
        assert root.is_absolute()

    def test_project_root_contains_panelcast_source(self):
        root = project_root()
        panelcast_path = root / "src" / "panelcast"
        assert panelcast_path.is_dir()
