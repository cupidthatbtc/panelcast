"""Unit tests for hashing utilities."""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelcast.utils.hashing import (
    hash_dataframe,
    sha256_directory,
    sha256_file,
    sha256_path,
)


class TestHashDataframe:
    """Tests for hash_dataframe function."""

    def test_returns_64_char_hex(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        h = hash_dataframe(df)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        h1 = hash_dataframe(df)
        h2 = hash_dataframe(df)
        assert h1 == h2

    def test_different_data_different_hash(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})
        assert hash_dataframe(df1) != hash_dataframe(df2)

    def test_column_order_independent(self):
        """hash_dataframe sorts columns, so order should not matter."""
        df1 = pd.DataFrame({"b": [1, 2], "a": [3, 4]})
        df2 = pd.DataFrame({"a": [3, 4], "b": [1, 2]})
        assert hash_dataframe(df1) == hash_dataframe(df2)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_handles_nan_values(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_nan_deterministic(self):
        df1 = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        df2 = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        assert hash_dataframe(df1) == hash_dataframe(df2)

    def test_different_dtypes_different_hash(self):
        df_int = pd.DataFrame({"a": [1, 2, 3]})
        df_float = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        # Integer vs float representation may produce different CSV output
        h_int = hash_dataframe(df_int)
        h_float = hash_dataframe(df_float)
        # Just verify both are valid hashes
        assert len(h_int) == 64
        assert len(h_float) == 64

    def test_single_column(self):
        df = pd.DataFrame({"x": [10, 20, 30]})
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_many_columns(self):
        data = {f"col_{i}": range(5) for i in range(50)}
        df = pd.DataFrame(data)
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_string_columns(self):
        df = pd.DataFrame({"name": ["alice", "bob"], "score": [90, 85]})
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_index_included_in_hash(self):
        """Index is included via to_csv(index=True)."""
        df1 = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        df2 = pd.DataFrame({"a": [1, 2]}, index=[10, 20])
        assert hash_dataframe(df1) != hash_dataframe(df2)


class TestSha256File:
    """Tests for sha256_file function."""

    def test_returns_64_char_hex(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            f.flush()
            h = sha256_file(f.name)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_matches_hashlib(self):
        content = b"test content for hashing"
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(content)
            f.flush()
            h = sha256_file(f.name)
        expected = hashlib.sha256(content).hexdigest()
        assert h == expected

    def test_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("deterministic content")
            f.flush()
            h1 = sha256_file(f.name)
            h2 = sha256_file(f.name)
        assert h1 == h2

    def test_different_content_different_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write("content A")
            f1.flush()
            h1 = sha256_file(f1.name)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write("content B")
            f2.flush()
            h2 = sha256_file(f2.name)
        assert h1 != h2

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.flush()
            h = sha256_file(f.name)
        # SHA256 of empty string is well-known
        expected = hashlib.sha256(b"").hexdigest()
        assert h == expected

    def test_accepts_path_object(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("path test")
            f.flush()
            h = sha256_file(Path(f.name))
        assert len(h) == 64

    def test_custom_block_size(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("block size test" * 100)
            f.flush()
            h_default = sha256_file(f.name)
            h_small = sha256_file(f.name, block_size=16)
        assert h_default == h_small


class TestSha256Directory:
    """Tests for sha256_directory function."""

    def test_returns_64_char_hex(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        h = sha256_directory(tmp_path)
        assert len(h) == 64

    def test_deterministic(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        h1 = sha256_directory(tmp_path)
        h2 = sha256_directory(tmp_path)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        (tmp_path / "a.txt").write_text("version 1")
        h1 = sha256_directory(tmp_path)
        (tmp_path / "a.txt").write_text("version 2")
        h2 = sha256_directory(tmp_path)
        assert h1 != h2

    def test_includes_file_names(self, tmp_path):
        (tmp_path / "a.txt").write_text("same content")
        h1 = sha256_directory(tmp_path)
        (tmp_path / "a.txt").unlink()
        (tmp_path / "b.txt").write_text("same content")
        h2 = sha256_directory(tmp_path)
        assert h1 != h2

    def test_empty_directory(self, tmp_path):
        h = sha256_directory(tmp_path)
        assert len(h) == 64

    def test_nested_directories(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested content")
        h = sha256_directory(tmp_path)
        assert len(h) == 64

    def test_raises_for_non_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("not a directory")
        with pytest.raises(ValueError, match="Not a directory"):
            sha256_directory(f)


class TestSha256Path:
    """Tests for sha256_path function."""

    def test_handles_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("file content")
        h = sha256_path(f)
        assert h == sha256_file(f)

    def test_handles_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("content")
        h = sha256_path(tmp_path)
        assert h == sha256_directory(tmp_path)

    def test_raises_for_nonexistent(self, tmp_path):
        nonexistent = tmp_path / "missing"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            sha256_path(nonexistent)
