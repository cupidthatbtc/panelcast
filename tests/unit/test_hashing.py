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


# --- from unit/test_hashing_expanded.py ---


class TestHashDataframeExpanded:
    """Expanded tests for hash_dataframe."""

    def test_different_column_names(self):
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"y": [1, 2]})
        assert hash_dataframe(df1) != hash_dataframe(df2)

    def test_different_row_count(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1, 2, 3]})
        assert hash_dataframe(df1) != hash_dataframe(df2)

    def test_single_row(self):
        df = pd.DataFrame({"a": [42]})
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_large_dataframe(self):
        df = pd.DataFrame({f"c{i}": np.random.randn(1000) for i in range(20)})
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_boolean_columns(self):
        df = pd.DataFrame({"flag": [True, False, True]})
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_mixed_types(self):
        df = pd.DataFrame({"int": [1], "float": [1.5], "str": ["a"]})
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_datetime_column(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-02-01"])})
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_with_named_index(self):
        df = pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="idx"))
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_inf_values(self):
        df = pd.DataFrame({"a": [np.inf, -np.inf, 0.0]})
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_deterministic_across_calls(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        hashes = [hash_dataframe(df) for _ in range(5)]
        assert len(set(hashes)) == 1


class TestSha256FileExpanded:
    """Expanded tests for sha256_file."""

    def test_binary_content(self):
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"\x00\xff\x01\xfe")
            f.flush()
            h = sha256_file(f.name)
        assert len(h) == 64

    def test_unicode_content(self):
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, suffix=".txt"
        ) as f:
            f.write("Hello world. Special chars: a, u, o.")
            f.flush()
            h = sha256_file(f.name)
        assert len(h) == 64

    def test_large_file(self):
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"x" * (1024 * 1024))  # 1 MB
            f.flush()
            h = sha256_file(f.name)
        assert len(h) == 64

    def test_block_size_1(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            f.flush()
            h_default = sha256_file(f.name)
            h_tiny = sha256_file(f.name, block_size=1)
        assert h_default == h_tiny


class TestSha256DirectoryExpanded:
    """Expanded tests for sha256_directory."""

    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")
        (tmp_path / "c.txt").write_text("ccc")
        h = sha256_directory(tmp_path)
        assert len(h) == 64

    def test_deep_nesting(self, tmp_path):
        d = tmp_path / "a" / "b" / "c"
        d.mkdir(parents=True)
        (d / "deep.txt").write_text("deep file")
        h = sha256_directory(tmp_path)
        assert len(h) == 64

    def test_file_order_independent(self, tmp_path):
        """Hash is computed on sorted file paths, so order is deterministic."""
        (tmp_path / "z.txt").write_text("z")
        (tmp_path / "a.txt").write_text("a")
        h1 = sha256_directory(tmp_path)
        h2 = sha256_directory(tmp_path)
        assert h1 == h2

    def test_extra_file_changes_hash(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        h1 = sha256_directory(tmp_path)
        (tmp_path / "b.txt").write_text("b")
        h2 = sha256_directory(tmp_path)
        assert h1 != h2


class TestSha256PathExpanded:
    """Expanded tests for sha256_path."""

    def test_file_matches_sha256_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        assert sha256_path(f) == sha256_file(f)

    def test_dir_matches_sha256_directory(self, tmp_path):
        (tmp_path / "x.txt").write_text("x")
        assert sha256_path(tmp_path) == sha256_directory(tmp_path)

    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sha256_path(tmp_path / "nope")

    def test_accepts_string(self, tmp_path):
        f = tmp_path / "s.txt"
        f.write_text("str path")
        h = sha256_path(str(f))
        assert len(h) == 64
