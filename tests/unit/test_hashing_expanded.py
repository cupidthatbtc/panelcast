"""Expanded tests for utils/hashing.py: hash_dataframe, sha256_file, sha256_directory, sha256_path."""

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
