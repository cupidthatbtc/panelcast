"""New coverage-targeted tests for utils/git_state.py.

Targets missed branches:
- capture_git_state: GitPython not installed (ImportError path)
- capture_git_state: repo_path not within discovered repo working tree
- capture_git_state: _PROJECT_REPO_ROOT mismatch (different repo root)
- capture_git_state: _PROJECT_REPO_ROOT is None (skip check)
- capture_git_state: detached HEAD
- capture_git_state: InvalidGitRepositoryError
- _find_repo_root: root check at filesystem root level
- GitState.short_commit with not-a-git-repo
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from panelcast.utils.git_state import (
    GitState,
    _find_repo_root,
    capture_git_state,
)


class TestFindRepoRootEdgeCases:
    """Tests for _find_repo_root edge cases."""

    def test_root_level_git(self, tmp_path):
        """Finds .git when it's in the immediate start path."""
        (tmp_path / ".git").mkdir()
        assert _find_repo_root(tmp_path) == tmp_path.resolve()

    def test_deeply_nested_finds_parent(self, tmp_path):
        """Finds .git in distant ancestor."""
        (tmp_path / ".git").mkdir()
        deep = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep.mkdir(parents=True)
        result = _find_repo_root(deep)
        assert result == tmp_path.resolve()

    def test_no_git_returns_none_in_isolated_dir(self):
        """When no .git exists in any parent, returns None (or finds a real one)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _find_repo_root(Path(tmpdir))
            # In an isolated temp dir, either None or a real repo (test runner's)
            if result is not None:
                assert (result / ".git").exists()


class TestCaptureGitStateImportError:
    """Tests for capture_git_state when GitPython is unavailable."""

    def test_gitpython_import_error_returns_placeholder(self, monkeypatch):
        """When 'git' import fails, return gitpython-not-installed placeholder."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "git":
                raise ImportError("No module named 'git'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        state = capture_git_state()

        assert state.commit == "gitpython-not-installed"
        assert state.branch == "unknown"
        assert state.dirty is False
        assert state.untracked_count == 0
        assert state.is_clean() is True


class TestCaptureGitStateRepoPathOutsideTree:
    """Tests for the repo_path-not-within-repo-root guard."""

    def test_repo_path_outside_working_dir_returns_placeholder(self, monkeypatch):
        """If discovered repo root doesn't contain repo_path, return placeholder."""
        try:
            from git import Repo
        except ImportError:
            pytest.skip("GitPython not installed")

        mock_repo = MagicMock()
        # Repo root is /project, but querying from /tmp/outside
        mock_repo.working_dir = "/project"

        monkeypatch.setattr("git.Repo", lambda *a, **kw: mock_repo)
        monkeypatch.setattr("panelcast.utils.git_state._PROJECT_REPO_ROOT", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            state = capture_git_state(tmpdir)
            assert state.commit == "not-a-git-repo"
            assert state.branch == "unknown"


class TestCaptureGitStateProjectRepoRootMismatch:
    """Tests for _PROJECT_REPO_ROOT mismatch handling."""

    def test_different_project_root_returns_placeholder(self, monkeypatch):
        """When discovered repo root != _PROJECT_REPO_ROOT, return placeholder."""
        try:
            from git import Repo
        except ImportError:
            pytest.skip("GitPython not installed")

        cwd = Path.cwd().resolve()
        mock_repo = MagicMock()
        mock_repo.working_dir = str(cwd)

        monkeypatch.setattr("git.Repo", lambda *a, **kw: mock_repo)
        # Set _PROJECT_REPO_ROOT to something different
        monkeypatch.setattr(
            "panelcast.utils.git_state._PROJECT_REPO_ROOT",
            Path("/some/completely/different/path"),
        )

        state = capture_git_state(cwd)
        assert state.commit == "not-a-git-repo"

    def test_none_project_root_allows_normal_flow(self, monkeypatch):
        """When _PROJECT_REPO_ROOT is None, skip the check and proceed normally."""
        try:
            from git import Repo
        except ImportError:
            pytest.skip("GitPython not installed")

        cwd = Path.cwd().resolve()
        mock_repo = MagicMock()
        mock_repo.working_dir = str(cwd)
        mock_repo.head.is_detached = False
        mock_repo.active_branch.name = "develop"
        mock_repo.head.commit.hexsha = "d" * 40
        mock_repo.is_dirty.return_value = False
        mock_repo.untracked_files = []

        monkeypatch.setattr("git.Repo", lambda *a, **kw: mock_repo)
        monkeypatch.setattr("panelcast.utils.git_state._PROJECT_REPO_ROOT", None)

        state = capture_git_state(cwd)
        assert state.commit == "d" * 40
        assert state.branch == "develop"


class TestCaptureGitStateDetachedHead:
    """Tests for detached HEAD detection."""

    def test_detached_head_sets_branch_to_detached(self, monkeypatch):
        """Detached HEAD returns 'DETACHED' as branch name."""
        try:
            from git import Repo
        except ImportError:
            pytest.skip("GitPython not installed")

        cwd = Path.cwd().resolve()
        mock_repo = MagicMock()
        mock_repo.working_dir = str(cwd)
        mock_repo.head.is_detached = True
        mock_repo.head.commit.hexsha = "e" * 40
        mock_repo.is_dirty.return_value = True
        mock_repo.untracked_files = ["file1.py"]

        monkeypatch.setattr("git.Repo", lambda *a, **kw: mock_repo)
        monkeypatch.setattr("panelcast.utils.git_state._PROJECT_REPO_ROOT", cwd)

        state = capture_git_state(cwd)
        assert state.branch == "DETACHED"
        assert state.commit == "e" * 40
        assert state.dirty is True
        assert state.untracked_count == 1


class TestCaptureGitStateInvalidGitRepo:
    """Tests for InvalidGitRepositoryError handling."""

    def test_invalid_git_repo_returns_placeholder(self, monkeypatch):
        """InvalidGitRepositoryError returns not-a-git-repo placeholder."""
        try:
            from git import InvalidGitRepositoryError
        except ImportError:
            pytest.skip("GitPython not installed")

        def raise_invalid(*args, **kwargs):
            raise InvalidGitRepositoryError("Not a git repo")

        monkeypatch.setattr("git.Repo", raise_invalid)

        state = capture_git_state("/tmp")
        assert state.commit == "not-a-git-repo"
        assert state.branch == "unknown"
        assert state.dirty is False
        assert state.untracked_count == 0


class TestCaptureGitStateDirtyRepo:
    """Tests for dirty repo detection."""

    def test_dirty_with_untracked(self, monkeypatch):
        """Dirty repo with untracked files reports both."""
        try:
            from git import Repo
        except ImportError:
            pytest.skip("GitPython not installed")

        cwd = Path.cwd().resolve()
        mock_repo = MagicMock()
        mock_repo.working_dir = str(cwd)
        mock_repo.head.is_detached = False
        mock_repo.active_branch.name = "feature/test"
        mock_repo.head.commit.hexsha = "f" * 40
        mock_repo.is_dirty.return_value = True
        mock_repo.untracked_files = ["a.py", "b.py", "c.py"]

        monkeypatch.setattr("git.Repo", lambda *a, **kw: mock_repo)
        monkeypatch.setattr("panelcast.utils.git_state._PROJECT_REPO_ROOT", cwd)

        state = capture_git_state(cwd)
        assert state.dirty is True
        assert state.untracked_count == 3
        assert state.is_clean() is False
        assert state.branch == "feature/test"


class TestGitStateMethodsEdgeCases:
    """Edge cases for GitState methods."""

    def test_is_clean_all_combos(self):
        """Exhaustive test of is_clean for all 4 combinations."""
        # clean
        assert GitState("a" * 40, "m", False, 0).is_clean() is True
        # dirty only
        assert GitState("a" * 40, "m", True, 0).is_clean() is False
        # untracked only
        assert GitState("a" * 40, "m", False, 1).is_clean() is False
        # both
        assert GitState("a" * 40, "m", True, 3).is_clean() is False

    def test_short_commit_not_a_git_repo_ignores_length(self):
        """short_commit returns full 'not-a-git-repo' regardless of length param."""
        state = GitState("not-a-git-repo", "unknown", False, 0)
        assert state.short_commit() == "not-a-git-repo"
        assert state.short_commit(1) == "not-a-git-repo"
        assert state.short_commit(100) == "not-a-git-repo"

    def test_short_commit_normal_sha(self):
        """short_commit truncates normal SHA correctly."""
        state = GitState("abcdef1234567890" * 2 + "12345678", "main", False, 0)
        assert state.short_commit(7) == "abcdef1"
        assert state.short_commit(10) == "abcdef1234"
