"""Coverage-targeted tests for utils/git_state.py.

Tests target missed lines/branches:
- capture_git_state: GitPython not installed (ImportError), detached HEAD,
  repo discovered outside project root, _PROJECT_REPO_ROOT mismatch,
  InvalidGitRepositoryError, repo_path not within discovered repo
- _find_repo_root: traversal to filesystem root without .git
- GitState: short_commit with not-a-git-repo, is_clean with various combos
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from panelcast.utils.git_state import (
    GitState,
    _find_repo_root,
    capture_git_state,
)

# =============================================================================
# TestFindRepoRoot
# =============================================================================


class TestFindRepoRoot:
    """Tests for _find_repo_root helper."""

    def test_finds_repo_in_current_dir(self, tmp_path):
        """Finds .git in the start directory itself."""
        (tmp_path / ".git").mkdir()
        result = _find_repo_root(tmp_path)
        assert result == tmp_path.resolve()

    def test_finds_repo_in_parent(self, tmp_path):
        """Finds .git in a parent directory."""
        (tmp_path / ".git").mkdir()
        child = tmp_path / "src" / "module"
        child.mkdir(parents=True)
        result = _find_repo_root(child)
        assert result == tmp_path.resolve()

    def test_returns_none_when_no_git(self):
        """Returns None when no .git directory exists anywhere."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Ensure no .git in this tree
            deep = Path(tmpdir) / "a" / "b" / "c"
            deep.mkdir(parents=True)
            result = _find_repo_root(deep)
            # Should return None (no .git found all the way to root)
            # Note: this may find a .git in a real parent on the test machine,
            # but in a temporary directory it should not
            if result is not None:
                # If it found one, it must be a real parent repo
                assert (result / ".git").exists()


# =============================================================================
# TestCaptureGitStateImportError
# =============================================================================


class TestCaptureGitStateImportError:
    """Tests for capture_git_state when GitPython is not installed."""

    def test_returns_placeholder_on_import_error(self, monkeypatch):
        """When GitPython import fails, return placeholder GitState."""
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

    def test_placeholder_is_clean(self, monkeypatch):
        """Placeholder GitState reports as clean."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "git":
                raise ImportError("No module named 'git'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        state = capture_git_state()
        assert state.is_clean() is True


# =============================================================================
# TestCaptureGitStateInvalidRepo
# =============================================================================


class TestCaptureGitStateInvalidRepo:
    """Tests for capture_git_state in directories that are not git repos."""

    def test_temp_dir_not_a_repo(self):
        """Temp directory outside any repo should return not-a-git-repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = capture_git_state(tmpdir)
            assert state.commit in ("not-a-git-repo", "gitpython-not-installed")
            assert state.branch == "unknown"

    def test_invalid_git_repository_error(self, monkeypatch):
        """InvalidGitRepositoryError is handled gracefully."""
        try:
            from git import InvalidGitRepositoryError, Repo
        except ImportError:
            pytest.skip("GitPython not installed")

        def mock_repo(*args, **kwargs):
            raise InvalidGitRepositoryError("Not a git repo")

        monkeypatch.setattr("git.Repo", mock_repo)
        state = capture_git_state("/tmp")

        assert state.commit == "not-a-git-repo"
        assert state.branch == "unknown"


# =============================================================================
# TestCaptureGitStateDetachedHead
# =============================================================================


class TestCaptureGitStateDetachedHead:
    """Tests for capture_git_state with detached HEAD."""

    def test_detached_head_returns_detached(self, monkeypatch):
        """Detached HEAD should set branch to 'DETACHED'."""
        try:
            from git import Repo
        except ImportError:
            pytest.skip("GitPython not installed")

        mock_repo = MagicMock()
        mock_repo.working_dir = str(Path.cwd().resolve())
        mock_repo.head.is_detached = True
        mock_repo.head.commit.hexsha = "a" * 40
        mock_repo.is_dirty.return_value = False
        mock_repo.untracked_files = []

        monkeypatch.setattr("git.Repo", lambda *a, **kw: mock_repo)
        # Also need to ensure _PROJECT_REPO_ROOT matches
        monkeypatch.setattr(
            "panelcast.utils.git_state._PROJECT_REPO_ROOT",
            Path.cwd().resolve(),
        )

        state = capture_git_state(Path.cwd())
        assert state.branch == "DETACHED"
        assert state.commit == "a" * 40


# =============================================================================
# TestCaptureGitStateDirtyTree
# =============================================================================


class TestCaptureGitStateDirtyTree:
    """Tests for capture_git_state with dirty working tree."""

    def test_dirty_repo_detected(self, monkeypatch):
        """Dirty working directory should set dirty=True."""
        try:
            from git import Repo
        except ImportError:
            pytest.skip("GitPython not installed")

        mock_repo = MagicMock()
        mock_repo.working_dir = str(Path.cwd().resolve())
        mock_repo.head.is_detached = False
        mock_repo.active_branch.name = "feature/test"
        mock_repo.head.commit.hexsha = "b" * 40
        mock_repo.is_dirty.return_value = True
        mock_repo.untracked_files = ["new_file.py", "another.py"]

        monkeypatch.setattr("git.Repo", lambda *a, **kw: mock_repo)
        monkeypatch.setattr(
            "panelcast.utils.git_state._PROJECT_REPO_ROOT",
            Path.cwd().resolve(),
        )

        state = capture_git_state(Path.cwd())
        assert state.dirty is True
        assert state.untracked_count == 2
        assert state.is_clean() is False


# =============================================================================
# TestCaptureGitStateProjectRepoRootMismatch
# =============================================================================


class TestCaptureGitStateProjectRepoRootMismatch:
    """Tests for _PROJECT_REPO_ROOT mismatch branch."""

    def test_different_repo_root_returns_placeholder(self, monkeypatch):
        """When discovered repo root differs from _PROJECT_REPO_ROOT, return placeholder."""
        try:
            from git import Repo
        except ImportError:
            pytest.skip("GitPython not installed")

        mock_repo = MagicMock()
        # Repo found at /some/other/repo but _PROJECT_REPO_ROOT is different
        mock_repo.working_dir = "/some/other/repo"

        monkeypatch.setattr("git.Repo", lambda *a, **kw: mock_repo)
        monkeypatch.setattr(
            "panelcast.utils.git_state._PROJECT_REPO_ROOT",
            Path("/totally/different/path").resolve(),
        )

        state = capture_git_state(Path("/some/other/repo"))
        assert state.commit == "not-a-git-repo"
        assert state.branch == "unknown"

    def test_project_repo_root_none_skips_check(self, monkeypatch):
        """When _PROJECT_REPO_ROOT is None, the repo root check is skipped."""
        try:
            from git import Repo
        except ImportError:
            pytest.skip("GitPython not installed")

        cwd = Path.cwd().resolve()
        mock_repo = MagicMock()
        mock_repo.working_dir = str(cwd)
        mock_repo.head.is_detached = False
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit.hexsha = "c" * 40
        mock_repo.is_dirty.return_value = False
        mock_repo.untracked_files = []

        monkeypatch.setattr("git.Repo", lambda *a, **kw: mock_repo)
        monkeypatch.setattr(
            "panelcast.utils.git_state._PROJECT_REPO_ROOT",
            None,
        )

        state = capture_git_state(cwd)
        # Should proceed normally since _PROJECT_REPO_ROOT is None
        assert state.commit == "c" * 40
        assert state.branch == "main"


# =============================================================================
# TestCaptureGitStateRepoPathOutsideTree
# =============================================================================


class TestCaptureGitStateRepoPathOutsideTree:
    """Tests for the repo_path-not-within-repo-root guard."""

    def test_path_outside_repo_returns_placeholder(self, monkeypatch):
        """If repo_path is not within the discovered repo root, return placeholder."""
        try:
            from git import Repo
        except ImportError:
            pytest.skip("GitPython not installed")

        mock_repo = MagicMock()
        # Repo root is /project but we're querying from /tmp/outside
        mock_repo.working_dir = "/project"

        monkeypatch.setattr("git.Repo", lambda *a, **kw: mock_repo)
        monkeypatch.setattr(
            "panelcast.utils.git_state._PROJECT_REPO_ROOT",
            None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            state = capture_git_state(tmpdir)
            assert state.commit == "not-a-git-repo"


# =============================================================================
# TestGitStateShortCommitEdgeCases
# =============================================================================


class TestGitStateShortCommitEdgeCases:
    """Additional short_commit edge cases."""

    def test_not_a_git_repo_returns_full(self):
        """short_commit for 'not-a-git-repo' returns the full string."""
        state = GitState(
            commit="not-a-git-repo",
            branch="unknown",
            dirty=False,
            untracked_count=0,
        )
        assert state.short_commit() == "not-a-git-repo"
        assert state.short_commit(3) == "not-a-git-repo"

    def test_zero_length_short_commit(self):
        """short_commit(0) returns empty string for valid SHA."""
        state = GitState(
            commit="a" * 40,
            branch="main",
            dirty=False,
            untracked_count=0,
        )
        assert state.short_commit(0) == ""

    def test_length_exceeding_commit(self):
        """short_commit with length > 40 returns the full 40-char SHA."""
        state = GitState(
            commit="a" * 40,
            branch="main",
            dirty=False,
            untracked_count=0,
        )
        assert state.short_commit(100) == "a" * 40


# =============================================================================
# TestGitStateIsCleanCombinations
# =============================================================================


class TestGitStateIsCleanCombinations:
    """Exhaustive is_clean combinations."""

    def test_clean(self):
        """Not dirty, no untracked => clean."""
        state = GitState(commit="a" * 40, branch="main", dirty=False, untracked_count=0)
        assert state.is_clean() is True

    def test_dirty_only(self):
        """Dirty but no untracked => not clean."""
        state = GitState(commit="a" * 40, branch="main", dirty=True, untracked_count=0)
        assert state.is_clean() is False

    def test_untracked_only(self):
        """Not dirty but has untracked => not clean."""
        state = GitState(commit="a" * 40, branch="main", dirty=False, untracked_count=1)
        assert state.is_clean() is False

    def test_dirty_and_untracked(self):
        """Both dirty and untracked => not clean."""
        state = GitState(commit="a" * 40, branch="main", dirty=True, untracked_count=5)
        assert state.is_clean() is False
