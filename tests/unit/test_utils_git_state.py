"""Tests for git state capture utility."""

from __future__ import annotations

import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from panelcast.utils.git_state import (
    GitState,
    _find_repo_root,
    capture_git_state,
)


class TestGitStateDataclass:
    """Tests for GitState dataclass creation and methods."""

    def test_create_git_state(self):
        """GitState can be created with all required fields."""
        state = GitState(
            commit="abc123def456789012345678901234567890abcd",
            branch="main",
            dirty=False,
            untracked_count=0,
        )
        assert state.commit == "abc123def456789012345678901234567890abcd"
        assert state.branch == "main"
        assert state.dirty is False
        assert state.untracked_count == 0

    def test_git_state_is_frozen(self):
        """GitState is immutable (frozen dataclass)."""
        state = GitState(
            commit="abc123def456789012345678901234567890abcd",
            branch="main",
            dirty=False,
            untracked_count=0,
        )
        with pytest.raises(AttributeError):
            state.commit = "changed"

    def test_is_clean_when_clean(self):
        """is_clean returns True when no dirty files and no untracked."""
        state = GitState(
            commit="abc123def456789012345678901234567890abcd",
            branch="main",
            dirty=False,
            untracked_count=0,
        )
        assert state.is_clean() is True

    def test_is_clean_when_dirty(self):
        """is_clean returns False when dirty=True."""
        state = GitState(
            commit="abc123def456789012345678901234567890abcd",
            branch="main",
            dirty=True,
            untracked_count=0,
        )
        assert state.is_clean() is False

    def test_is_clean_when_untracked(self):
        """is_clean returns False when untracked_count > 0."""
        state = GitState(
            commit="abc123def456789012345678901234567890abcd",
            branch="main",
            dirty=False,
            untracked_count=5,
        )
        assert state.is_clean() is False

    def test_short_commit_default_length(self):
        """short_commit returns 7-character abbreviation by default."""
        state = GitState(
            commit="abc123def456789012345678901234567890abcd",
            branch="main",
            dirty=False,
            untracked_count=0,
        )
        assert state.short_commit() == "abc123d"
        assert len(state.short_commit()) == 7

    def test_short_commit_custom_length(self):
        """short_commit supports custom length parameter."""
        state = GitState(
            commit="abc123def456789012345678901234567890abcd",
            branch="main",
            dirty=False,
            untracked_count=0,
        )
        assert state.short_commit(10) == "abc123def4"

    def test_short_commit_not_a_repo(self):
        """short_commit returns full string for non-repo placeholder."""
        state = GitState(
            commit="not-a-git-repo",
            branch="unknown",
            dirty=False,
            untracked_count=0,
        )
        assert state.short_commit() == "not-a-git-repo"


class TestCaptureGitState:
    """Tests for capture_git_state function."""

    def test_capture_in_real_repo(self):
        """capture_git_state returns real commit hash in git repo."""
        # The project itself is a git repo
        state = capture_git_state()

        # Should return a GitState
        assert isinstance(state, GitState)

        # Should have a valid 40-character SHA-1 commit hash
        # (unless we're not in a git repo or GitPython is unavailable)
        if state.commit not in ("not-a-git-repo", "gitpython-not-installed"):
            assert len(state.commit) == 40
            # All characters should be hex digits
            assert all(c in "0123456789abcdef" for c in state.commit)

    def test_capture_returns_branch_name(self):
        """capture_git_state returns a branch name."""
        state = capture_git_state()

        # Should have a branch name (or DETACHED for detached HEAD)
        assert isinstance(state.branch, str)
        assert len(state.branch) > 0

    def test_capture_returns_dirty_status(self):
        """capture_git_state returns dirty boolean."""
        state = capture_git_state()
        assert isinstance(state.dirty, bool)

    def test_capture_returns_untracked_count(self):
        """capture_git_state returns untracked count as int."""
        state = capture_git_state()
        assert isinstance(state.untracked_count, int)
        assert state.untracked_count >= 0

    def test_capture_not_in_git_repo(self):
        """capture_git_state returns placeholder when not in git repo."""
        # Create a temporary directory that is NOT a git repo
        with tempfile.TemporaryDirectory() as tmpdir:
            state = capture_git_state(tmpdir)

            # Should return placeholder GitState
            assert isinstance(state, GitState)
            assert state.commit in ("not-a-git-repo", "gitpython-not-installed")
            assert state.branch == "unknown"
            assert state.dirty is False
            assert state.untracked_count == 0

    def test_capture_accepts_path_string(self):
        """capture_git_state accepts string path."""
        state = capture_git_state(".")
        assert isinstance(state, GitState)

    def test_capture_accepts_path_object(self):
        """capture_git_state accepts Path object."""
        state = capture_git_state(Path("."))
        assert isinstance(state, GitState)

    def test_return_type_always_git_state(self):
        """capture_git_state always returns GitState, never raises."""
        # Test with valid path
        state1 = capture_git_state()
        assert isinstance(state1, GitState)

        # Test with non-repo path
        with tempfile.TemporaryDirectory() as tmpdir:
            state2 = capture_git_state(tmpdir)
            assert isinstance(state2, GitState)

    def test_capture_searches_parent_directories(self):
        """capture_git_state finds repo when called from subdirectory."""
        # Create a subdirectory in the project
        sub_path = Path("src/panelcast/utils")
        state = capture_git_state(sub_path)

        # Should still find the git repo in parent
        if state.commit not in ("not-a-git-repo", "gitpython-not-installed"):
            assert len(state.commit) == 40


class TestGitStateEquality:
    """Tests for GitState equality and hashing."""

    def test_equal_states(self):
        """GitState instances with same values are equal."""
        state1 = GitState(
            commit="abc123def456789012345678901234567890abcd",
            branch="main",
            dirty=False,
            untracked_count=0,
        )
        state2 = GitState(
            commit="abc123def456789012345678901234567890abcd",
            branch="main",
            dirty=False,
            untracked_count=0,
        )
        assert state1 == state2

    def test_unequal_commit(self):
        """GitState instances with different commits are not equal."""
        state1 = GitState(
            commit="abc123def456789012345678901234567890abcd",
            branch="main",
            dirty=False,
            untracked_count=0,
        )
        state2 = GitState(
            commit="def456abc789012345678901234567890abcd123",
            branch="main",
            dirty=False,
            untracked_count=0,
        )
        assert state1 != state2

    def test_hashable(self):
        """GitState can be used in sets and as dict keys."""
        state = GitState(
            commit="abc123def456789012345678901234567890abcd",
            branch="main",
            dirty=False,
            untracked_count=0,
        )
        # Should be hashable
        hash(state)

        # Should work in a set
        state_set = {state}
        assert len(state_set) == 1


# --- from unit/test_utils_git_state_coverage.py ---


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


# --- from unit/test_utils_git_state_expanded.py ---


class TestGitStateExpanded:
    """Expanded tests for GitState dataclass."""

    def test_frozen(self):
        state = GitState(commit="a" * 40, branch="main", dirty=False, untracked_count=0)
        with pytest.raises(FrozenInstanceError):
            state.branch = "dev"

    def test_is_clean_dirty_and_untracked(self):
        state = GitState(commit="a" * 40, branch="main", dirty=True, untracked_count=3)
        assert state.is_clean() is False

    def test_short_commit_full_length(self):
        state = GitState(commit="a" * 40, branch="main", dirty=False, untracked_count=0)
        assert state.short_commit(40) == "a" * 40

    def test_short_commit_one_char(self):
        state = GitState(commit="abc123" + "0" * 34, branch="main", dirty=False, untracked_count=0)
        assert state.short_commit(1) == "a"

    def test_short_commit_gitpython_not_installed(self):
        state = GitState(
            commit="gitpython-not-installed",
            branch="unknown",
            dirty=False,
            untracked_count=0,
        )
        # not-a-git-repo check is explicit, but gitpython-not-installed is just truncated
        assert state.short_commit() == "gitpyth"

    def test_equality(self):
        a = GitState(commit="a" * 40, branch="main", dirty=False, untracked_count=0)
        b = GitState(commit="a" * 40, branch="main", dirty=False, untracked_count=0)
        assert a == b

    def test_inequality_branch(self):
        a = GitState(commit="a" * 40, branch="main", dirty=False, untracked_count=0)
        b = GitState(commit="a" * 40, branch="dev", dirty=False, untracked_count=0)
        assert a != b

    def test_inequality_dirty(self):
        a = GitState(commit="a" * 40, branch="main", dirty=False, untracked_count=0)
        b = GitState(commit="a" * 40, branch="main", dirty=True, untracked_count=0)
        assert a != b

    def test_hashable_in_set(self):
        state = GitState(commit="a" * 40, branch="main", dirty=False, untracked_count=0)
        s = {state, state}
        assert len(s) == 1

    def test_repr(self):
        state = GitState(commit="a" * 40, branch="main", dirty=False, untracked_count=0)
        r = repr(state)
        assert "GitState" in r
        assert "main" in r

    def test_detached_branch(self):
        state = GitState(commit="a" * 40, branch="DETACHED", dirty=False, untracked_count=0)
        assert state.branch == "DETACHED"

    def test_untracked_count_large(self):
        state = GitState(commit="a" * 40, branch="main", dirty=False, untracked_count=1000)
        assert state.untracked_count == 1000
        assert state.is_clean() is False


class TestCaptureGitStateExpanded:
    """Expanded tests for capture_git_state."""

    def test_nonexistent_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "does_not_exist"
            nonexistent.mkdir()
            state = capture_git_state(nonexistent)
            assert isinstance(state, GitState)

    def test_nested_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "c"
            nested.mkdir(parents=True)
            state = capture_git_state(nested)
            assert isinstance(state, GitState)

    def test_returns_git_state_always(self):
        """capture_git_state should never raise."""
        state = capture_git_state("/")
        assert isinstance(state, GitState)

    def test_project_root(self):
        """Capture from the project root should return valid state."""
        state = capture_git_state(Path(__file__).parent.parent.parent.parent)
        assert isinstance(state, GitState)


# --- from unit/test_utils_git_state_new.py ---


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


class TestCaptureGitStateImportError_new:
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


class TestCaptureGitStateRepoPathOutsideTree_new:
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


class TestCaptureGitStateProjectRepoRootMismatch_new:
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


class TestCaptureGitStateDetachedHead_new:
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
