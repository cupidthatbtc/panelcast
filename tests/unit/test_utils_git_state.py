"""Tests for git state capture utility."""

import tempfile
from pathlib import Path

import pytest

from panelcast.utils.git_state import GitState, capture_git_state


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
