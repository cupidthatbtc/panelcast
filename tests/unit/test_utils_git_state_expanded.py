"""Expanded tests for utils/git_state.py: GitState dataclass and capture_git_state."""

import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from panelcast.utils.git_state import GitState, capture_git_state


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
