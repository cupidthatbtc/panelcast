"""Git repository state capture for reproducibility tracking.

This module provides utilities to capture the current git repository state
including commit hash, branch name, dirty status, and untracked file count.
This information is essential for run manifests and reproducibility verification.
"""

from dataclasses import dataclass
from pathlib import Path


def _find_repo_root(start_path: Path) -> Path | None:
    """Find nearest ancestor containing .git, or None if absent."""
    current = start_path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    if (current / ".git").exists():
        return current
    return None


_PROJECT_REPO_ROOT = _find_repo_root(Path(__file__))


@dataclass(frozen=True)
class GitState:
    """Immutable record of git repository state at a point in time.

    Attributes:
        commit: Full SHA-1 hash of the current commit (40 characters).
            Returns "not-a-git-repo" if not in a git repository.
        branch: Active branch name. Returns "DETACHED" if HEAD is detached,
            or "unknown" if not in a git repository.
        dirty: True if working directory has uncommitted changes
            (staged or unstaged modifications to tracked files).
        untracked_count: Number of untracked files in the repository.

    Example:
        >>> state = capture_git_state()
        >>> len(state.commit) == 40 or state.commit == "not-a-git-repo"
        True
    """

    commit: str
    branch: str
    dirty: bool
    untracked_count: int

    def is_clean(self) -> bool:
        """Check if repository is in a clean state (no dirty files, no untracked)."""
        return not self.dirty and self.untracked_count == 0

    def short_commit(self, length: int = 7) -> str:
        """Return abbreviated commit hash.

        Args:
            length: Number of characters for short hash (default 7).

        Returns:
            Abbreviated commit hash, or full string if not a valid SHA.
        """
        if self.commit == "not-a-git-repo":
            return self.commit
        return self.commit[:length]


def capture_git_state(repo_path: Path | str = Path(".")) -> GitState:
    """Capture current git repository state.

    Searches for a git repository starting from repo_path and looking
    in parent directories. Returns a placeholder GitState if not in
    a git repository.

    Args:
        repo_path: Path to start searching for git repository.
            Defaults to current directory.

    Returns:
        GitState with current repository information, or placeholder
        values if not in a git repository.

    Example:
        >>> state = capture_git_state()
        >>> isinstance(state, GitState)
        True
        >>> # In a real git repo:
        >>> # state.branch in ["main", "develop", "DETACHED", ...]
    """
    # Import here to avoid import errors if git is not installed
    try:
        from git import InvalidGitRepositoryError, Repo
    except ImportError:
        # GitPython not installed - return placeholder
        return GitState(
            commit="gitpython-not-installed",
            branch="unknown",
            dirty=False,
            untracked_count=0,
        )

    repo_path = Path(repo_path).resolve()

    try:
        # Search parent directories for .git folder
        repo = Repo(repo_path, search_parent_directories=True)

        # Verify repo_path is actually within the discovered repo's working tree.
        # Without this check, a temp directory outside any repo would traverse up
        # and find an unrelated parent repo (e.g., the project running the tests).
        repo_root = Path(repo.working_dir).resolve()
        if not (repo_path == repo_root or repo_root in repo_path.parents):
            return GitState(
                commit="not-a-git-repo",
                branch="unknown",
                dirty=False,
                untracked_count=0,
            )

        # Ensure we only capture state for this project's repository, not
        # unrelated parent repositories on the filesystem.
        if _PROJECT_REPO_ROOT is not None and repo_root != _PROJECT_REPO_ROOT:
            return GitState(
                commit="not-a-git-repo",
                branch="unknown",
                dirty=False,
                untracked_count=0,
            )

        # Get commit hash
        commit = repo.head.commit.hexsha

        # Get branch name (handle detached HEAD)
        if repo.head.is_detached:
            branch = "DETACHED"
        else:
            branch = repo.active_branch.name

        # Get dirty status (staged + unstaged modifications to tracked files)
        dirty = repo.is_dirty()

        # Count untracked files
        untracked_count = len(repo.untracked_files)

        return GitState(
            commit=commit,
            branch=branch,
            dirty=dirty,
            untracked_count=untracked_count,
        )

    except InvalidGitRepositoryError:
        # Not in a git repository
        return GitState(
            commit="not-a-git-repo",
            branch="unknown",
            dirty=False,
            untracked_count=0,
        )
