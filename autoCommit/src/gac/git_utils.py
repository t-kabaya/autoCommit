"""Git utilities for gac."""

import subprocess
from pathlib import Path
from typing import List, Optional


class GitError(Exception):
    """Exception raised for git-related errors."""

    pass


def _run_git_command(args: List[str], cwd: Optional[Path] = None) -> str:
    """Run a git command and return output.

    Args:
        args: Git command arguments (without 'git' prefix)
        cwd: Working directory for command

    Returns:
        Command output as string

    Raises:
        GitError: If git command fails
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise GitError(f"Git command failed: {e.stderr}") from e
    except FileNotFoundError:
        raise GitError("git command not found. Is git installed?")


def is_git_repo(path: Optional[Path] = None) -> bool:
    """Check if current directory is a git repository.

    Args:
        path: Directory to check. Defaults to current directory.

    Returns:
        True if directory is a git repo
    """
    try:
        _run_git_command(["rev-parse", "--git-dir"], cwd=path)
        return True
    except GitError:
        return False


def get_staged_diff() -> str:
    """Get the diff of staged changes.

    Returns:
        Staged diff as string

    Raises:
        GitError: If not in a git repo or command fails
    """
    diff = _run_git_command(["diff", "--cached"])
    if not diff:
        raise GitError("No staged changes found. Use 'git add' to stage changes first.")
    return diff


def get_changed_files() -> List[str]:
    """Get list of staged file names.

    Returns:
        List of staged file paths

    Raises:
        GitError: If not in a git repo or command fails
    """
    output = _run_git_command(["diff", "--cached", "--name-only"])
    if not output.strip():
        return []
    return [line.strip() for line in output.strip().split("\n") if line.strip()]


def get_recent_commit_messages(count: int = 5) -> List[str]:
    """Get recent commit messages for context.

    Args:
        count: Number of recent commits to retrieve

    Returns:
        List of commit messages

    Raises:
        GitError: If not in a git repo or command fails
    """
    try:
        output = _run_git_command(
            ["log", f"-{count}", "--pretty=format:%s", "--no-merges"]
        )
        if not output.strip():
            return []
        return [line.strip() for line in output.strip().split("\n") if line.strip()]
    except GitError:
        # No commits yet in repo
        return []


def commit(message: str, push: bool = False) -> None:
    """Create a git commit with the given message.

    Args:
        message: Commit message
        push: Whether to push after committing

    Raises:
        GitError: If commit fails
    """
    _run_git_command(["commit", "-m", message])

    if push:
        _run_git_command(["push"])


def get_git_status() -> str:
    """Get git status output.

    Returns:
        Git status output

    Raises:
        GitError: If not in a git repo or command fails
    """
    return _run_git_command(["status", "--short"])
