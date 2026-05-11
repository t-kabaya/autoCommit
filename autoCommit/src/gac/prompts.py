"""Prompt templates for commit message generation."""

from typing import List


def build_commit_prompt(
    diff: str,
    changed_files: List[str],
    recent_commits: List[str],
) -> str:
    """Build prompt for commit message generation.

    Args:
        diff: The staged diff
        changed_files: List of changed file names
        recent_commits: Recent commit messages for context

    Returns:
        Formatted prompt string
    """
    files_str = "\n".join(f"- {f}" for f in changed_files)
    recent_str = "\n".join(f"- {c}" for c in recent_commits) if recent_commits else "None"

    # Truncate diff if too long (keep first 3000 chars)
    if len(diff) > 3000:
        diff = diff[:3000] + "\n... (diff truncated)"

    prompt = f"""Generate a git commit message (under 30 chars) following conventional commits format (feat/fix/docs/style/refactor/test/chore).

Files changed:
{files_str}

Recent commits:
{recent_str}

Diff:
{diff}

Commit message:"""

    return prompt


def build_multi_candidate_prompt(
    diff: str,
    changed_files: List[str],
    recent_commits: List[str],
    num_candidates: int = 3,
) -> str:
    """Build prompt for generating multiple commit message candidates.

    Args:
        diff: The staged diff
        changed_files: List of changed file names
        recent_commits: Recent commit messages for context
        num_candidates: Number of candidates to generate

    Returns:
        Formatted prompt string
    """
    files_str = "\n".join(f"- {f}" for f in changed_files)
    recent_str = "\n".join(f"- {c}" for c in recent_commits) if recent_commits else "None"

    # Truncate diff if too long
    if len(diff) > 3000:
        diff = diff[:3000] + "\n... (diff truncated)"

    prompt = f"""You are a git commit message generator. Generate {num_candidates} different commit message options following Conventional Commits format.

RULES:
1. Follow Conventional Commits: type(scope): description
2. Types: feat, fix, docs, style, refactor, test, chore
3. Keep each under 72 characters
4. Be specific and descriptive
5. Focus on WHAT changed and WHY
6. Output ONLY the commit messages, one per line, numbered

Changed Files:
{files_str}

Recent Commit Messages (for style reference):
{recent_str}

Staged Diff:
{diff}

Generate {num_candidates} commit message options (numbered 1, 2, 3):"""

    return prompt
