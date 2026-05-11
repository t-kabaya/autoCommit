"""CLI interface for gac."""

import sys
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt

from . import git_utils, prompts
from .config import Config
from .llm import LlamaLLM, LLMError

app = typer.Typer(
    name="gac",
    help="Git Auto Commit - Local LLM powered commit message generator",
    add_completion=False,
)
console = Console()


def check_setup() -> tuple[Config, LlamaLLM]:
    """Check if gac is set up and return config and LLM.

    Returns:
        Tuple of (Config, LlamaLLM)

    Raises:
        typer.Exit: If not set up
    """
    config = Config()

    if not config.is_configured():
        console.print(
            "[bold red]Error:[/bold red] No model configured. Set 'model' in ~/.gac/config.toml"
        )
        raise typer.Exit(1)

    try:
        llm = LlamaLLM(
            model_path=config.model_path,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        return config, llm
    except LLMError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def commit(
    push: bool = typer.Option(False, "--push", "-p", help="Push after committing"),
    yes: bool = typer.Option(True, "--yes/--no-yes", "-y", help="Skip confirmation (default: yes)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Generate message only"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug output"),
    interactive: bool = typer.Option(
        False, "--interactive/--no-interactive", "-i/-I", help="Show multiple candidates"
    ),
    fast: bool = typer.Option(False, "--fast", "-f", help="Use gemma-3-1b model (~2GB)"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model size: small, medium (or HF model ID)"
    ),
) -> None:
    """Generate commit message and create commit."""
    # Check if in git repo
    if not git_utils.is_git_repo():
        console.print("[bold red]Error:[/bold red] Not in a git repository")
        raise typer.Exit(1)

    # Check setup
    config, llm = check_setup()

    # Fast mode: use smaller model
    if fast:
        console.print(f"[cyan]Fast mode: Using {Config.FAST_MODEL}[/cyan]")
        llm = LlamaLLM(
            model_path=Config.FAST_MODEL,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    # Override model if specified
    elif model:
        if model in Config.MODELS:
            model_id = Config.MODELS[model]
            console.print(f"[cyan]Using {model} model: {model_id}[/cyan]")
        else:
            model_id = model
            console.print(f"[cyan]Using custom model: {model_id}[/cyan]")
        llm = LlamaLLM(
            model_path=model_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    # Auto-stage all changes
    try:
        # Check if there are any changes to stage
        status = git_utils.get_git_status()
        if status.strip():
            console.print("[cyan]Auto-staging changes...[/cyan]")
            git_utils._run_git_command(["add", "."])

        # Get git information
        diff = git_utils.get_staged_diff()
        changed_files = git_utils.get_changed_files()
        recent_commits = git_utils.get_recent_commit_messages()
    except git_utils.GitError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)

    # Show changed files
    console.print("\n[bold cyan]Changed files:[/bold cyan]")
    for file in changed_files:
        console.print(f"  • {file}")

    # Generate commit message(s)
    console.print("\n[bold cyan]Generating commit message...[/bold cyan]")

    try:
        if interactive:
            # Generate multiple candidates
            prompt = prompts.build_multi_candidate_prompt(
                diff=diff,
                changed_files=changed_files,
                recent_commits=recent_commits,
                num_candidates=config.num_candidates,
            )
            candidates = llm.generate_candidates(
                prompt, num_candidates=config.num_candidates, verbose=verbose
            )

            if not candidates:
                console.print("[bold red]Error:[/bold red] Failed to generate commit messages")
                raise typer.Exit(1)

            # Display candidates
            console.print("\n[bold cyan]Commit message options:[/bold cyan]\n")
            for i, msg in enumerate(candidates, 1):
                console.print(f"  [bold]{i}.[/bold] {msg}")

            # User selection
            console.print()
            if yes:
                choice = 1
            else:
                choice = IntPrompt.ask(
                    "Select option (or 0 to cancel)",
                    default=1,
                    show_default=True,
                )

            if choice == 0:
                console.print("[yellow]Cancelled[/yellow]")
                raise typer.Exit(0)

            if choice < 1 or choice > len(candidates):
                console.print("[bold red]Error:[/bold red] Invalid choice")
                raise typer.Exit(1)

            commit_message = candidates[choice - 1]

        else:
            # Generate single message
            prompt = prompts.build_commit_prompt(
                diff=diff,
                changed_files=changed_files,
                recent_commits=recent_commits,
            )
            commit_message = llm.generate(prompt, verbose=verbose)

    except LLMError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)

    # Display final message
    console.print("\n[bold green]Generated commit message:[/bold green]")
    console.print(Panel(commit_message, border_style="green"))

    if dry_run:
        console.print("\n[yellow]Dry run - no commit created[/yellow]")
        raise typer.Exit(0)

    # Confirm
    if not yes:
        if not Confirm.ask("\nCreate commit with this message?", default=True):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    # Create commit
    try:
        git_utils.commit(commit_message, push=push)
        console.print("\n[bold green]✓ Commit created successfully![/bold green]")

        if push:
            console.print("[bold green]✓ Changes pushed to remote[/bold green]")

    except git_utils.GitError as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def config() -> None:
    """Show current configuration."""
    conf = Config()

    console.print("\n[bold cyan]═══ gac Configuration ═══[/bold cyan]\n")

    # Check if configured
    if not conf.is_configured():
        console.print(
            "[yellow]Warning:[/yellow] gac is not fully configured. Run [bold]gac install[/bold]"
        )
        console.print()

    # Display config
    config_items = [
        ("Model", conf.get("model"), conf.model_path.exists()),
        ("llama-cli", conf.get("llama_cli"), conf.llama_cli_path.exists()),
        ("Temperature", conf.get("temperature"), True),
        ("Max tokens", conf.get("max_tokens"), True),
        ("Candidates", conf.get("num_candidates"), True),
    ]

    for key, value, exists in config_items:
        status = "✓" if exists else "✗"
        color = "green" if exists else "red"
        console.print(f"  [{color}]{status}[/{color}] [bold]{key}:[/bold] {value}")

    console.print(f"\n[dim]Config file: {conf.config_path}[/dim]\n")


@app.command()
def version() -> None:
    """Show version information."""
    console.print("[bold cyan]gac[/bold cyan] version [bold]0.1.0[/bold]")
    console.print("Git Auto Commit - Local LLM powered commit message generator")


def main() -> None:
    """Main entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
