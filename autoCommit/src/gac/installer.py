"""Installation utilities for gac."""

import subprocess
import sys
import platform
from pathlib import Path
from typing import Optional
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import Config


console = Console()


class InstallError(Exception):
    """Exception raised for installation errors."""

    pass


def get_system_info() -> tuple[str, str]:
    """Get system OS and architecture.

    Returns:
        Tuple of (os, arch) strings

    Raises:
        InstallError: If system is not supported
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system not in ["darwin", "linux"]:
        raise InstallError(f"Unsupported OS: {system}. Only macOS and Linux are supported.")

    # Map architecture names
    arch_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }

    arch = arch_map.get(machine)
    if not arch:
        raise InstallError(f"Unsupported architecture: {machine}")

    return system, arch


def download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Download a file with progress bar.

    Args:
        url: URL to download from
        dest: Destination path
        desc: Description for progress bar

    Raises:
        InstallError: If download fails
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        dest.parent.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"{desc}...", total=total_size)

            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

    except requests.RequestException as e:
        raise InstallError(f"Failed to download {url}: {str(e)}") from e


def install_llama_cpp(install_dir: Path) -> Path:
    """Install llama.cpp binary.

    Args:
        install_dir: Installation directory

    Returns:
        Path to installed llama-cli binary

    Raises:
        InstallError: If installation fails
    """
    system, arch = get_system_info()

    # llama.cpp release URLs
    # Using latest release from GitHub (supports Gemma 3)
    version = "b9070"  # Latest version as of May 2025
    base_url = f"https://github.com/ggerganov/llama.cpp/releases/download/{version}"

    if system == "darwin":
        if arch == "arm64":
            filename = f"llama-{version}-bin-macos-arm64.tar.gz"
        else:
            filename = f"llama-{version}-bin-macos-x64.tar.gz"
    else:  # linux
        if arch == "arm64":
            filename = f"llama-{version}-bin-ubuntu-aarch64.tar.gz"
        else:
            filename = f"llama-{version}-bin-ubuntu-x64.tar.gz"

    url = f"{base_url}/{filename}"
    bin_dir = install_dir / "bin"
    archive_path = install_dir / filename

    console.print(f"[cyan]Downloading llama.cpp for {system}/{arch}...[/cyan]")

    try:
        download_file(url, archive_path, "Downloading llama.cpp")

        # Extract
        console.print("[cyan]Extracting...[/cyan]")
        subprocess.run(["tar", "-xzf", str(archive_path), "-C", str(install_dir)], check=True)

        # Find llama-cli binary and its directory
        llama_cli = None
        source_dir = None
        for path in install_dir.rglob("llama-cli"):
            if path.is_file():
                llama_cli = path
                source_dir = path.parent
                break

        if not llama_cli or not source_dir:
            raise InstallError("llama-cli binary not found in extracted files")

        # Create bin directory
        bin_dir.mkdir(parents=True, exist_ok=True)

        # Move llama-cli binary
        final_path = bin_dir / "llama-cli"
        llama_cli.rename(final_path)
        final_path.chmod(0o755)

        # Move all .dylib files (shared libraries) to bin directory
        for dylib in source_dir.glob("*.dylib*"):
            dest = bin_dir / dylib.name
            dylib.rename(dest)
            console.print(f"[dim]  Moved {dylib.name}[/dim]")

        # Cleanup
        archive_path.unlink()
        for item in install_dir.iterdir():
            if item.is_dir() and item != bin_dir:
                subprocess.run(["rm", "-rf", str(item)], check=False)

        console.print(f"[green]✓ llama.cpp installed to {final_path}[/green]")
        return final_path

    except Exception as e:
        raise InstallError(f"Failed to install llama.cpp: {str(e)}") from e


def install_model(install_dir: Path, model_url: Optional[str] = None) -> Path:
    """Install GGUF model.

    Args:
        install_dir: Installation directory
        model_url: Custom model URL (optional)

    Returns:
        Path to installed model

    Raises:
        InstallError: If installation fails
    """
    models_dir = install_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Default: Gemma 3 1B IT Q4_K_M (lightweight, fast)
    if model_url is None:
        model_name = "gemma-3-1b-it-Q4_K_M.gguf"
        # Using unsloth's repo - reliable and well-maintained
        model_url = f"https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/{model_name}"
    else:
        model_name = model_url.split("/")[-1]

    model_path = models_dir / model_name

    if model_path.exists():
        console.print(f"[yellow]Model already exists at {model_path}[/yellow]")
        return model_path

    console.print(f"[cyan]Downloading model: {model_name}...[/cyan]")
    console.print("[yellow]This may take several minutes...[/yellow]")

    download_file(model_url, model_path, f"Downloading {model_name}")

    console.print(f"[green]✓ Model installed to {model_path}[/green]")
    return model_path


def run_installation() -> None:
    """Run full installation process.

    Raises:
        InstallError: If installation fails
    """
    install_dir = Path.home() / ".gac"
    install_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]═══ gac Installation ═══[/bold cyan]\n")

    try:
        # Install llama.cpp
        llama_cli_path = install_llama_cpp(install_dir)

        # Install model
        model_path = install_model(install_dir)

        # Create config
        config = Config()
        config.set("llama_cli", str(llama_cli_path))
        config.set("model", str(model_path))
        config.save()

        console.print("\n[bold green]✓ Installation complete![/bold green]")
        console.print(f"\nConfiguration saved to: {config.config_path}")
        console.print("\nYou can now run: [bold]gac commit[/bold]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Installation failed: {str(e)}[/bold red]")
        raise
