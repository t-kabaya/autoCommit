# gac - Git Auto Commit

Local LLM-powered git commit message generator. Generate high-quality commit messages using a locally-running language model.

**100% offline. No external APIs. Complete privacy.**

## Features

- 🤖 **Local LLM**: Uses llama.cpp with GGUF models (no transformers, no torch)
- 🎯 **Conventional Commits**: Generates standard-compliant commit messages
- 🚀 **Interactive Mode**: Choose from multiple generated candidates
- 📝 **Context-Aware**: Analyzes diffs, file names, and commit history
- 🔒 **Privacy-First**: Everything runs locally
- ⚡ **Fast**: Lightweight and efficient
- 🛠️ **Easy Setup**: One command installation

## Installation

### Prerequisites

- Python 3.11+
- Git
- macOS or Linux
- [uv](https://github.com/astral-sh/uv) (recommended)

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install gac

```bash
# Clone or download the repository
cd gac

# Install with uv
uv pip install -e .

# Or with pip
pip install -e .

# Run initial setup (downloads llama.cpp and model)
gac install
```

The `gac install` command will:
- Download llama.cpp binary for your platform
- Download Gemma 3 1B IT GGUF model (~700MB)
- Create configuration at `~/.gac/config.toml`

**Note**: Initial setup may take a few minutes depending on your internet connection.

## Usage

### Basic Usage

```bash
# Stage your changes
git add .

# Generate commit message and commit
gac commit
```

This will:
1. Analyze your staged changes
2. Generate 3 commit message options
3. Let you choose which one to use
4. Create the commit

### Command Options

```bash
# Commit and push immediately
gac commit --push

# Skip confirmation (use first generated message)
gac commit --yes

# Dry run (generate message only, don't commit)
gac commit --dry-run

# Show debug output
gac commit --verbose

# Single message mode (no interactive selection)
gac commit --no-interactive

# Combine options
gac commit --yes --push
```

### Other Commands

```bash
# Show current configuration
gac config

# Reinstall/update
gac install

# Show version
gac version
```

## Configuration

Configuration file location: `~/.gac/config.toml`

```toml
model = "~/.gac/models/gemma-3-1b-it-Q4_K_M.gguf"
llama_cli = "~/.gac/bin/llama-cli"
temperature = 0.2
max_tokens = 64
num_candidates = 3
```

You can edit this file to:
- Change model path
- Adjust temperature (0.0-1.0, lower = more focused)
- Change max tokens for generation
- Set number of candidates in interactive mode

## Architecture

```
gac/
├── src/gac/
│   ├── cli.py          # Typer CLI interface
│   ├── config.py       # Configuration management
│   ├── git_utils.py    # Git operations
│   ├── llm.py          # llama.cpp integration
│   ├── prompts.py      # Prompt templates
│   └── installer.py    # Setup automation
└── pyproject.toml      # Project configuration
```

### Design Principles

- **Subprocess over libraries**: Uses llama.cpp directly via subprocess (no transformers)
- **Type safety**: Full type hints throughout
- **Error handling**: Comprehensive error messages
- **Clean architecture**: Separated concerns, testable code
- **User experience**: Rich terminal UI with progress indicators

## Examples

### Example 1: Feature Addition

```bash
$ git add src/api/auth.py
$ gac commit

Changed files:
  • src/api/auth.py

Generating commit message...

Commit message options:

  1. feat(auth): add retry logic for token refresh
  2. feat(api): implement automatic token retry handling
  3. refactor(auth): improve token refresh reliability

Select option (or 0 to cancel) [1]: 1

Generated commit message:
╭─────────────────────────────────────────────╮
│ feat(auth): add retry logic for token refresh │
╰─────────────────────────────────────────────╯

Create commit with this message? [Y/n]: y

✓ Commit created successfully!
```

### Example 2: Bug Fix with Push

```bash
$ git add src/utils/validator.py
$ gac commit --yes --push

Changed files:
  • src/utils/validator.py

Generating commit message...

Generated commit message:
╭──────────────────────────────────────────────╮
│ fix(validator): handle null input correctly │
╰──────────────────────────────────────────────╯

✓ Commit created successfully!
✓ Changes pushed to remote
```

### Example 3: Dry Run

```bash
$ git add .
$ gac commit --dry-run --no-interactive

Changed files:
  • README.md
  • docs/api.md

Generating commit message...

Generated commit message:
╭────────────────────────────────────╮
│ docs: update API documentation │
╰────────────────────────────────────╯

Dry run - no commit created
```

## Troubleshooting

### "gac is not installed"

Run `gac install` to download required components.

### "Model not found"

The model file may not have downloaded correctly. Try:

```bash
rm -rf ~/.gac
gac install
```

### "llama-cli not found"

Similar to above - reinstall:

```bash
gac install
```

### Generated messages are too generic

Try adjusting temperature in `~/.gac/config.toml`:

```toml
temperature = 0.1  # Lower = more focused/consistent
```

### Slow generation

The default Gemma 3 1B model is very lightweight and fast. If it's still slow:

1. Check CPU usage - llama.cpp should use multiple cores
2. Reduce `max_tokens` in config
3. Try a smaller quantization (e.g., Q3 instead of Q4)

### "No staged changes found"

Make sure to `git add` your changes before running `gac commit`:

```bash
git add <files>
gac commit
```

### Platform-specific issues

**macOS**: On Apple Silicon, the ARM64 build is automatically selected for better performance.

**Linux**: Ensure you have `unzip` installed for the setup process:

```bash
sudo apt-get install unzip  # Debian/Ubuntu
sudo yum install unzip      # RHEL/CentOS
```

## Model Information

Default model: **Gemma 3 1B IT** (Q4_K_M quantization)

- Size: ~700MB
- Context: 8K tokens
- Speed: Very fast, even on older CPUs
- Quality: Great for commit messages, optimized for lightweight usage

### Using a Custom Model

You can use any GGUF model that works with llama.cpp:

1. Download your preferred GGUF model
2. Update `~/.gac/config.toml`:

```toml
model = "/path/to/your/model.gguf"
```

3. Test with `gac commit --dry-run`

Recommended models:
- Gemma 3 series (1B, 2B) - lightweight and fast
- Qwen2.5-Coder series (1.5B, 3B, 7B) - excellent for code
- CodeLlama GGUF models
- Phi-3 Mini - very small and efficient

## Development

### Setup for Development

```bash
# Clone repository
git clone <your-repo>
cd gac

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Type check
mypy src/
```

### Project Structure

- `src/gac/cli.py` - Main CLI entry point using Typer
- `src/gac/git_utils.py` - Git operations (diff, status, commit)
- `src/gac/llm.py` - LLM interface (llama.cpp subprocess)
- `src/gac/prompts.py` - Prompt engineering
- `src/gac/config.py` - Configuration management
- `src/gac/installer.py` - Installation automation

## Why gac?

- **Privacy**: Your code never leaves your machine
- **Speed**: No network latency, instant generation
- **Cost**: Zero API costs
- **Reliability**: Works offline, no rate limits
- **Customizable**: Use any GGUF model you prefer

## Limitations

- Requires ~2GB disk space for model
- CPU-based inference (no GPU acceleration in default setup)
- macOS and Linux only (no Windows support)
- Requires staging files before commit (standard git workflow)

## License

MIT

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes type checking and formatting
5. Submit a pull request

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Fast LLM inference
- [Qwen2.5-Coder](https://huggingface.co/Qwen) - Excellent code model
- [Typer](https://typer.tiangolo.com/) - Great CLI framework
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal output

---

**Made with ❤️ for developers who value privacy and efficiency**



