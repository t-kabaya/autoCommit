# gac - Git Auto Commit

AI-powered git commit message generator using local LLM.

**Fully offline. No external API. Privacy-focused.**

## Features

- 🤖 Local LLM (transformers + Gemma 3)
- 🎯 Conventional Commits format
- 📝 Analyzes diff and history
- ⚡ Lightweight and fast
- 🔒 Completely private

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install gac
cd autoCommit
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

## Usage

```bash
# Basic usage
gac commit

# Fast mode (uses Gemma 3-1b)
gac commit --fast

# With push
gac commit --push

# Interactive mode (choose from multiple candidates)
gac commit --interactive

# Dry run (generate message only)
gac commit --dry-run
```

**Note**: `git add .` and auto-commit are executed by default.

## Configuration

`~/.gac/config.toml`

```toml
model = "google/gemma-2-2b-it"  # HuggingFace model ID
temperature = 0.2
max_tokens = 64
num_candidates = 3
```

## Other Commands

```bash
gac config   # Show configuration
gac version  # Show version
```

## Model Information

- **Default**: Gemma 2 2B IT
- **Fast mode**: Gemma 3 1B IT (~2GB)
- **Size**: ~4GB (default), ~2GB (fast)
- **Speed**: Optimized for Apple Silicon (M1/M2/M3/M4)

### Custom Models

Edit `~/.gac/config.toml`:
```bash
model = "google/gemma-3-3b-it"  # or other HuggingFace models
```

Recommended models:
- Gemma 3 (1B, 3B) - Lightweight
- Gemma 2 (2B, 9B) - Balanced
- Qwen2.5-Coder - Code-specific

## Troubleshooting

### Slow generation

Edit `~/.gac/config.toml`:
```toml
max_tokens = 32
temperature = 0.1
```

### First run is slow

The model downloads on first use (~2-4GB). Subsequent runs use the cached model and are much faster.

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Publishing to PyPI

```bash
# Get PyPI API token (first time only)
# https://pypi.org/manage/account/token/

# Set environment variables (optional)
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=your-pypi-token

# Run publish script
./publish.sh
```

After publishing, users can install with:
```bash
pip install gac
# or
uv pip install gac
```

## License

MIT

## Credits

- [Transformers](https://github.com/huggingface/transformers) - LLM framework
- [Gemma 3](https://huggingface.co/google/gemma-3-1b-it) - Language model
- [Typer](https://typer.tiangolo.com/) - CLI framework


