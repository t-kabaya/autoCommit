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

## Dataset

This project uses the [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) dataset for fine-tuning commit message generation models.

### Dataset Examples

**Original CommitPackFT (Python) - 56,025 samples**

Sample commit messages from the dataset:

```json
{
  "commit": "e905334869af72025592de586b81650cb3468b8a",
  "old_file": "sentry/queue/client.py",
  "new_file": "sentry/queue/client.py",
  "subject": "Declare queues when broker is instantiated",
  "message": "Declare queues when broker is instantiated\n",
  "lang": "Python",
  "license": "bsd-3-clause"
}
```

```json
{
  "subject": "Fix % only showing 0 or 100%, everything between goes to 0%.",
  "old_file": "src/dashboard/src/main/templatetags/percentage.py"
}
```

```json
{
  "subject": "Remove 'validation' from RejectionException docstring",
  "old_file": "automata/base/exceptions.py"
}
```

### Filtered Dataset - 33,510 samples (59.81%)

Filtered to include only commits starting with action verbs:
- Add (additions)
- Fix (bug fixes)
- Update (updates)
- Remove (deletions)
- Refactor (code improvements)

**Examples:**

```json
{
  "subject": "Fix interpretation of parameters for names list modification",
  "old_file": "txircd/modules/umode_i.py"
}
```

```json
{
  "subject": "Add missing 'add' line",
  "old_file": "main.py"
}
```

```json
{
  "subject": "Update the utils.py to include the new version",
  "old_file": "utils.py"
}
```

### Dataset Statistics

| Dataset | Samples | Match Rate | Format |
|---------|---------|------------|--------|
| CommitPackFT (Python) | 56,025 | 100% | Natural language |
| Filtered (Action verbs) | 33,510 | 59.81% | Verb-first format |

### Using the Dataset

Load the full dataset:
```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="hf://datasets/bigcode/commitpackft/data/python/data.jsonl"
)
```

Load the filtered dataset:
```python
dataset = load_dataset("json", data_files="datasets/commitpack/filtered_commitpack.jsonl")
```

### Dataset Filtering

See `datasets/commitpack/filter_dataset.py` for filtering by commit type:

```bash
python datasets/commitpack/filter_dataset.py \
  --types "Add,Fix,Update,Remove,Refactor" \
  --output filtered_commitpack.jsonl
```

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


