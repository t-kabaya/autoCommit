"""
Prepare CommitPackFT dataset with git diff format
Creates a new dataset with actual code diffs for better training
"""
from datasets import load_dataset, Dataset
import difflib
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_unified_diff(old_content, new_content, old_file, new_file, context_lines=3):
    """
    Generate unified diff format (like git diff) from old and new content

    Args:
        old_content: Original file content
        new_content: Modified file content
        old_file: Old file path
        new_file: New file path
        context_lines: Number of context lines to include

    Returns:
        Unified diff string
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{old_file}",
        tofile=f"b/{new_file}",
        lineterm='',
        n=context_lines
    )

    return ''.join(diff)


def prepare_dataset_with_diff(dataset, max_diff_length=500, num_samples=None):
    """
    Prepare dataset by adding git diff for each sample

    Args:
        dataset: Original CommitPackFT dataset
        max_diff_length: Maximum diff length in characters
        num_samples: Number of samples to process (None for all)

    Returns:
        New dataset with diff field added
    """
    logger.info("Generating diffs for dataset...")

    prepared_samples = []
    total = len(dataset['train']) if num_samples is None else min(num_samples, len(dataset['train']))

    for i in tqdm(range(total)):
        sample = dataset['train'][i]

        old_content = sample.get('old_contents', '')
        new_content = sample.get('new_contents', '')
        old_file = sample.get('old_file', 'file.py')
        new_file = sample.get('new_file', 'file.py')

        # Generate diff
        diff = generate_unified_diff(old_content, new_content, old_file, new_file)

        # Truncate diff if too long
        if len(diff) > max_diff_length:
            diff = diff[:max_diff_length] + "\n... (diff truncated)"

        # Skip if diff is empty (no changes)
        if not diff.strip():
            continue

        # Create new sample with diff
        prepared_sample = {
            'file': old_file,
            'diff': diff,
            'subject': sample.get('subject', ''),
            'message': sample.get('message', ''),
            'commit': sample.get('commit', ''),
            'lang': sample.get('lang', ''),
        }

        prepared_samples.append(prepared_sample)

    logger.info(f"Prepared {len(prepared_samples)} samples with diffs")

    # Convert to Dataset
    return Dataset.from_list(prepared_samples)


def save_prepared_dataset(dataset, output_file="prepared_dataset.jsonl"):
    """Save prepared dataset to JSONL file"""
    logger.info(f"Saving prepared dataset to {output_file}...")

    with open(output_file, 'w') as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(dataset)} samples to {output_file}")


def show_examples(dataset, num_examples=3):
    """Show example diffs from the prepared dataset"""
    print("\n" + "="*80)
    print("EXAMPLE DIFFS FROM PREPARED DATASET")
    print("="*80)

    for i in range(min(num_examples, len(dataset))):
        sample = dataset[i]
        print(f"\n{'='*80}")
        print(f"Example {i+1}")
        print(f"{'='*80}")
        print(f"File: {sample['file']}")
        print(f"\nDiff:")
        print(sample['diff'])
        print(f"\nCommit Message: {sample['subject']}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare CommitPackFT with git diffs")
    parser.add_argument("--language", type=str, default="python", help="Programming language")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--max_diff_length", type=int, default=500,
                       help="Maximum diff length in characters")
    parser.add_argument("--output", type=str, default="prepared_commitpack.jsonl",
                       help="Output file name")
    parser.add_argument("--show_examples", type=int, default=3,
                       help="Number of examples to show")
    parser.add_argument("--analyze", action="store_true",
                       help="Only analyze dataset without preparing")

    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading CommitPackFT dataset ({args.language} language)...")
    dataset = load_dataset(
        "json",
        data_files=f"hf://datasets/bigcode/commitpackft/data/{args.language}/data.jsonl"
    )

    if args.analyze:
        # Just show statistics
        logger.info(f"Total samples: {len(dataset['train'])}")
        logger.info("Showing first 5 samples...")
        show_examples(Dataset.from_list([dataset['train'][i] for i in range(5)]), 5)
    else:
        # Prepare dataset with diffs
        prepared_dataset = prepare_dataset_with_diff(
            dataset,
            max_diff_length=args.max_diff_length,
            num_samples=args.num_samples
        )

        # Save to file
        save_prepared_dataset(prepared_dataset, args.output)

        # Show examples
        show_examples(prepared_dataset, args.show_examples)
