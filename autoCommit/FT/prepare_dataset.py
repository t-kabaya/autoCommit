"""
Dataset preparation script for CommitPackFT
This script loads and explores the dataset before training
"""
from datasets import load_dataset
import json


def load_and_explore_dataset(language="python", num_samples=10):
    """Load and explore the CommitPackFT dataset"""

    print(f"Loading CommitPackFT dataset ({language} language)...")
    dataset = load_dataset(
        "json",
        data_files=f"hf://datasets/bigcode/commitpackft/data/{language}/data.jsonl"
    )

    print(f"\nDataset loaded: {len(dataset['train'])} samples")
    print(f"\nDataset features: {dataset['train'].features}")

    # Show sample statistics
    print(f"\n=== Sample Statistics ===")

    # Message lengths
    message_lengths = [len(sample['message']) for sample in dataset['train']]
    print(f"Average message length: {sum(message_lengths) / len(message_lengths):.1f} chars")
    print(f"Max message length: {max(message_lengths)} chars")
    print(f"Min message length: {min(message_lengths)} chars")

    # Show first few samples
    print(f"\n=== First {num_samples} Samples ===")
    for i in range(min(num_samples, len(dataset['train']))):
        sample = dataset['train'][i]
        print(f"\n--- Sample {i+1} ---")
        print(f"File: {sample['old_file']}")
        print(f"Subject: {sample['subject']}")
        print(f"Message: {sample['message'][:100]}...")
        print(f"Old contents length: {len(sample['old_contents'])} chars")
        print(f"New contents length: {len(sample['new_contents'])} chars")

    return dataset


def create_training_subset(dataset, output_file="train_subset.jsonl", num_samples=1000):
    """Create a smaller subset for quick training experiments"""
    print(f"\nCreating training subset with {num_samples} samples...")

    subset = dataset['train'].select(range(min(num_samples, len(dataset['train']))))

    # Save to JSONL
    with open(output_file, 'w') as f:
        for sample in subset:
            f.write(json.dumps(sample) + '\n')

    print(f"Saved {len(subset)} samples to {output_file}")
    return subset


def analyze_commit_patterns(dataset, num_samples=1000):
    """Analyze common patterns in commit messages"""
    print("\n=== Analyzing Commit Patterns ===")

    subjects = [dataset['train'][i]['subject'] for i in range(min(num_samples, len(dataset['train'])))]

    # Find common starting words
    first_words = {}
    for subject in subjects:
        if subject:
            first_word = subject.split()[0] if subject.split() else ''
            first_words[first_word] = first_words.get(first_word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(first_words.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 20 most common first words in commit subjects:")
    for word, count in sorted_words[:20]:
        print(f"  {word}: {count} ({count/num_samples*100:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare CommitPackFT dataset")
    parser.add_argument("--language", type=str, default="python", help="Programming language")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to show")
    parser.add_argument("--create_subset", action="store_true", help="Create training subset")
    parser.add_argument("--subset_size", type=int, default=1000, help="Size of training subset")
    parser.add_argument("--analyze", action="store_true", help="Analyze commit patterns")

    args = parser.parse_args()

    # Load and explore dataset
    dataset = load_and_explore_dataset(
        language=args.language,
        num_samples=args.num_samples
    )

    # Create subset if requested
    if args.create_subset:
        create_training_subset(
            dataset,
            output_file=f"train_subset_{args.language}.jsonl",
            num_samples=args.subset_size
        )

    # Analyze patterns if requested
    if args.analyze:
        analyze_commit_patterns(dataset, num_samples=min(args.subset_size, len(dataset['train'])))
