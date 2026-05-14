from datasets import load_dataset

# The commitpackft dataset uses JSONL files instead of Parquet
# Load Python language data as an example
print("Loading commitpackft dataset (Python language)...")

try:
    dataset = load_dataset(
        "json",
        data_files="hf://datasets/bigcode/commitpackft/data/python/data.jsonl"
    )

    print("\nDataset loaded successfully!")
    print(f"Dataset info: {dataset}")
    print(f"\nNumber of samples: {len(dataset['train'])}")
    print(f"\nFirst 3 samples:")
    for i in range(min(3, len(dataset['train']))):
        sample = dataset['train'][i]
        print(f"\n--- Sample {i+1} ---")
        print(f"Commit: {sample.get('commit', 'N/A')[:100]}...")
        print(f"Subject: {sample.get('subject', 'N/A')}")

except Exception as e:
    print(f"Error: {e}")
