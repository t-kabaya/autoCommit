from datasets import load_dataset

# Load the commitpackft dataset (Python language)
print("Loading commitpackft dataset (Python language)...")

dataset = load_dataset(
    "json",
    data_files="hf://datasets/bigcode/commitpackft/data/python/data.jsonl"
)

# Print the first sample
first_sample = dataset['train'][0]

print("\n=== First Sample ===\n")
for key, value in first_sample.items():
    print(f"{key}:")
    if isinstance(value, str) and len(value) > 200:
        print(f"{value[:200]}...\n")
    else:
        print(f"{value}\n")
