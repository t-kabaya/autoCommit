"""
Evaluate fine-tuned model using BERTScore and other metrics
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path):
    """Load fine-tuned model and tokenizer"""
    logger.info(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_test_dataset(dataset_file=None, num_samples=100):
    """Load test dataset"""
    if dataset_file:
        logger.info(f"Loading test data from: {dataset_file}")
        dataset = load_dataset("json", data_files=dataset_file)
    else:
        logger.info("Loading CommitPackFT dataset (Python)...")
        dataset = load_dataset(
            "json",
            data_files="hf://datasets/bigcode/commitpackft/data/python/data.jsonl"
        )

    # Use last N samples as test set (assuming they weren't in training)
    total = len(dataset['train'])
    start_idx = max(0, total - num_samples)
    test_dataset = dataset['train'].select(range(start_idx, total))

    logger.info(f"Loaded {len(test_dataset)} test samples")
    return test_dataset


def generate_commit_message(model, tokenizer, file_name, max_length=100):
    """Generate commit message for a given file"""
    prompt = f"File: {file_name}\nCommit: "

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the commit message part (after "Commit: ")
    if "Commit: " in generated_text:
        commit_msg = generated_text.split("Commit: ", 1)[1].strip()
        # Take only the first line
        commit_msg = commit_msg.split('\n')[0].strip()
        return commit_msg

    return generated_text


def calculate_bertscore(predictions, references):
    """Calculate BERTScore for predictions vs references"""
    try:
        from bert_score import score

        logger.info("Calculating BERTScore...")
        P, R, F1 = score(
            predictions,
            references,
            lang="en",
            verbose=False,
            device="cpu"
        )

        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }
    except ImportError:
        logger.warning("bert_score not installed. Run: pip install bert-score")
        return None


def calculate_simple_metrics(predictions, references):
    """Calculate simple text similarity metrics"""
    from difflib import SequenceMatcher

    similarities = []
    exact_matches = 0

    for pred, ref in zip(predictions, references):
        # Exact match
        if pred.lower() == ref.lower():
            exact_matches += 1

        # Sequence similarity
        similarity = SequenceMatcher(None, pred.lower(), ref.lower()).ratio()
        similarities.append(similarity)

    return {
        "exact_match": exact_matches / len(predictions),
        "avg_similarity": np.mean(similarities),
        "median_similarity": np.median(similarities)
    }


def evaluate_model(model_path, test_dataset_file=None, num_samples=100, output_file="evaluation_results.json"):
    """Main evaluation function"""

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load test dataset
    test_dataset = load_test_dataset(test_dataset_file, num_samples)

    # Generate predictions
    logger.info("Generating commit messages...")
    predictions = []
    references = []

    for sample in tqdm(test_dataset):
        file_name = sample.get('old_file', 'unknown.py')
        reference = sample.get('subject', '')

        # Generate prediction
        prediction = generate_commit_message(model, tokenizer, file_name)

        predictions.append(prediction)
        references.append(reference)

    # Calculate metrics
    logger.info("Calculating metrics...")

    # BERTScore
    bertscore_results = calculate_bertscore(predictions, references)

    # Simple metrics
    simple_metrics = calculate_simple_metrics(predictions, references)

    # Combine results
    results = {
        "model_path": model_path,
        "num_samples": num_samples,
        "bertscore": bertscore_results,
        "simple_metrics": simple_metrics,
        "examples": [
            {
                "file": test_dataset[i].get('old_file', ''),
                "reference": references[i],
                "prediction": predictions[i]
            }
            for i in range(min(5, len(predictions)))
        ]
    }

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nModel: {model_path}")
    print(f"Test samples: {num_samples}")

    if bertscore_results:
        print("\nBERTScore:")
        print(f"  Precision: {bertscore_results['precision']:.4f}")
        print(f"  Recall:    {bertscore_results['recall']:.4f}")
        print(f"  F1:        {bertscore_results['f1']:.4f}")

    print("\nSimple Metrics:")
    print(f"  Exact Match:      {simple_metrics['exact_match']:.2%}")
    print(f"  Avg Similarity:   {simple_metrics['avg_similarity']:.4f}")
    print(f"  Median Similarity: {simple_metrics['median_similarity']:.4f}")

    print("\nExample Predictions:")
    for i, example in enumerate(results['examples'], 1):
        print(f"\n{i}. File: {example['file']}")
        print(f"   Reference:  {example['reference']}")
        print(f"   Prediction: {example['prediction']}")

    print("\n" + "="*60)

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned commit message model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--test_dataset", type=str, default=None,
                       help="Path to test dataset JSONL file (optional)")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of test samples")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output JSON file for results")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        test_dataset_file=args.test_dataset,
        num_samples=args.num_samples,
        output_file=args.output
    )
