"""
Fine-tune GPT-2 on CommitPackFT with git diff format
Uses actual code diffs instead of just file names for better commit message generation
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prepared_dataset(dataset_file, num_samples=None):
    """Load prepared dataset with diffs from JSONL file"""
    logger.info(f"Loading prepared dataset from {dataset_file}...")

    dataset = load_dataset("json", data_files=dataset_file)

    if num_samples:
        dataset['train'] = dataset['train'].select(range(min(num_samples, len(dataset['train']))))

    logger.info(f"Loaded {len(dataset['train'])} samples")
    return dataset


def format_prompt(sample):
    """
    Format a sample with diff into a training prompt

    Format:
    File: path/to/file.py

    Diff:
    @@ -1,3 +1,4 @@
     line1
    +line2
     line3

    Commit: Subject line
    Full commit message
    """
    file_name = sample.get('file', 'unknown.py')
    diff = sample.get('diff', '')
    subject = sample.get('subject', '')
    message = sample.get('message', '')

    # Format prompt with diff
    prompt = f"""File: {file_name}

Diff:
{diff}

Commit: {subject}
{message}"""

    return prompt


def preprocess_dataset(dataset, tokenizer, max_length=512):
    """Preprocess dataset by tokenizing"""
    def tokenize_function(examples):
        prompt = format_prompt(examples)
        return tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_function,
        remove_columns=dataset['train'].column_names,
    )

    return tokenized


def train_model(
    dataset_file="../datasets/commitpack/prepared_commitpack.jsonl",
    model_name="gpt2",
    output_dir="../models/gpt2-diff",
    num_samples=1000,
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
):
    """Main training function"""
    logger.info(f"Loading model: {model_name}")

    # Load prepared dataset with diffs
    dataset = load_prepared_dataset(dataset_file, num_samples=num_samples)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    logger.info("Model loaded successfully")

    # Preprocess dataset
    tokenized_dataset = preprocess_dataset(dataset, tokenizer)

    # Split into train and validation
    train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")

    # Training arguments - optimized for disk space
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=50,
        save_strategy="no",  # Disable checkpoint saving
        eval_strategy="no",  # Disable evaluation during training
        save_total_limit=1,
        push_to_hub=False,
        report_to=[],  # Disable tensorboard
        load_best_model_at_end=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training completed!")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on CommitPackFT with diffs")
    parser.add_argument("--dataset", type=str, default="../datasets/commitpack/prepared_commitpack.jsonl",
                       help="Path to prepared dataset JSONL file")
    parser.add_argument("--model_name", type=str, default="gpt2",
                       help="Model name (gpt2, gpt2-medium, etc.)")
    parser.add_argument("--output_dir", type=str, default="../models/gpt2-diff",
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to use")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")

    args = parser.parse_args()

    train_model(
        dataset_file=args.dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
