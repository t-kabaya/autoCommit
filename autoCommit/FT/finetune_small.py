"""
Fine-tune a smaller model (GPT-2) on CommitPackFT dataset for testing
This is a lighter alternative when disk space or compute is limited
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


def load_commitpack_dataset(num_samples=None):
    """Load CommitPackFT dataset (Python language)"""
    logger.info("Loading CommitPackFT dataset...")
    dataset = load_dataset(
        "json",
        data_files="hf://datasets/bigcode/commitpackft/data/python/data.jsonl"
    )

    if num_samples:
        dataset['train'] = dataset['train'].select(range(min(num_samples, len(dataset['train']))))

    logger.info(f"Loaded {len(dataset['train'])} samples")
    return dataset


def format_prompt(sample):
    """Format a sample into a prompt for commit message generation"""
    subject = sample.get('subject', '')
    message = sample.get('message', '')
    old_file = sample.get('old_file', '')

    # Simplified prompt for smaller models
    prompt = f"File: {old_file}\nCommit: {subject}\n{message}"
    return prompt


def preprocess_dataset(dataset, tokenizer, max_length=256):
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
    model_name="gpt2",
    output_dir="../models/gpt2-commitpack-ft",
    num_samples=100,
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
):
    """Main training function"""
    logger.info(f"Loading model: {model_name}")

    # Load dataset
    dataset = load_commitpack_dataset(num_samples=num_samples)

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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=50,
        save_strategy="no",  # Disable checkpoint saving to save disk space
        eval_strategy="no",  # Disable evaluation during training to save disk space
        save_total_limit=1,  # Keep only 1 checkpoint if saving
        push_to_hub=False,
        report_to=[],  # Disable tensorboard to save disk space
        load_best_model_at_end=False,  # Disable to save disk space
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

    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on CommitPackFT")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name (gpt2, gpt2-medium, etc.)")
    parser.add_argument("--output_dir", type=str, default="../models/gpt2-commitpack-ft", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to use")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")

    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
