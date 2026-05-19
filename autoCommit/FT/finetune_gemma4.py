"""
Fine-tune Gemma 4 E2B model on CommitPackFT dataset for commit message generation
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    old_contents = sample.get('old_contents', '')
    new_contents = sample.get('new_contents', '')
    subject = sample.get('subject', '')
    message = sample.get('message', '')

    # Create a diff-like representation (simplified)
    prompt = f"""### Instruction:
Generate a commit message for the following code changes.

### Code Changes:
Old file: {sample.get('old_file', '')}
New file: {sample.get('new_file', '')}

### Diff:
OLD:
{old_contents[:500]}

NEW:
{new_contents[:500]}

### Commit Message:
{subject}

{message}"""

    return prompt


def preprocess_dataset(dataset, tokenizer, max_length=512):
    """Preprocess dataset by tokenizing"""
    def tokenize_function(examples):
        prompts = [format_prompt(sample) for sample in examples]
        return tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    # Convert to list of dicts format
    samples = []
    for i in range(len(dataset['train'])):
        samples.append(dataset['train'][i])

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda examples: tokenize_function([examples]),
        batched=False,
        remove_columns=dataset['train'].column_names,
    )

    return tokenized


def setup_lora_model(model_name="google/gemma-4-E2B", use_8bit=True):
    """Setup Gemma 4 model with LoRA for efficient fine-tuning"""
    logger.info(f"Loading model: {model_name}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Tokenizer loaded successfully")

        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load model with appropriate settings
        if use_8bit and torch.cuda.is_available():
            logger.info("Loading model with 8-bit quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
        else:
            logger.info("Loading model in FP32 (CPU mode or non-quantized)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            if device == "cuda":
                model = model.to(device)

        logger.info("Model loaded successfully")

        # Configure LoRA for Gemma 4
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("\nTroubleshooting tips:")
        logger.info("1. Make sure you're logged in to Hugging Face: huggingface-cli login")
        logger.info("2. Accept the Gemma license at: https://huggingface.co/google/gemma-4-E2B")
        logger.info("3. Check your internet connection")
        raise


def train_model(
    model_name="google/gemma-4-E2B",
    output_dir="../models/gemma-4-E2B-commitpack-ft",
    num_samples=1000,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
):
    """Main training function for Gemma 4 E2B"""

    # Load dataset
    dataset = load_commitpack_dataset(num_samples=num_samples)

    # Setup model and tokenizer
    use_8bit = torch.cuda.is_available()
    model, tokenizer = setup_lora_model(model_name, use_8bit=use_8bit)

    # Preprocess dataset
    tokenized_dataset = preprocess_dataset(dataset, tokenizer)

    # Split into train and validation
    train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")

    # Training arguments
    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=50,
        save_strategy="no",  # Disable checkpoint saving to save disk space
        eval_strategy="no",  # Disable evaluation during training to save disk space
        save_total_limit=1,  # Keep only 1 checkpoint if saving
        fp16=use_cuda,  # Only use FP16 with CUDA
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 E2B on CommitPackFT")
    parser.add_argument("--model_name", type=str, default="google/gemma-4-E2B", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="../models/gemma-4-E2B-commitpack-ft", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to use")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
