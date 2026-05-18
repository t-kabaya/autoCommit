"""
Test the fine-tuned model for commit message generation
"""
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def test_model(model_path, prompt=None):
    """Test the fine-tuned model with a sample prompt"""
    print(f"Loading model from: {model_path}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Set model to evaluation mode
    model.eval()

    print("Model loaded successfully!\n")

    # Default prompt if none provided
    if prompt is None:
        prompt = "File: test.py\nCommit: Fix bug in"

    print(f"Input prompt:\n{prompt}\n")
    print("=" * 50)
    print("Generated commit message:")
    print("=" * 50)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    print("=" * 50)


def interactive_mode(model_path):
    """Interactive mode for testing the model"""
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    print("\nModel loaded! Type 'quit' to exit.\n")

    while True:
        prompt = input("Enter prompt (or 'quit'): ")
        if prompt.lower() == 'quit':
            break

        if not prompt.strip():
            prompt = "File: main.py\nCommit: "

        print("\nGenerating...")
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n{generated_text}\n")
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fine-tuned commit message model")
    parser.add_argument("--model_path", type=str, default="../models/gpt2-commitpack-test",
                       help="Path to fine-tuned model")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Test prompt")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.model_path)
    else:
        test_model(args.model_path, args.prompt)
