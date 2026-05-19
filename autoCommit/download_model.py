"""
Download Gemma 2B model to local models directory
"""
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model(
    model_name="google/gemma-2-2b",
    output_dir="models"
):
    """
    Download model and tokenizer to local directory

    Args:
        model_name: HuggingFace model name
        output_dir: Local directory to save the model
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading model: {model_name}")
    logger.info(f"Output directory: {output_path.absolute()}")

    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Tokenizer saved to {output_dir}")

        # Download model
        logger.info("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        model.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

        logger.info("Download completed successfully!")
        logger.info(f"Model location: {output_path.absolute()}")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")

        return output_dir

    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        logger.info("\nTroubleshooting tips:")
        logger.info("1. Make sure you're logged in to Hugging Face: huggingface-cli login")
        logger.info("2. Accept the Gemma license at: https://huggingface.co/google/gemma-2-2b")
        logger.info("3. Check your internet connection")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Gemma model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Output directory to save the model"
    )

    args = parser.parse_args()

    download_model(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
