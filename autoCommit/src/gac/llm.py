"""LLM integration using transformers."""

from pathlib import Path
from typing import List, Optional


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    pass


class TransformersLLM:
    """LLM interface using transformers library."""

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.2,
        max_tokens: int = 64,
        **kwargs  # Accept but ignore llama_cli_path for compatibility
    ) -> None:
        """Initialize TransformersLLM.

        Args:
            model_path: HuggingFace model ID or local path
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None
        self.device = None

    def _load_model(self) -> None:
        """Load model and tokenizer lazily."""
        if self.model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # Determine device
            if torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float16
            elif torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            else:
                device = "cpu"
                dtype = torch.float32

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            self.device = device

        except Exception as e:
            raise LLMError(f"Failed to load model: {str(e)}") from e

    def generate(self, prompt: str, verbose: bool = False) -> str:
        """Generate text using transformers.

        Args:
            prompt: Input prompt
            verbose: Whether to show debug output

        Returns:
            Generated text

        Raises:
            LLMError: If generation fails
        """
        try:
            self._load_model()

            if verbose:
                print(f"\n[DEBUG] Model: {self.model_path}")
                print(f"[DEBUG] Device: {self.device}")
                print(f"[DEBUG] Prompt:\n{prompt}\n")

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")

            if self.device in ["mps", "cuda"]:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            if verbose:
                print(f"[DEBUG] Raw output:\n{generated_text}\n")

            # Clean output
            output = self._clean_output(generated_text)

            if not output:
                raise LLMError("No output generated from LLM")

            return output

        except Exception as e:
            raise LLMError(f"LLM generation failed: {str(e)}") from e

    def _clean_output(self, output: str) -> str:
        """Clean LLM output to extract commit message.

        Args:
            output: Raw LLM output

        Returns:
            Cleaned commit message
        """
        # Remove leading/trailing whitespace
        output = output.strip()

        # Take first line (commit messages are single line)
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        if lines:
            output = lines[0]

        # Remove quotes
        output = output.strip('"\'`')

        # Remove common prefixes
        prefixes = ["commit message:", "message:", "answer:", "response:"]
        for prefix in prefixes:
            if output.lower().startswith(prefix):
                output = output[len(prefix):].strip()

        # Remove markdown code blocks
        if output.startswith("```") and output.endswith("```"):
            output = output[3:-3].strip()

        return output

    def generate_candidates(
        self, prompt: str, num_candidates: int = 3, verbose: bool = False
    ) -> List[str]:
        """Generate multiple commit message candidates.

        Args:
            prompt: Input prompt
            num_candidates: Number of candidates to generate
            verbose: Whether to show debug output

        Returns:
            List of commit message candidates

        Raises:
            LLMError: If generation fails
        """
        candidates = []

        # Generate multiple candidates with different temperatures
        original_temp = self.temperature
        temps = [0.1, 0.3, 0.5][:num_candidates]

        for i, temp in enumerate(temps):
            self.temperature = temp
            try:
                candidate = self.generate(prompt, verbose=verbose and i == 0)
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
            except Exception:
                pass

        self.temperature = original_temp

        # If we don't have enough candidates, return what we have
        if not candidates:
            # Fallback: try once more with original temperature
            candidates = [self.generate(prompt, verbose)]

        return candidates[:num_candidates]


# Alias for backward compatibility
LlamaLLM = TransformersLLM
