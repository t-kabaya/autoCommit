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
        use_4bit: bool = False,  # Enable 4-bit quantization
        **kwargs  # Accept but ignore llama_cli_path for compatibility
    ) -> None:
        """Initialize TransformersLLM.

        Args:
            model_path: HuggingFace model ID or local path
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_4bit: Use 4-bit quantization (faster, less memory)
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self.device = None

    def _is_model_cached(self) -> bool:
        """Check if model is already cached locally.

        Returns:
            True if model is cached, False otherwise
        """
        try:
            from transformers import AutoTokenizer
            # Try to load with local_files_only to check cache
            AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            return True
        except Exception:
            return False

    def _load_model(self) -> None:
        """Load model and tokenizer lazily."""
        if self.model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # Check if model is cached
            use_local_only = self._is_model_cached()

            # Load tokenizer
            # trust_remote_code: Allow execution of custom code in the model
            # local_files_only: Use cache if available, download if not
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=use_local_only
            )

            # Prepare model loading arguments
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            # 4-bit quantization
            if self.use_4bit:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
                device = "cuda" if torch.cuda.is_available() else "mps"
            else:
                # Determine device for non-quantized
                if torch.backends.mps.is_available():
                    device = "mps"
                    dtype = torch.float16
                elif torch.cuda.is_available():
                    device = "cuda"
                    dtype = torch.float16
                else:
                    device = "cpu"
                    dtype = torch.float32

                model_kwargs["dtype"] = dtype
                model_kwargs["device_map"] = device

            # Load model (use cache if available, download if not)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=use_local_only,
                **model_kwargs
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

            # Don't use chat template - use simple instruction format
            # Chat template causes Gemma 3 to not generate properly
            formatted_prompt = prompt

            if verbose:
                print(f"[DEBUG] Using simple prompt format (no chat template)")

            # Tokenize input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")

            if self.device in ["mps", "cuda"]:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            import torch
            with torch.no_grad():
                generation_config = {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "do_sample": True if self.temperature > 0 else False,
                    "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
                }

                # Don't set eos_token_id explicitly to avoid premature termination
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )

            # Decode output
            if verbose:
                print(f"[DEBUG] Input token length: {inputs['input_ids'].shape[1]}")
                print(f"[DEBUG] Output token length: {outputs.shape[1]}")
                print(f"[DEBUG] New tokens generated: {outputs.shape[1] - inputs['input_ids'].shape[1]}")

            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            full_text_with_tokens = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            if verbose:
                print(f"[DEBUG] Full output (with special tokens):\n{full_text_with_tokens}\n")
                print(f"[DEBUG] Full output (without special tokens):\n{full_text}\n")

            # Extract generated text by removing the prompt
            if full_text.startswith(formatted_prompt):
                generated_text = full_text[len(formatted_prompt):].strip()
            elif full_text.startswith(prompt):
                generated_text = full_text[len(prompt):].strip()
            else:
                generated_text = full_text.strip()

            if verbose:
                print(f"[DEBUG] After prompt removal:\n{generated_text}\n")

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

        # Remove markdown code blocks first
        if "```" in output:
            # Extract content between code blocks or remove them
            parts = output.split("```")
            if len(parts) >= 2:
                # Get the content between first ``` and second ```
                output = parts[1].strip() if len(parts) >= 3 else parts[0].strip()

        # Take first line (commit messages are single line)
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        if lines:
            output = lines[0]

        # Remove list markers (-, *, 1., etc.)
        import re
        output = re.sub(r'^[-*]\s+', '', output)
        output = re.sub(r'^\d+\.\s+', '', output)

        # Remove quotes
        output = output.strip('"\'`')

        # Remove common prefixes
        prefixes = ["commit message:", "commit:", "message:", "answer:", "response:", "type:", "description:"]
        for prefix in prefixes:
            if output.lower().startswith(prefix):
                output = output[len(prefix):].strip()

        # Add conventional commit type if missing
        types = ["feat", "fix", "docs", "style", "refactor", "test", "chore"]
        has_type = any(output.lower().startswith(f"{t}:") for t in types)

        if not has_type:
            # Infer type from keywords in message
            output_lower = output.lower()
            if "add" in output_lower or "new" in output_lower or "create" in output_lower:
                output = f"feat: {output}"
            elif "fix" in output_lower or "bug" in output_lower or "resolve" in output_lower:
                output = f"fix: {output}"
            elif "update" in output_lower or "change" in output_lower or "modify" in output_lower:
                output = f"refactor: {output}"
            else:
                output = f"chore: {output}"

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
