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

    def _load_model(self) -> None:
        """Load model and tokenizer lazily."""
        if self.model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            print('debug0.3')

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )

            print('debug0.5')

            # Prepare model loading arguments
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            print('debug1')

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

                model_kwargs["torch_dtype"] = dtype
                model_kwargs["device_map"] = device

            print('debug2')
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
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
            print('load_model')
            self._load_model()
            print('verbose')
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

"""Configuration management for gac."""

from pathlib import Path
from typing import Optional
import toml


class Config:
    """Configuration manager for gac."""

    # Available models
    MODELS = {
        "small": "google/gemma-2-2b-it",  # ~2B, balanced (~4GB)
        "medium": "google/gemma-3-3b-it",  # ~3B, powerful (~6GB)
    }

    # Fast mode: quantized model
    FAST_MODEL = "google/gemma-3-1b-it"  # 4-bit quantized (~500MB)

    DEFAULT_CONFIG = {
        "model": "google/gemma-2-2b-it",  # HuggingFace model ID
        "temperature": 0.2,
        "max_tokens": 64,
        "num_candidates": 3,
    }

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize config manager.

        Args:
            config_path: Path to config file. Defaults to ~/.gac/config.toml
        """
        if config_path is None:
            config_path = Path.home() / ".gac" / "config.toml"
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from file or return defaults.

        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            return self.DEFAULT_CONFIG.copy()

        try:
            with open(self.config_path, "r") as f:
                loaded_config = toml.load(f)
            # Merge with defaults
            config = self.DEFAULT_CONFIG.copy()
            config.update(loaded_config)
            return config
        except Exception:
            return self.DEFAULT_CONFIG.copy()

    def save(self) -> None:
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            toml.dump(self._config, f)

    def get(self, key: str, default: Optional[str] = None) -> str:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self._config.get(key, default)

    def set(self, key: str, value: str) -> None:
        """Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value

    @property
    def model_path(self) -> str:
        """Get model ID or path."""
        return self.get("model", "google/gemma-2-2b-it")

    @property
    def temperature(self) -> float:
        """Get temperature setting."""
        return float(self.get("temperature", "0.2"))

    @property
    def max_tokens(self) -> int:
        """Get max tokens setting."""
        return int(self.get("max_tokens", "64"))

    @property
    def num_candidates(self) -> int:
        """Get number of candidates to generate."""
        return int(self.get("num_candidates", "3"))

    def is_configured(self) -> bool:
        """Check if gac is properly configured.

        Returns:
            True if model is specified
        """
        return bool(self.model_path)

    def __str__(self) -> str:
        """String representation of config."""
        lines = ["Current Configuration:"]
        for key, value in self._config.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

config = Config()


llm = LlamaLLM(
    model_path=config.model_path,
    temperature=config.temperature,
    max_tokens=config.max_tokens,
)

print(llm)
            
candidates = llm.generate(
    "hi",
    True
)

print(candidates)
