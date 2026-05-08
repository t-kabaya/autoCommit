"""LLM integration using llama.cpp."""

import subprocess
import re
from pathlib import Path
from typing import List, Optional


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    pass


class LlamaLLM:
    """LLM interface using llama.cpp subprocess."""

    def __init__(
        self,
        model_path: Path,
        llama_cli_path: Path,
        temperature: float = 0.2,
        max_tokens: int = 64,
    ) -> None:
        """Initialize LlamaLLM.

        Args:
            model_path: Path to GGUF model file
            llama_cli_path: Path to llama-cli binary
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.llama_cli_path = llama_cli_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.model_path.exists():
            raise LLMError(f"Model not found at {self.model_path}")
        if not self.llama_cli_path.exists():
            raise LLMError(f"llama-cli not found at {self.llama_cli_path}")

    def generate(self, prompt: str, verbose: bool = False) -> str:
        """Generate text using llama.cpp.

        Args:
            prompt: Input prompt
            verbose: Whether to show debug output

        Returns:
            Generated text

        Raises:
            LLMError: If generation fails
        """
        try:
            cmd = [
                str(self.llama_cli_path),
                "-m",
                str(self.model_path),
                "-p",
                prompt,
                "-n",
                str(self.max_tokens),
                "--temp",
                str(self.temperature),
                "--no-display-prompt",
            ]

            if verbose:
                print(f"\n[DEBUG] Running command: {' '.join(cmd)}\n")
                print(f"[DEBUG] Prompt:\n{prompt}\n")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise LLMError(f"llama-cli failed: {result.stderr}")

            output = result.stdout.strip()

            if verbose:
                print(f"[DEBUG] Raw output:\n{output}\n")

            # Clean up output
            output = self._clean_output(output)

            if not output:
                raise LLMError("No output generated from LLM")

            return output

        except subprocess.TimeoutExpired:
            raise LLMError("LLM generation timed out")
        except FileNotFoundError:
            raise LLMError(f"llama-cli not found at {self.llama_cli_path}")
        except Exception as e:
            raise LLMError(f"LLM generation failed: {str(e)}") from e

    def _clean_output(self, output: str) -> str:
        """Clean LLM output to extract commit message.

        Args:
            output: Raw LLM output

        Returns:
            Cleaned commit message
        """
        # Remove common prefixes/suffixes
        output = output.strip()

        # Remove any leading/trailing quotes
        output = output.strip('"\'')

        # Take first line if multiple lines (for single commit message)
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        if lines:
            output = lines[0]

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
        # Generate with higher max_tokens for multiple candidates
        original_max_tokens = self.max_tokens
        self.max_tokens = max(128, num_candidates * 80)

        try:
            output = self.generate(prompt, verbose=verbose)
            candidates = self._parse_candidates(output, num_candidates)

            if not candidates:
                # Fallback: generate single message
                self.max_tokens = original_max_tokens
                single = self.generate(
                    prompt.replace(f"Generate {num_candidates}", "Generate ONE"), verbose
                )
                candidates = [single]

            return candidates

        finally:
            self.max_tokens = original_max_tokens

    def _parse_candidates(self, output: str, expected: int) -> List[str]:
        """Parse multiple candidates from output.

        Args:
            output: Raw LLM output
            expected: Expected number of candidates

        Returns:
            List of parsed candidates
        """
        candidates = []
        lines = output.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering like "1. ", "2. ", etc.
            line = re.sub(r"^\d+[\.\)]\s*", "", line)

            # Remove quotes
            line = line.strip('"\'')

            # Skip non-commit-like lines
            if len(line) < 10 or len(line) > 100:
                continue

            # Check if looks like a commit message
            if ":" in line or any(
                line.startswith(t)
                for t in ["feat", "fix", "docs", "style", "refactor", "test", "chore"]
            ):
                candidates.append(line)

            if len(candidates) >= expected:
                break

        return candidates
