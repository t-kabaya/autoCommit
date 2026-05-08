"""Configuration management for gac."""

from pathlib import Path
from typing import Optional
import toml


class Config:
    """Configuration manager for gac."""

    DEFAULT_CONFIG = {
        "model": "~/.gac/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        "llama_cli": "~/.gac/bin/llama-cli",
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
    def model_path(self) -> Path:
        """Get model path with expansion."""
        return Path(self.get("model")).expanduser()

    @property
    def llama_cli_path(self) -> Path:
        """Get llama-cli path with expansion."""
        return Path(self.get("llama_cli")).expanduser()

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
            True if model and llama-cli exist
        """
        return self.model_path.exists() and self.llama_cli_path.exists()

    def __str__(self) -> str:
        """String representation of config."""
        lines = ["Current Configuration:"]
        for key, value in self._config.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
