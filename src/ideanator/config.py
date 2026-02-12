"""Configuration management using Pydantic Settings.

Supports:
- Environment variables (IDEANATOR_* prefix)
- .env file loading
- Type validation
- Default values

Preserves backward compatibility with existing imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ideanator.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# ── Temperature & Token Configs (frozen dataclasses for compatibility) ──


@dataclass(frozen=True)
class TemperatureConfig:
    """Temperature strategy for different LLM call types."""

    decision: float = 0.0
    questioning: float = 0.6
    synthesis: float = 0.3
    simulation: float = 0.7


@dataclass(frozen=True)
class TokenConfig:
    """Max token limits for different LLM call types."""

    decision: int = 200
    question: int = 250
    synthesis: int = 500
    simulation: int = 200


TEMPERATURES = TemperatureConfig()
TOKENS = TokenConfig()


# ── Backend Enum & Config ────────────────────────────────────────────


class Backend(str, Enum):
    """Supported LLM server backends."""

    MLX = "mlx"
    OLLAMA = "ollama"
    EXTERNAL = "external"


@dataclass(frozen=True)
class BackendConfig:
    """Default settings for each backend."""

    default_model: str
    default_url: str
    needs_server: bool


BACKEND_DEFAULTS: dict[Backend, BackendConfig] = {
    Backend.MLX: BackendConfig(
        default_model="mlx-community/Llama-3.2-3B-Instruct-4bit",
        default_url="http://localhost:8080/v1",
        needs_server=True,
    ),
    Backend.OLLAMA: BackendConfig(
        default_model="llama3.2:3b",
        default_url="http://localhost:11434/v1",
        needs_server=True,
    ),
    Backend.EXTERNAL: BackendConfig(
        default_model="default",
        default_url="http://localhost:8080/v1",
        needs_server=False,
    ),
}


def get_backend_config(backend: Backend) -> BackendConfig:
    """Get the default configuration for a backend."""
    return BACKEND_DEFAULTS[backend]


# ── Legacy aliases ───────────────────────────────────────────────────

DEFAULT_MODEL_ID = BACKEND_DEFAULTS[Backend.MLX].default_model
DEFAULT_SERVER_URL = BACKEND_DEFAULTS[Backend.MLX].default_url
DEFAULT_OUTPUT_FILE = "arise_results.json"
SERVER_STARTUP_TIMEOUT = 120
VAGUENESS_WORD_THRESHOLD = 20


# ── Pydantic Settings ───────────────────────────────────────────────


class Settings(BaseSettings):
    """Application settings loaded from environment and .env file.

    Environment variables are prefixed with IDEANATOR_
    Example: IDEANATOR_OLLAMA_URL=http://localhost:11434
    """

    model_config = SettingsConfigDict(
        env_prefix="IDEANATOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Ollama settings
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_model: str = Field(
        default="qwen2.5:7b",
        description="Ollama model name",
    )

    # MLX settings (Apple Silicon only)
    mlx_model: str = Field(
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        description="MLX model identifier",
    )
    mlx_port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="MLX server port",
    )

    # External API settings
    external_url: str = Field(
        default="http://localhost:8000",
        description="External OpenAI-compatible API URL",
    )
    external_api_key: str | None = Field(
        default=None,
        description="API key for external service (if required)",
    )

    # Application settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    output_dir: Path = Field(
        default_factory=lambda: Path.home() / ".local" / "share" / "ideanator",
        description="Directory for output files",
    )

    config_dir: Path = Field(
        default_factory=lambda: Path.home() / ".config" / "ideanator",
        description="Directory for configuration files",
    )

    @field_validator("ollama_url", "external_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URLs have protocol."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"URL must start with http:// or https://, got: {v}")
        return v

    @field_validator("output_dir", "config_dir")
    @classmethod
    def validate_directory(cls, v: Path) -> Path:
        """Ensure directories exist or can be created."""
        try:
            v.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create directory {v}: {e}") from e
        return v

    def get_backend_url(self, backend: Backend) -> str:
        """Get URL for specified backend."""
        if backend == Backend.OLLAMA:
            return self.ollama_url
        elif backend == Backend.MLX:
            return f"http://localhost:{self.mlx_port}"
        elif backend == Backend.EXTERNAL:
            return self.external_url
        else:
            raise ConfigurationError(f"Unknown backend: {backend}")

    def get_backend_model(self, backend: Backend) -> str:
        """Get model name for specified backend."""
        if backend == Backend.OLLAMA:
            return self.ollama_model
        elif backend == Backend.MLX:
            return self.mlx_model
        elif backend == Backend.EXTERNAL:
            return "gpt-3.5-turbo"
        else:
            raise ConfigurationError(f"Unknown backend: {backend}")


# Global settings instance (loaded once at startup)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        try:
            _settings = Settings()
            logger.debug("Settings loaded: %s", _settings.model_dump())
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load settings: {e}",
                details={"error": str(e)},
            ) from e
    return _settings


def reload_settings() -> Settings:
    """Force reload settings (useful for testing)."""
    global _settings
    _settings = None
    return get_settings()
