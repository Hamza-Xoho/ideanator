"""Configuration constants for the ARISE pipeline."""

from dataclasses import dataclass
from enum import Enum


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


# Legacy aliases for backwards compatibility
DEFAULT_MODEL_ID = BACKEND_DEFAULTS[Backend.MLX].default_model
DEFAULT_SERVER_URL = BACKEND_DEFAULTS[Backend.MLX].default_url

DEFAULT_OUTPUT_FILE = "arise_results.json"
SERVER_STARTUP_TIMEOUT = 120

# Safety net: ideas under this word count with "NONE" score get overridden
VAGUENESS_WORD_THRESHOLD = 20

TEMPERATURES = TemperatureConfig()
TOKENS = TokenConfig()
