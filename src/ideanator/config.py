"""Configuration constants for the ARISE pipeline."""

from dataclasses import dataclass


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


DEFAULT_MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_SERVER_URL = "http://localhost:8080/v1"
DEFAULT_OUTPUT_FILE = "arise_results.json"
SERVER_STARTUP_TIMEOUT = 120

# Safety net: ideas under this word count with "NONE" score get overridden
VAGUENESS_WORD_THRESHOLD = 20

TEMPERATURES = TemperatureConfig()
TOKENS = TokenConfig()
