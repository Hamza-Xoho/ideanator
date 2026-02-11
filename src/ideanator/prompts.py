"""Load and access prompts from the YAML configuration file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _default_prompts_path() -> Path:
    """Return the default prompts.yaml path (project root)."""
    return Path(__file__).resolve().parent.parent.parent / "prompts.yaml"


@lru_cache(maxsize=1)
def load_prompts(path: str | None = None) -> dict[str, Any]:
    """Load and cache all prompts from the YAML file."""
    target = Path(path) if path else _default_prompts_path()
    with open(target, "r") as f:
        return yaml.safe_load(f)


def clear_cache() -> None:
    """Clear the prompt cache (useful for testing)."""
    load_prompts.cache_clear()


def get_vagueness_prompt(prompts: dict | None = None) -> str:
    p = prompts or load_prompts()
    return p["vagueness_prompt"]


def get_phase_prompt_template(phase: str, prompts: dict | None = None) -> str:
    p = prompts or load_prompts()
    return p["phase_prompts"][phase]


def get_simulated_user_prompt(prompts: dict | None = None) -> str:
    p = prompts or load_prompts()
    return p["simulated_user_prompt"]


def get_synthesis_prompt(prompts: dict | None = None) -> str:
    p = prompts or load_prompts()
    return p["synthesis_prompt"]


def get_example_pool(prompts: dict | None = None) -> dict:
    p = prompts or load_prompts()
    return p["example_pool"]
