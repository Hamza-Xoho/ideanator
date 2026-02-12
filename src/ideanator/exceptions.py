"""Custom exceptions for ideanator.

All exceptions inherit from IdeanatorError for easy catching.
Each exception includes context in its message.
"""

from __future__ import annotations


class IdeanatorError(Exception):
    """Base exception for all ideanator errors."""

    def __init__(self, message: str, details: dict[str, str] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ConfigurationError(IdeanatorError):
    """Configuration loading or validation failed."""


class ServerError(IdeanatorError):
    """LLM server start/stop/communication failed."""


class ValidationError(IdeanatorError):
    """Input validation failed."""


class PromptLoadError(IdeanatorError):
    """Failed to load prompt templates."""


class RefactoringError(IdeanatorError):
    """Refactoring pipeline failed."""


class ParseError(IdeanatorError):
    """Failed to parse LLM response."""
