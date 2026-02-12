"""Tests for AppSettings dataclass."""

from __future__ import annotations

import pytest

from ideanator.tui.screens.settings import AppSettings
from ideanator.config import Backend, DEFAULT_OUTPUT_FILE


class TestAppSettings:
    """AppSettings defaults and overrides."""

    def test_defaults(self):
        s = AppSettings()
        assert s.backend == Backend.OLLAMA
        assert s.model == ""
        assert s.server_url == ""
        assert s.batch_file == ""
        assert s.output_file == DEFAULT_OUTPUT_FILE
        assert s.verbose is False

    def test_override_all(self):
        s = AppSettings(
            backend=Backend.MLX,
            model="my-model",
            server_url="http://localhost:1234/v1",
            batch_file="ideas.json",
            output_file="out.json",
            verbose=True,
        )
        assert s.backend == Backend.MLX
        assert s.model == "my-model"
        assert s.server_url == "http://localhost:1234/v1"
        assert s.batch_file == "ideas.json"
        assert s.output_file == "out.json"
        assert s.verbose is True

    def test_partial_override(self):
        s = AppSettings(backend=Backend.EXTERNAL, verbose=True)
        assert s.backend == Backend.EXTERNAL
        assert s.verbose is True
        assert s.model == ""
        assert s.server_url == ""

    def test_is_dataclass(self):
        """AppSettings should be a mutable dataclass."""
        s = AppSettings()
        s.model = "changed"
        assert s.model == "changed"
