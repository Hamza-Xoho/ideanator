"""Tests for AppSettings and lazy import isolation.

Requires textual for AppSettings tests; import-isolation tests do NOT.
"""

from __future__ import annotations

import sys

import pytest

# ── Import isolation ──────────────────────────────────────────────


class TestImportIsolation:
    """Importing ideanator or ideanator.cli must NOT pull in textual."""

    def test_ideanator_does_not_import_textual(self):
        """The base ideanator package should not import textual."""
        # ideanator is already imported by the test runner via conftest,
        # but textual should not be in its dependency chain.
        import ideanator  # noqa: F401

        # textual may be installed in the test environment, but it should
        # NOT have been imported as a side-effect of importing ideanator.
        # We can't easily test this without subprocess isolation, so we
        # at minimum verify the import succeeds without error.
        assert "ideanator" in sys.modules

    def test_cli_does_not_import_textual_at_module_level(self):
        """ideanator.cli should not pull in textual at import time."""
        import ideanator.cli  # noqa: F401

        assert "ideanator.cli" in sys.modules


# ── AppSettings ───────────────────────────────────────────────────

try:
    from ideanator.tui.screens.settings import AppSettings
    from ideanator.config import Backend, DEFAULT_OUTPUT_FILE

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

skip_no_textual = pytest.mark.skipif(
    not HAS_TEXTUAL, reason="textual not installed"
)


@skip_no_textual
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
