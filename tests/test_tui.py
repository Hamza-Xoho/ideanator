"""Tests for the --tui CLI flag integration.

These tests verify that ``ideanator --tui`` dispatches correctly to
``_launch_tui()`` and that CLI flags are forwarded.  They do NOT require
textual — ``_launch_tui`` is always mocked.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ideanator.cli import main
from ideanator.config import Backend


class TestTuiFlagHelp:
    """The --tui option appears in help output."""

    def test_help_shows_tui_option(self):
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--tui" in result.output

    def test_help_shows_tui_examples(self):
        result = CliRunner().invoke(main, ["--help"])
        assert "Terminal UI" in result.output
        assert "ideanator --tui" in result.output


class TestTuiDispatch:
    """``--tui`` dispatches to ``_launch_tui`` and does NOT start the
    normal CLI pipeline."""

    @patch("ideanator.cli._launch_tui")
    def test_tui_flag_calls_launch_tui(self, mock_launch):
        result = CliRunner().invoke(main, ["--tui"])
        assert result.exit_code == 0
        mock_launch.assert_called_once()

    @patch("ideanator.cli._launch_tui")
    def test_tui_does_not_run_normal_pipeline(self, mock_launch):
        """Normal pipeline code should not be reached when --tui is set."""
        with patch("ideanator.cli._dispatch") as mock_dispatch:
            CliRunner().invoke(main, ["--tui"])
            mock_dispatch.assert_not_called()

    @patch("ideanator.cli._launch_tui")
    def test_tui_without_flag_does_not_call_launch_tui(self, mock_launch):
        """When --tui is omitted, _launch_tui should not be called."""
        with patch("ideanator.cli._dispatch"):
            with patch("ideanator.cli._resolve_backend", return_value=Backend.EXTERNAL):
                with patch("ideanator.cli.get_backend_config") as mock_cfg:
                    mock_cfg.return_value.needs_server = False
                    mock_cfg.return_value.default_model = "default"
                    mock_cfg.return_value.default_url = "http://localhost:8080/v1"
                    CliRunner().invoke(main, ["--external"])
                    mock_launch.assert_not_called()


class TestTuiFlagForwarding:
    """CLI flags are forwarded to ``_launch_tui`` with correct values."""

    @patch("ideanator.cli._launch_tui")
    def test_defaults(self, mock_launch):
        CliRunner().invoke(main, ["--tui"])
        _, kwargs = mock_launch.call_args
        assert kwargs["use_ollama"] is False
        assert kwargs["use_mlx"] is False
        assert kwargs["use_external"] is False
        assert kwargs["model"] is None
        assert kwargs["server_url"] is None
        assert kwargs["file_path"] is None
        assert kwargs["output_path"] is None
        assert kwargs["verbose"] is False

    @patch("ideanator.cli._launch_tui")
    def test_ollama_backend(self, mock_launch):
        CliRunner().invoke(main, ["--tui", "--ollama"])
        _, kwargs = mock_launch.call_args
        assert kwargs["use_ollama"] is True

    @patch("ideanator.cli._launch_tui")
    def test_mlx_backend(self, mock_launch):
        CliRunner().invoke(main, ["--tui", "--mlx"])
        _, kwargs = mock_launch.call_args
        assert kwargs["use_mlx"] is True

    @patch("ideanator.cli._launch_tui")
    def test_external_backend(self, mock_launch):
        CliRunner().invoke(main, ["--tui", "--external"])
        _, kwargs = mock_launch.call_args
        assert kwargs["use_external"] is True

    @patch("ideanator.cli._launch_tui")
    def test_model_flag(self, mock_launch):
        CliRunner().invoke(main, ["--tui", "-m", "qwen2.5:7b"])
        _, kwargs = mock_launch.call_args
        assert kwargs["model"] == "qwen2.5:7b"

    @patch("ideanator.cli._launch_tui")
    def test_server_url_flag(self, mock_launch):
        CliRunner().invoke(main, ["--tui", "--server-url", "http://example.com/v1"])
        _, kwargs = mock_launch.call_args
        assert kwargs["server_url"] == "http://example.com/v1"

    @patch("ideanator.cli._launch_tui")
    def test_verbose_flag(self, mock_launch):
        CliRunner().invoke(main, ["--tui", "-v"])
        _, kwargs = mock_launch.call_args
        assert kwargs["verbose"] is True

    @patch("ideanator.cli._launch_tui")
    def test_all_flags_combined(self, mock_launch):
        CliRunner().invoke(main, [
            "--tui", "--ollama",
            "-m", "llama3:8b",
            "--server-url", "http://localhost:11434/v1",
            "-v",
        ])
        _, kwargs = mock_launch.call_args
        assert kwargs["use_ollama"] is True
        assert kwargs["model"] == "llama3:8b"
        assert kwargs["server_url"] == "http://localhost:11434/v1"
        assert kwargs["verbose"] is True
