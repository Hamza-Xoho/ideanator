"""Tests for the CLI interface."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ideanator.cli import main, _resolve_backend
from ideanator.config import Backend


# All CLI tests mock preflight_check to avoid network calls.
_PREFLIGHT_PATCH = patch("ideanator.cli.preflight_check", return_value=True)


# ── Help & version ────────────────────────────────────────────────────


class TestCLIHelp:
    def test_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "ideanator" in result.output

    def test_help_shows_all_backends(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "--ollama" in result.output
        assert "--mlx" in result.output
        assert "--external" in result.output

    def test_help_shows_examples(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "EXAMPLES" in result.output
        assert "ideanator --ollama" in result.output
        assert "ideanator --mlx" in result.output

    def test_help_shows_options(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "-m, --model" in result.output
        assert "-f, --file" in result.output
        assert "-o, --output" in result.output
        assert "-v, --verbose" in result.output

    def test_help_shows_backend_defaults_table(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "BACKEND DEFAULTS" in result.output
        assert "llama3.2:3b" in result.output
        assert "11434" in result.output
        assert "8080" in result.output

    def test_help_shows_tui_option(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "--tui" in result.output

    def test_help_shows_input_format(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "INPUT FILE FORMAT" in result.output

    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0


# ── Backend resolution ────────────────────────────────────────────────


class TestResolveBackend:
    def test_default_is_ollama(self):
        assert _resolve_backend(False, False, False, False) == Backend.OLLAMA

    def test_ollama_flag(self):
        assert _resolve_backend(False, True, False, False) == Backend.OLLAMA

    def test_mlx_flag(self):
        assert _resolve_backend(True, False, False, False) == Backend.MLX

    def test_external_flag(self):
        assert _resolve_backend(False, False, True, False) == Backend.EXTERNAL

    def test_no_server_resolves_to_external(self):
        assert _resolve_backend(False, False, False, True) == Backend.EXTERNAL

    def test_multiple_backends_raises(self):
        import click
        with pytest.raises(click.UsageError, match="Pick only one"):
            _resolve_backend(True, True, False, False)

    def test_all_three_raises(self):
        import click
        with pytest.raises(click.UsageError, match="Pick only one"):
            _resolve_backend(True, True, True, False)


# ── Batch mode ────────────────────────────────────────────────────────


BATCH_MOCK_RESPONSES = [
    "NONE",
    "[REFLECTION] R\n[QUESTION 1] Q1?\n[QUESTION 2] Q2?",
    "Simulated response.",
    "[REFLECTION] R\n[QUESTION 1] Q1?\n[QUESTION 2] Q2?",
    "Simulated response.",
    "[REFLECTION] R\n[QUESTION 1] Q1?\n[QUESTION 2] Q2?",
    "Simulated response.",
    "[REFLECTION] R\n[QUESTION 1] Q1?\n[QUESTION 2] Q2?",
    "Simulated response.",
    "Synthesis output.",
]


class TestCLIBatchMode:
    def test_file_not_found(self):
        runner = CliRunner()
        result = runner.invoke(
            main, ["-f", "/nonexistent/path.json", "--external"]
        )
        assert result.exit_code != 0

    def test_batch_with_external(self):
        """ideanator --external -f ideas.json -o results.json"""
        ideas = {"ideas": [{"content": "I want to build a test app."}]}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(ideas, f)
            input_path = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            output_path = f.name

        from tests.conftest import MockLLMClient

        mock_client = MockLLMClient(BATCH_MOCK_RESPONSES)

        runner = CliRunner()
        with patch("ideanator.cli.OpenAILocalClient", return_value=mock_client), \
             _PREFLIGHT_PATCH:
            result = runner.invoke(
                main,
                ["--external", "-f", input_path, "-o", output_path],
            )

        assert result.exit_code == 0
        assert "PIPELINE COMPLETE" in result.output

        with open(output_path) as f:
            results = json.load(f)
        assert len(results) == 1
        assert results[0]["original_idea"] == "I want to build a test app."

        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)

    def test_batch_with_no_server_compat(self):
        """--no-server still works for backwards compatibility."""
        ideas = {"ideas": [{"content": "I want to build a test app."}]}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(ideas, f)
            input_path = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            output_path = f.name

        from tests.conftest import MockLLMClient

        mock_client = MockLLMClient(BATCH_MOCK_RESPONSES)

        runner = CliRunner()
        with patch("ideanator.cli.OpenAILocalClient", return_value=mock_client), \
             _PREFLIGHT_PATCH:
            result = runner.invoke(
                main,
                ["--no-server", "-f", input_path, "-o", output_path],
            )

        assert result.exit_code == 0
        assert "PIPELINE COMPLETE" in result.output

        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


# ── Batch validation ─────────────────────────────────────────────────


class TestCLIBatchValidation:
    def test_invalid_json(self):
        """Malformed JSON should produce a clear error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{not valid json")
            input_path = f.name

        runner = CliRunner()
        with _PREFLIGHT_PATCH:
            result = runner.invoke(
                main, ["--external", "-f", input_path]
            )

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output or "Error" in result.output
        Path(input_path).unlink(missing_ok=True)

    def test_missing_ideas_key(self):
        """JSON without 'ideas' key should error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"items": []}, f)
            input_path = f.name

        runner = CliRunner()
        with _PREFLIGHT_PATCH:
            result = runner.invoke(
                main, ["--external", "-f", input_path]
            )

        assert result.exit_code != 0
        Path(input_path).unlink(missing_ok=True)

    def test_empty_ideas_list(self):
        """Empty ideas list should exit gracefully."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"ideas": []}, f)
            input_path = f.name

        runner = CliRunner()
        with _PREFLIGHT_PATCH:
            result = runner.invoke(
                main, ["--external", "-f", input_path]
            )

        assert "No ideas found" in result.output
        Path(input_path).unlink(missing_ok=True)

    def test_entry_missing_content(self):
        """Entry without 'content' key should error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"ideas": [{"text": "something"}]}, f)
            input_path = f.name

        runner = CliRunner()
        with _PREFLIGHT_PATCH:
            result = runner.invoke(
                main, ["--external", "-f", input_path]
            )

        assert result.exit_code != 0
        Path(input_path).unlink(missing_ok=True)

    def test_entry_empty_content(self):
        """Entry with empty content should error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"ideas": [{"content": "  "}]}, f)
            input_path = f.name

        runner = CliRunner()
        with _PREFLIGHT_PATCH:
            result = runner.invoke(
                main, ["--external", "-f", input_path]
            )

        assert result.exit_code != 0
        assert "empty content" in result.output
        Path(input_path).unlink(missing_ok=True)


# ── Interactive mode ──────────────────────────────────────────────────


INTERACTIVE_MOCK_RESPONSES = [
    "NONE",  # vagueness (safety net triggers for short idea)
    "[REFLECTION] R\n[QUESTION 1] What sparked this?\n[QUESTION 2] A or B?",
    "[REFLECTION] R\n[QUESTION 1] What's broken?\n[QUESTION 2] Cost or quality?",
    "[REFLECTION] R\n[QUESTION 1] Perfect version?\n[QUESTION 2] Scale?",
    "[REFLECTION] R\n[QUESTION 1] Risk?\n[QUESTION 2] Smallest version?",
    "Synthesis output.",
]

INTERACTIVE_USER_INPUT = (
    "I want to build a test app.\n"
    "My personal experience.\n"
    "The tools are bad.\n"
    "It would feel amazing.\n"
    "Adoption is hard.\n"
)


class TestCLIInteractiveMode:
    def test_interactive_with_external(self):
        """ideanator --external"""
        from tests.conftest import MockLLMClient

        mock_client = MockLLMClient(INTERACTIVE_MOCK_RESPONSES)
        runner = CliRunner()
        with patch("ideanator.cli.OpenAILocalClient", return_value=mock_client), \
             _PREFLIGHT_PATCH:
            result = runner.invoke(
                main, ["--external"], input=INTERACTIVE_USER_INPUT
            )

        assert result.exit_code == 0
        assert "ARISE Pipeline" in result.output

    def test_interactive_with_no_server_compat(self):
        """--no-server backwards compat works for interactive mode."""
        from tests.conftest import MockLLMClient

        mock_client = MockLLMClient(INTERACTIVE_MOCK_RESPONSES)
        runner = CliRunner()
        with patch("ideanator.cli.OpenAILocalClient", return_value=mock_client), \
             _PREFLIGHT_PATCH:
            result = runner.invoke(
                main, ["--no-server"], input=INTERACTIVE_USER_INPUT
            )

        assert result.exit_code == 0
        assert "ARISE Pipeline" in result.output


# ── Backend defaults & overrides ──────────────────────────────────────


class TestCLIBackendDefaults:
    def test_external_uses_default_url(self):
        """--external should use the external backend's default URL."""
        from tests.conftest import MockLLMClient

        mock_client = MockLLMClient(INTERACTIVE_MOCK_RESPONSES)
        runner = CliRunner()
        with patch(
            "ideanator.cli.OpenAILocalClient", return_value=mock_client
        ) as mock_cls, _PREFLIGHT_PATCH:
            runner.invoke(main, ["--external"], input=INTERACTIVE_USER_INPUT)
            call_args = mock_cls.call_args
            assert call_args is not None
            url = call_args.kwargs.get(
                "base_url", call_args.args[0] if call_args.args else ""
            )
            assert "localhost:8080" in url

    def test_ollama_uses_ollama_url(self):
        """--ollama should use Ollama's default URL (port 11434)."""
        from tests.conftest import MockLLMClient

        mock_client = MockLLMClient(INTERACTIVE_MOCK_RESPONSES)
        runner = CliRunner()
        with patch(
            "ideanator.cli.OpenAILocalClient", return_value=mock_client
        ) as mock_cls, patch(
            "ideanator.cli.create_server"
        ) as mock_server, _PREFLIGHT_PATCH:
            mock_server.return_value.__enter__ = lambda s: s
            mock_server.return_value.__exit__ = lambda s, *a: None

            runner.invoke(main, ["--ollama"], input=INTERACTIVE_USER_INPUT)

            mock_server.assert_called_once()
            assert mock_server.call_args[0][0] == Backend.OLLAMA

            client_call = mock_cls.call_args
            url = client_call.kwargs.get(
                "base_url", client_call.args[0] if client_call.args else ""
            )
            assert "11434" in url

    def test_mlx_uses_mlx_url(self):
        """--mlx should use MLX's default URL (port 8080)."""
        from tests.conftest import MockLLMClient

        mock_client = MockLLMClient(INTERACTIVE_MOCK_RESPONSES)
        runner = CliRunner()
        with patch(
            "ideanator.cli.OpenAILocalClient", return_value=mock_client
        ) as mock_cls, patch(
            "ideanator.cli.create_server"
        ) as mock_server, _PREFLIGHT_PATCH:
            mock_server.return_value.__enter__ = lambda s: s
            mock_server.return_value.__exit__ = lambda s, *a: None

            runner.invoke(main, ["--mlx"], input=INTERACTIVE_USER_INPUT)

            mock_server.assert_called_once()
            assert mock_server.call_args[0][0] == Backend.MLX

            client_call = mock_cls.call_args
            url = client_call.kwargs.get(
                "base_url", client_call.args[0] if client_call.args else ""
            )
            assert "8080" in url

    def test_model_override(self):
        """--external -m my-model should override the default model."""
        from tests.conftest import MockLLMClient

        mock_client = MockLLMClient(INTERACTIVE_MOCK_RESPONSES)
        runner = CliRunner()
        with patch(
            "ideanator.cli.OpenAILocalClient", return_value=mock_client
        ) as mock_cls, _PREFLIGHT_PATCH:
            runner.invoke(
                main,
                ["--external", "-m", "my-custom-model"],
                input=INTERACTIVE_USER_INPUT,
            )

            client_call = mock_cls.call_args
            model = client_call.kwargs.get(
                "model_id", client_call.args[1] if len(client_call.args) > 1 else ""
            )
            assert model == "my-custom-model"

    def test_server_url_override(self):
        """--external --server-url should override the default URL."""
        from tests.conftest import MockLLMClient

        mock_client = MockLLMClient(INTERACTIVE_MOCK_RESPONSES)
        runner = CliRunner()
        with patch(
            "ideanator.cli.OpenAILocalClient", return_value=mock_client
        ) as mock_cls, _PREFLIGHT_PATCH:
            runner.invoke(
                main,
                ["--external", "--server-url", "http://myserver:9999/v1"],
                input=INTERACTIVE_USER_INPUT,
            )

            client_call = mock_cls.call_args
            url = client_call.kwargs.get(
                "base_url", client_call.args[0] if client_call.args else ""
            )
            assert url == "http://myserver:9999/v1"

    def test_ollama_with_model_override(self):
        """ideanator --ollama -m mistral:7b"""
        from tests.conftest import MockLLMClient

        mock_client = MockLLMClient(INTERACTIVE_MOCK_RESPONSES)
        runner = CliRunner()
        with patch(
            "ideanator.cli.OpenAILocalClient", return_value=mock_client
        ) as mock_cls, patch(
            "ideanator.cli.create_server"
        ) as mock_server, _PREFLIGHT_PATCH:
            mock_server.return_value.__enter__ = lambda s: s
            mock_server.return_value.__exit__ = lambda s, *a: None

            runner.invoke(
                main,
                ["--ollama", "-m", "mistral:7b"],
                input=INTERACTIVE_USER_INPUT,
            )

            # Server should be created with the overridden model
            assert mock_server.call_args[0][1] == "mistral:7b"

            # Client should use the overridden model
            client_call = mock_cls.call_args
            model = client_call.kwargs.get(
                "model_id", client_call.args[1] if len(client_call.args) > 1 else ""
            )
            assert model == "mistral:7b"
