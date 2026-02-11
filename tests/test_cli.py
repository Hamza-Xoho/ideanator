"""Tests for the CLI interface."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from ideanator.cli import main


class TestCLIHelp:
    def test_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "ARISE" in result.output
        assert "--file" in result.output
        assert "--model" in result.output

    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0


class TestCLIBatchMode:
    def test_file_not_found(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--file", "/nonexistent/path.json", "--no-server"])
        assert result.exit_code != 0

    def test_batch_mode_with_mock(self):
        """Batch mode processes ideas from JSON file."""
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

        mock_responses = [
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

        from tests.conftest import MockLLMClient

        mock_client = MockLLMClient(mock_responses)

        runner = CliRunner()
        with patch("ideanator.cli.OpenAILocalClient", return_value=mock_client):
            result = runner.invoke(
                main,
                [
                    "--file", input_path,
                    "--output", output_path,
                    "--no-server",
                ],
            )

        assert result.exit_code == 0
        assert "PIPELINE COMPLETE" in result.output

        # Verify output file was written
        with open(output_path) as f:
            results = json.load(f)
        assert len(results) == 1
        assert results[0]["original_idea"] == "I want to build a test app."

        # Cleanup
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


class TestCLIInteractiveMode:
    def test_interactive_mode_prompts_for_idea(self):
        """Interactive mode asks user for their idea."""
        from tests.conftest import MockLLMClient

        # Short idea → safety net → 4 phases (anchor, reveal, imagine, scope)
        # Each phase needs: question generation response
        # Plus: 1 vagueness assessment + 1 synthesis = total 6 mock responses
        mock_responses = [
            "NONE",  # vagueness (safety net triggers)
            "[REFLECTION] R\n[QUESTION 1] What sparked this?\n[QUESTION 2] A or B?",
            "[REFLECTION] R\n[QUESTION 1] What's broken?\n[QUESTION 2] Cost or quality?",
            "[REFLECTION] R\n[QUESTION 1] Perfect version?\n[QUESTION 2] Scale?",
            "[REFLECTION] R\n[QUESTION 1] Risk?\n[QUESTION 2] Smallest version?",
            "Synthesis output.",
        ]
        mock_client = MockLLMClient(mock_responses)

        # Need 1 line for idea + 4 lines for phase responses
        user_input = (
            "I want to build a test app.\n"
            "My personal experience.\n"
            "The tools are bad.\n"
            "It would feel amazing.\n"
            "Adoption is hard.\n"
        )

        runner = CliRunner()
        with patch("ideanator.cli.OpenAILocalClient", return_value=mock_client):
            result = runner.invoke(
                main,
                ["--no-server"],
                input=user_input,
            )

        assert result.exit_code == 0
        assert "ARISE Pipeline" in result.output
