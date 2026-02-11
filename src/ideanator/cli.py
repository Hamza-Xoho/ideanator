"""CLI entry point for the ideanator tool."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from ideanator.config import (
    Backend,
    DEFAULT_OUTPUT_FILE,
    get_backend_config,
)
from ideanator.llm import OpenAILocalClient, create_server
from ideanator.pipeline import run_arise_for_idea, run_arise_interactive


# ── Custom help formatter ─────────────────────────────────────────────


class IdeanatorHelpFormatter(click.HelpFormatter):
    """Wider formatter so examples don't wrap awkwardly."""

    def __init__(self, **kwargs):
        kwargs.setdefault("max_width", 90)
        super().__init__(**kwargs)


class IdeanatorCommand(click.Command):
    """Custom command class with a polished help layout."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        formatter = IdeanatorHelpFormatter()

        # Title
        formatter.write("\n")
        formatter.write("  ideanator — develop vague ideas through structured questioning\n")
        formatter.write("\n")

        # Usage
        formatter.write("  USAGE\n")
        formatter.write("    ideanator [backend] [options]\n")
        formatter.write("\n")

        # Backends
        formatter.write("  BACKENDS (pick one)\n")
        formatter.write("    --ollama            Use Ollama  (Linux, macOS, Windows)\n")
        formatter.write("    --mlx               Use MLX     (macOS + Apple Silicon)\n")
        formatter.write("    --external          Use any already-running server\n")
        formatter.write("    (default: --ollama)\n")
        formatter.write("\n")

        # Options
        formatter.write("  OPTIONS\n")
        formatter.write("    -m, --model ID      Model to use (default depends on backend)\n")
        formatter.write("    --server-url URL    Override the server URL\n")
        formatter.write("    -f, --file PATH     Batch mode: process ideas from a JSON file\n")
        formatter.write("    -o, --output PATH   Save results to a JSON file\n")
        formatter.write("    -v, --verbose       Show debug logs\n")
        formatter.write("    --version           Show version\n")
        formatter.write("    --help              Show this help\n")
        formatter.write("\n")

        # Examples
        formatter.write("  EXAMPLES\n")
        formatter.write("\n")
        formatter.write("    Interactive (type your idea, answer questions):\n")
        formatter.write("      ideanator --ollama\n")
        formatter.write("      ideanator --ollama -m mistral:7b\n")
        formatter.write("      ideanator --mlx -m mlx-community/Llama-3.2-1B-Instruct-4bit\n")
        formatter.write("      ideanator --external --server-url http://localhost:1234/v1\n")
        formatter.write("\n")
        formatter.write("    Batch (process a file of ideas with simulated responses):\n")
        formatter.write("      ideanator --ollama -f ideas.json -o results.json\n")
        formatter.write("      ideanator --mlx -f ideas.json\n")
        formatter.write("\n")

        # Backend defaults
        formatter.write("  BACKEND DEFAULTS\n")
        formatter.write("    ┌────────────┬────────────────────────────────────────┬──────────────────────────────┐\n")
        formatter.write("    │ Backend    │ Default model                          │ Default URL                  │\n")
        formatter.write("    ├────────────┼────────────────────────────────────────┼──────────────────────────────┤\n")
        formatter.write("    │ --ollama   │ llama3.2:3b                            │ http://localhost:11434/v1     │\n")
        formatter.write("    │ --mlx      │ mlx-community/Llama-3.2-3B-Instruct.. │ http://localhost:8080/v1      │\n")
        formatter.write("    │ --external │ default                                │ http://localhost:8080/v1      │\n")
        formatter.write("    └────────────┴────────────────────────────────────────┴──────────────────────────────┘\n")
        formatter.write("\n")
        formatter.write("    Override any default with -m or --server-url.\n")
        formatter.write("\n")

        # Input format
        formatter.write("  INPUT FILE FORMAT (for -f / --file)\n")
        formatter.write('    { "ideas": [{"content": "I want to build..."}, ...] }\n')
        formatter.write("\n")

        click.echo(formatter.getvalue(), color=ctx.color)


# ── Resolve backend from flags ────────────────────────────────────────


def _resolve_backend(
    mlx: bool, ollama: bool, external: bool, no_server: bool
) -> Backend:
    """
    Resolve which backend to use from the mutually exclusive flags.

    Priority: explicit flags > --no-server > default (ollama).
    """
    selected = sum([mlx, ollama, external])
    if selected > 1:
        raise click.UsageError(
            "Pick only one backend: --mlx, --ollama, or --external"
        )

    if mlx:
        return Backend.MLX
    if ollama:
        return Backend.OLLAMA
    if external or no_server:
        return Backend.EXTERNAL

    # Default backend
    return Backend.OLLAMA


# ── Main command ──────────────────────────────────────────────────────


@click.command(cls=IdeanatorCommand)
@click.option("--ollama", "use_ollama", is_flag=True,
              help="Use Ollama backend (Linux, macOS, Windows).")
@click.option("--mlx", "use_mlx", is_flag=True,
              help="Use MLX backend (macOS + Apple Silicon).")
@click.option("--external", "use_external", is_flag=True,
              help="Connect to an already-running server.")
@click.option("--no-server", is_flag=True, hidden=True,
              help="Alias for --external (backwards compat).")
@click.option("-m", "--model", default=None,
              help="Model ID (default depends on backend).")
@click.option("--server-url", default=None,
              help="Override the LLM server URL.")
@click.option("-f", "--file", "file_path", type=click.Path(exists=True),
              help="JSON file with ideas for batch processing.")
@click.option("-o", "--output", "output_path", type=click.Path(),
              help="Output path for results JSON.")
@click.option("-v", "--verbose", is_flag=True,
              help="Enable verbose debug logging.")
@click.version_option(package_name="ideanator")
def main(
    use_ollama: bool,
    use_mlx: bool,
    use_external: bool,
    no_server: bool,
    model: str | None,
    server_url: str | None,
    file_path: str | None,
    output_path: str | None,
    verbose: bool,
) -> None:
    """ideanator — develop vague ideas through the ARISE questioning pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # Resolve backend
    try:
        backend = _resolve_backend(use_mlx, use_ollama, use_external, no_server)
    except click.UsageError as e:
        raise SystemExit(f"Error: {e}")

    cfg = get_backend_config(backend)

    # Apply defaults: user-provided values override backend defaults
    resolved_model = model or cfg.default_model
    resolved_url = server_url or cfg.default_url

    if cfg.needs_server:
        try:
            server = create_server(backend, resolved_model)
            with server:
                client = OpenAILocalClient(
                    base_url=resolved_url, model_id=resolved_model
                )
                _dispatch(client, file_path, output_path, resolved_model)
        except KeyboardInterrupt:
            click.echo("\n\nInterrupted. Cleaning up...")
    else:
        client = OpenAILocalClient(
            base_url=resolved_url, model_id=resolved_model
        )
        _dispatch(client, file_path, output_path, resolved_model)


# ── Dispatch + modes ──────────────────────────────────────────────────


def _dispatch(
    client: OpenAILocalClient,
    file_path: str | None,
    output_path: str | None,
    model: str,
) -> None:
    """Route to batch or interactive mode."""
    if file_path:
        _run_batch(client, file_path, output_path or DEFAULT_OUTPUT_FILE, model)
    else:
        _run_interactive(client, output_path)


def _run_batch(
    client: OpenAILocalClient,
    file_path: str,
    output_path: str,
    model: str = "",
) -> None:
    """Process multiple ideas from a JSON file with simulated user responses."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            ideas = data.get("ideas", [])
    except (json.JSONDecodeError, KeyError) as e:
        click.echo(f"Error loading ideas file: {e}", err=True)
        sys.exit(1)

    model_label = getattr(client, "model_id", model) or "unknown"
    click.echo(f"\n{'='*60}")
    click.echo("  ARISE Pipeline — Batch Mode")
    click.echo(f"  Ideas: {len(ideas)} | Model: {model_label}")
    click.echo(f"{'='*60}\n")

    all_results = []
    total_phases = 0
    total_generic = 0

    for i, entry in enumerate(ideas):
        idea = entry["content"]
        click.echo(f"\n{'─'*60}")
        click.echo(f"  IDEA {i + 1}/{len(ideas)}: {idea[:60]}...")
        click.echo(f"{'─'*60}")

        result = run_arise_for_idea(client, idea, callback=_batch_callback)
        all_results.append(_result_to_dict(result))
        total_phases += len(result.phases_executed)
        total_generic += len(result.generic_flags)

        # Save incrementally after each idea
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    avg_phases = total_phases / len(all_results) if all_results else 0
    click.echo(f"\n{'='*60}")
    click.echo("  PIPELINE COMPLETE")
    click.echo(f"  Ideas: {len(all_results)}")
    click.echo(f"  Total phases: {total_phases} (avg {avg_phases:.1f}/idea)")
    click.echo(f"  Generic flags: {total_generic}")
    click.echo(f"  Output: {output_path}")
    click.echo(f"{'='*60}\n")


def _run_interactive(
    client: OpenAILocalClient,
    output_path: str | None,
) -> None:
    """Interactive mode: prompt user for their idea, run ARISE pipeline."""
    click.echo("\n" + "=" * 60)
    click.echo("  ARISE Pipeline — Interactive Mode")
    click.echo("  Develop your idea through guided questioning.")
    click.echo("=" * 60 + "\n")

    idea = click.prompt("What's your idea?", type=str)
    if not idea.strip():
        click.echo("No idea provided. Exiting.", err=True)
        return

    click.echo()

    def interactive_callback(event: str, data: str) -> str | None:
        if event == "status":
            click.echo(f"  → {data}")
        elif event == "vagueness":
            click.echo(f"    {data}")
        elif event == "phase_start":
            click.echo(f"\n  ━━ {data} ━━")
        elif event == "interviewer":
            click.echo(f"\n{data}\n")
        elif event == "prompt_user":
            return click.prompt("Your response")
        elif event == "synthesis":
            click.echo(f"\n{'─'*60}")
            click.echo("  SYNTHESIS")
            click.echo(f"{'─'*60}")
            click.echo(data)
        return None

    result = run_arise_interactive(client, idea, callback=interactive_callback)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(_result_to_dict(result), f, indent=2, ensure_ascii=False)
        click.echo(f"\n  Results saved to {output_path}")


# ── Callbacks + helpers ───────────────────────────────────────────────


def _batch_callback(event: str, data: str) -> str | None:
    """Progress callback for batch mode."""
    if event == "status":
        click.echo(f"  → {data}")
    elif event == "vagueness":
        click.echo(f"    {data}")
    elif event == "phase_start":
        click.echo(f"\n  → {data}")
    elif event == "interviewer":
        click.echo(f"    Q: {data[:120]}...")
    elif event == "user_sim":
        click.echo(f"    A: {data[:120]}...")
    elif event == "generic_flag":
        click.echo(f"    ⚠ Generic: {data[:60]}...")
    return None


def _result_to_dict(result) -> dict:
    """Convert an IdeaResult to a JSON-serializable dict."""
    return {
        "original_idea": result.original_idea,
        "timestamp": result.timestamp,
        "vagueness_assessment": result.vagueness_assessment,
        "phases_executed": result.phases_executed,
        "conversation": [
            {
                "phase": t.phase,
                "role": t.role,
                "content": t.content,
            }
            for t in result.conversation
        ],
        "generic_flags": [
            {"phase": g.phase, "question": g.question, "flag": g.flag}
            for g in result.generic_flags
        ],
        "synthesis": result.synthesis,
    }
