"""CLI entry point for the ideanator tool."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click

from ideanator.config import DEFAULT_MODEL_ID, DEFAULT_OUTPUT_FILE, DEFAULT_SERVER_URL
from ideanator.llm import MLXServer, OpenAILocalClient
from ideanator.pipeline import run_arise_for_idea, run_arise_interactive


@click.command()
@click.option(
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True),
    help="JSON file with ideas for batch processing.",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL_ID,
    show_default=True,
    help="Model ID for the MLX server.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    help="Output path for results JSON.",
)
@click.option(
    "--server-url",
    default=DEFAULT_SERVER_URL,
    show_default=True,
    help="LLM server URL (if server is already running).",
)
@click.option(
    "--no-server",
    is_flag=True,
    help="Skip auto-starting the MLX server (assumes it's already running).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose debug logging.",
)
@click.version_option(package_name="ideanator")
def main(
    file_path: str | None,
    model: str,
    output_path: str | None,
    server_url: str,
    no_server: bool,
    verbose: bool,
) -> None:
    """
    ideanator — Develop vague ideas through the ARISE questioning pipeline.

    By default, runs in interactive mode: you type your idea and answer
    questions from the ARISE framework to develop it further.

    Use --file to process multiple ideas from a JSON file in batch mode
    (with simulated user responses for testing prompt efficacy).
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if no_server:
        client = OpenAILocalClient(base_url=server_url, model_id=model)
        if file_path:
            _run_batch(client, file_path, output_path or DEFAULT_OUTPUT_FILE, model)
        else:
            _run_interactive(client, output_path)
    else:
        try:
            with MLXServer(model_id=model) as _server:
                client = OpenAILocalClient(base_url=server_url, model_id=model)
                if file_path:
                    _run_batch(
                        client, file_path, output_path or DEFAULT_OUTPUT_FILE, model
                    )
                else:
                    _run_interactive(client, output_path)
        except KeyboardInterrupt:
            click.echo("\n\nInterrupted. Cleaning up...")


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
    click.echo(f"  ARISE Pipeline — Batch Mode")
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
    click.echo(f"  PIPELINE COMPLETE")
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
    click.echo("\n" + "="*60)
    click.echo("  ARISE Pipeline — Interactive Mode")
    click.echo("  Develop your idea through guided questioning.")
    click.echo("="*60 + "\n")

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
