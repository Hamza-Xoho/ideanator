"""Synthesis screen — displays final results with save option."""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, RichLog, Static

from ideanator.types import IdeaResult


class SynthesisScreen(Screen):
    """Displays the final synthesis and offers save/restart/quit."""

    BINDINGS = [
        ("s", "save", "Save"),
        ("n", "new_idea", "New Idea"),
        ("q", "quit_app", "Quit"),
    ]

    def __init__(self, result: IdeaResult, output_path: str = "arise_results.json") -> None:
        super().__init__()
        self.result = result
        self._output_path = output_path

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="synthesis-panel"):
            yield Static("Synthesis", id="synthesis-title")
            yield Static(
                f"Phases: {', '.join(self.result.phases_executed)}  |  "
                f"Generic flags: {len(self.result.generic_flags)}",
                id="synthesis-meta",
            )
            yield RichLog(id="synthesis-content", wrap=True, markup=True)
            with Horizontal(id="synthesis-actions"):
                yield Button("Save Results", variant="primary", id="save-btn")
                yield Button("New Idea", variant="default", id="new-btn")
                yield Button("Quit", variant="error", id="quit-btn")
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#synthesis-content", RichLog)
        # Format each synthesis section with color
        for line in self.result.synthesis.split("\n"):
            stripped = line.strip()
            if stripped.startswith("[") and "]:" in stripped:
                # Header line like "[IDEA]: ..."
                bracket_end = stripped.index("]:")
                header = stripped[: bracket_end + 2]
                body = stripped[bracket_end + 2 :]
                log.write(f"[bold cyan]{header}[/bold cyan]{body}")
            elif stripped:
                log.write(f"  {stripped}")
            else:
                log.write("")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self.action_save()
        elif event.button.id == "new-btn":
            self.action_new_idea()
        elif event.button.id == "quit-btn":
            self.action_quit_app()

    def action_save(self) -> None:
        """Save results to JSON file."""
        path = Path(self._output_path)
        data = {
            "original_idea": self.result.original_idea,
            "timestamp": self.result.timestamp,
            "vagueness_assessment": self.result.vagueness_assessment,
            "phases_executed": self.result.phases_executed,
            "conversation": [
                {"phase": t.phase, "role": t.role, "content": t.content}
                for t in self.result.conversation
            ],
            "generic_flags": [
                {"phase": g.phase, "question": g.question, "flag": g.flag}
                for g in self.result.generic_flags
            ],
            "synthesis": self.result.synthesis,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        self.notify(f"Saved to {path}", title="Results Saved")

    def action_new_idea(self) -> None:
        """Go back to start a new idea."""
        self.dismiss("new_idea")

    def action_quit_app(self) -> None:
        self.app.exit()
