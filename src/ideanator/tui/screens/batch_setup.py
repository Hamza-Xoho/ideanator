"""Batch setup screen — specify input/output paths before running batch mode."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Static

from ideanator.config import DEFAULT_OUTPUT_FILE

from ideanator.tui.screens.settings import AppSettings


class BatchSetupScreen(Screen):
    """Configure batch input/output paths, then launch the batch pipeline."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
    ]

    def __init__(self, settings: AppSettings, **kwargs) -> None:
        super().__init__(**kwargs)
        self._settings = settings

    # ── Layout ──────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Vertical(id="batch-setup-panel"):
                yield Static("Batch Run", id="batch-setup-title")
                yield Static(
                    "Process multiple ideas from a JSON file with "
                    "simulated user responses.",
                    id="batch-setup-hint",
                )

                # Input file
                yield Static("Input File", classes="section-label")
                with Horizontal(classes="field-row"):
                    yield Static("-f, --file", classes="field-label")
                    yield Input(
                        value=self._settings.batch_file,
                        placeholder="path/to/ideas.json",
                        id="batch-input-file",
                    )

                # Output file
                yield Static("Output File", classes="section-label")
                with Horizontal(classes="field-row"):
                    yield Static("-o, --output", classes="field-label")
                    yield Input(
                        value=self._settings.output_file,
                        placeholder=DEFAULT_OUTPUT_FILE,
                        id="batch-output-file",
                    )

                # Format hint
                yield Static(
                    "Input Format", classes="section-label section-divider"
                )
                yield Static(
                    '{ "ideas": [{"content": "I want to build..."}, ...] }',
                    id="batch-format-hint",
                )

                # Actions
                with Horizontal(id="batch-setup-actions"):
                    yield Button(
                        "Run Batch", variant="primary", id="run-batch-btn"
                    )
                    yield Button(
                        "Back", variant="default", id="back-batch-btn"
                    )

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#batch-input-file", Input).focus()

    # ── Events ──────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-batch-btn":
            self._submit()
        elif event.button.id == "back-batch-btn":
            self.dismiss(None)

    def action_go_back(self) -> None:
        self.dismiss(None)

    # ── Helpers ─────────────────────────────────────────

    def _submit(self) -> None:
        input_path = self.query_one("#batch-input-file", Input).value.strip()
        output_path = (
            self.query_one("#batch-output-file", Input).value.strip()
            or DEFAULT_OUTPUT_FILE
        )

        if not input_path:
            self.notify(
                "Please specify the path to your ideas JSON file.",
                severity="warning",
            )
            return

        self.dismiss({"input_path": input_path, "output_path": output_path})
