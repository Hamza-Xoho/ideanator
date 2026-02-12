"""Idea input screen — large text area for the user's idea."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static, TextArea


class IdeaInputScreen(Screen):
    """Full-screen text area for describing the idea."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Vertical(id="idea-panel"):
                yield Static("What's your idea?", id="idea-prompt")
                yield Static(
                    "Describe it in a few sentences. The more detail you share, "
                    "the more targeted the questions will be.",
                    id="idea-hint",
                )
                yield TextArea(id="idea-text")
                yield Button("Submit", variant="primary", id="submit-btn")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#idea-text", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit-btn":
            self._submit_idea()

    def _submit_idea(self) -> None:
        text = self.query_one("#idea-text", TextArea).text.strip()
        if text:
            self.dismiss(text)
        else:
            self.notify("Please enter your idea first.", severity="warning")
