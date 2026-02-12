"""Welcome screen — landing page with Start, Batch Run, and Settings."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static


class WelcomeScreen(Screen):
    """Splash screen — start interactive, run batch, or configure settings."""

    BINDINGS = [("escape", "app.quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Vertical(id="welcome-panel"):
                yield Static("ideanator", id="welcome-title")
                yield Static(
                    "Develop vague ideas through guided questioning",
                    id="welcome-subtitle",
                )
                yield Button("Start", variant="primary", id="start-btn")
                yield Button(
                    "Batch Run",
                    variant="default",
                    id="batch-btn",
                )
                yield Button("Settings", variant="default", id="settings-btn")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-btn":
            self.dismiss("start")
        elif event.button.id == "batch-btn":
            self.dismiss("batch")
        elif event.button.id == "settings-btn":
            self.dismiss("settings")
