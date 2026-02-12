"""Main Textual App for ideanator TUI — screen routing and entry point."""

from __future__ import annotations

import logging

from textual.app import App

from ideanator.config import DEFAULT_OUTPUT_FILE
from ideanator.types import IdeaResult

from ideanator.tui.screens.batch_pipeline import BatchPipelineScreen
from ideanator.tui.screens.batch_setup import BatchSetupScreen
from ideanator.tui.screens.idea_input import IdeaInputScreen
from ideanator.tui.screens.pipeline import PipelineScreen
from ideanator.tui.screens.settings import AppSettings, SettingsScreen
from ideanator.tui.screens.synthesis import SynthesisScreen
from ideanator.tui.screens.welcome import WelcomeScreen


class IdeanatorApp(App):
    """TUI application for the ARISE idea development pipeline."""

    CSS_PATH = "theme.tcss"
    TITLE = "ideanator"
    SUB_TITLE = "Develop ideas through guided questioning"

    def __init__(self, settings: AppSettings | None = None) -> None:
        super().__init__()
        self._settings = settings or AppSettings()

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen(), callback=self._on_welcome_done)

    # ── Screen transition callbacks ───────────────────────────

    def _on_welcome_done(self, action: str) -> None:
        """Welcome screen dismissed — route to the chosen action."""
        if action == "settings":
            self.push_screen(
                SettingsScreen(self._settings),
                callback=self._on_settings_done,
            )
        elif action == "batch":
            self._start_batch()
        else:
            self.push_screen(IdeaInputScreen(), callback=self._on_idea_submitted)

    def _on_settings_done(self, settings: AppSettings) -> None:
        """Settings screen dismissed — save and go back to welcome."""
        self._settings = settings
        # Apply verbose logging
        logging.basicConfig(
            level=logging.DEBUG if self._settings.verbose else logging.WARNING,
            format="%(levelname)s: %(message)s",
            force=True,
        )
        self.push_screen(WelcomeScreen(), callback=self._on_welcome_done)

    def _on_idea_submitted(self, idea: str) -> None:
        """Idea input dismissed — start the interactive pipeline."""
        self.push_screen(
            PipelineScreen(
                idea=idea,
                backend=self._settings.backend,
                model=self._settings.model,
                server_url=self._settings.server_url,
            ),
            callback=self._on_pipeline_done,
        )

    def _on_pipeline_done(self, result: IdeaResult) -> None:
        """Pipeline finished — show synthesis."""
        self.push_screen(
            SynthesisScreen(
                result=result,
                output_path=self._settings.output_file,
            ),
            callback=self._on_synthesis_done,
        )

    def _on_synthesis_done(self, action: str) -> None:
        """Synthesis screen dismissed — handle action."""
        if action == "new_idea":
            self.push_screen(IdeaInputScreen(), callback=self._on_idea_submitted)

    # ── Batch mode ────────────────────────────────────────────

    def _start_batch(self) -> None:
        """Push the batch setup screen so the user can specify paths."""
        self.push_screen(
            BatchSetupScreen(self._settings),
            callback=self._on_batch_setup_done,
        )

    def _on_batch_setup_done(self, result: dict | None) -> None:
        """Batch setup dismissed — launch pipeline or return to welcome."""
        if result is None:
            # User cancelled — back to welcome
            self.push_screen(WelcomeScreen(), callback=self._on_welcome_done)
            return

        # Persist the paths the user entered
        self._settings.batch_file = result["input_path"]
        self._settings.output_file = result["output_path"]

        self.push_screen(
            BatchPipelineScreen(
                file_path=result["input_path"],
                output_path=result["output_path"],
                backend=self._settings.backend,
                model=self._settings.model,
                server_url=self._settings.server_url,
            ),
            callback=self._on_batch_done,
        )

    def _on_batch_done(self, results: list) -> None:
        """Batch pipeline finished — go back to welcome."""
        self.push_screen(WelcomeScreen(), callback=self._on_welcome_done)


def main(settings: AppSettings | None = None) -> None:
    """Entry point for the TUI. Called by ``ideanator --tui`` or directly."""
    app = IdeanatorApp(settings=settings)
    app.run()


if __name__ == "__main__":
    main()
