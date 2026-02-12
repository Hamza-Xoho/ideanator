"""Pipeline screen — messaging-style conversation with inline status."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Input

from ideanator.config import Backend

from ideanator.tui.messages import (
    GenericFlagDetected,
    InterviewerMessage,
    PhaseStarted,
    PipelineError,
    PipelineStatus,
    SynthesisComplete,
    UserPromptRequested,
    VaguenessResult,
)
from ideanator.tui.widgets.conversation_view import ConversationView
from ideanator.tui.widgets.dimension_tracker import DimensionTracker
from ideanator.tui.widgets.phase_indicator import PhaseIndicator
from ideanator.tui.worker import PipelineWorker


class PipelineScreen(Screen):
    """Messaging-style pipeline screen — conversation fills the viewport."""

    BINDINGS = [
        ("escape", "confirm_quit", "Quit"),
    ]

    def __init__(
        self,
        idea: str,
        backend: Backend,
        model: str,
        server_url: str,
    ) -> None:
        super().__init__()
        self.idea = idea
        self.backend = backend
        self.model = model
        self.server_url = server_url
        self._worker: PipelineWorker | None = None
        self._waiting_for_input = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="chat-wrapper"):
            # ── top bar: phase dots + dimension pills ──
            with Horizontal(id="chat-topbar"):
                yield PhaseIndicator(id="phase-indicator")
                yield DimensionTracker(id="dim-tracker")
            # ── conversation stream ──
            yield ConversationView(id="conversation")
            # ── input bar ──
            with Horizontal(id="input-bar"):
                yield Input(
                    placeholder="Waiting for pipeline to start...",
                    id="user-input",
                    disabled=True,
                )
        yield Footer()

    def on_mount(self) -> None:
        self._run_pipeline()

    @work(thread=True, exclusive=True)
    def _run_pipeline(self) -> None:
        """Start the ARISE pipeline in a background thread."""
        self._worker = PipelineWorker(self)
        self._worker.run(
            idea=self.idea,
            backend=self.backend,
            model=self.model,
            server_url=self.server_url,
        )

    # ── Pipeline event handlers ──────────────────────────────

    def on_pipeline_status(self, event: PipelineStatus) -> None:
        conv = self.query_one("#conversation", ConversationView)
        conv.add_status(event.text)

    def on_vagueness_result(self, event: VaguenessResult) -> None:
        conv = self.query_one("#conversation", ConversationView)
        conv.add_status(event.text)
        indicator = self.query_one("#phase-indicator", PhaseIndicator)
        indicator.set_phases(event.phases)

    def on_phase_started(self, event: PhaseStarted) -> None:
        indicator = self.query_one("#phase-indicator", PhaseIndicator)
        indicator.advance_to(event.phase_index, event.phase_label)
        tracker = self.query_one("#dim-tracker", DimensionTracker)
        tracker.mark_phase_complete(event.phase_index)
        conv = self.query_one("#conversation", ConversationView)
        conv.add_phase_header(event.phase_label)

    def on_interviewer_message(self, event: InterviewerMessage) -> None:
        conv = self.query_one("#conversation", ConversationView)
        conv.add_interviewer_message(event.text)

    def on_user_prompt_requested(self, event: UserPromptRequested) -> None:
        """Pipeline is waiting for user input — enable the input field."""
        self._waiting_for_input = True
        user_input = self.query_one("#user-input", Input)
        user_input.disabled = False
        user_input.placeholder = "Type your response and press Enter..."
        user_input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """User pressed Enter in the input field."""
        if not self._waiting_for_input:
            return
        text = event.value.strip()
        if not text:
            return

        self._waiting_for_input = False

        conv = self.query_one("#conversation", ConversationView)
        conv.add_user_message(text)

        user_input = self.query_one("#user-input", Input)
        user_input.value = ""
        user_input.disabled = True
        user_input.placeholder = "Thinking..."

        if self._worker:
            self._worker.submit_user_response(text)

    def on_generic_flag_detected(self, event: GenericFlagDetected) -> None:
        conv = self.query_one("#conversation", ConversationView)
        truncated = (
            event.question[:80] + "..."
            if len(event.question) > 80
            else event.question
        )
        conv.add_warning(f"Generic question detected: {truncated}")

    def on_synthesis_complete(self, event: SynthesisComplete) -> None:
        tracker = self.query_one("#dim-tracker", DimensionTracker)
        tracker.mark_all_done()
        self.dismiss(event.result)

    def on_pipeline_error(self, event: PipelineError) -> None:
        conv = self.query_one("#conversation", ConversationView)
        conv.add_error(event.error)
        user_input = self.query_one("#user-input", Input)
        user_input.disabled = True
        user_input.placeholder = "Pipeline error — press Escape to quit."

    def action_confirm_quit(self) -> None:
        if self._worker:
            self._worker.cancel()
        self.app.exit()
