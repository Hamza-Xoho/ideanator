"""Batch pipeline screen — processes multiple ideas with simulated responses."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from ideanator.config import Backend

from ideanator.tui.messages import (
    BatchComplete,
    BatchIdeaComplete,
    BatchIdeaStarted,
    BatchSimulatedResponse,
    GenericFlagDetected,
    InterviewerMessage,
    PhaseStarted,
    PipelineError,
    PipelineStatus,
    VaguenessResult,
)
from ideanator.tui.widgets.conversation_view import ConversationView
from ideanator.tui.widgets.dimension_tracker import DimensionTracker
from ideanator.tui.widgets.phase_indicator import PhaseIndicator
from ideanator.tui.worker import BatchPipelineWorker


class BatchPipelineScreen(Screen):
    """Messaging-style batch pipeline — shows each idea being processed
    with LLM-simulated user responses. No manual input required.
    """

    BINDINGS = [
        ("escape", "confirm_quit", "Quit"),
    ]

    def __init__(
        self,
        file_path: str,
        output_path: str,
        backend: Backend,
        model: str,
        server_url: str,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.output_path = output_path
        self.backend = backend
        self.model = model
        self.server_url = server_url
        self._worker: BatchPipelineWorker | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="chat-wrapper"):
            with Horizontal(id="chat-topbar"):
                yield PhaseIndicator(id="phase-indicator")
                yield DimensionTracker(id="dim-tracker")
            yield ConversationView(id="conversation")
            # no input bar — batch mode is fully automated
            with Horizontal(id="batch-status-bar"):
                yield Static(
                    "Batch mode — processing ideas with simulated responses...",
                    id="batch-status-text",
                )
        yield Footer()

    def on_mount(self) -> None:
        self._run_batch()

    @work(thread=True, exclusive=True)
    def _run_batch(self) -> None:
        """Start the batch pipeline in a background thread."""
        self._worker = BatchPipelineWorker(self)
        self._worker.run(
            file_path=self.file_path,
            output_path=self.output_path,
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

    def on_batch_simulated_response(self, event: BatchSimulatedResponse) -> None:
        conv = self.query_one("#conversation", ConversationView)
        conv.add_simulated_response(event.text)

    def on_generic_flag_detected(self, event: GenericFlagDetected) -> None:
        conv = self.query_one("#conversation", ConversationView)
        truncated = (
            event.question[:80] + "..."
            if len(event.question) > 80
            else event.question
        )
        conv.add_warning(f"Generic question detected: {truncated}")

    def on_batch_idea_started(self, event: BatchIdeaStarted) -> None:
        conv = self.query_one("#conversation", ConversationView)
        truncated = (
            event.idea[:80] + "..." if len(event.idea) > 80 else event.idea
        )
        conv.add_status(
            f"[bold]Idea {event.idea_index + 1}/{event.total_ideas}:[/bold] "
            f"{truncated}",
        )
        self.query_one("#batch-status-text", Static).update(
            f"Processing idea {event.idea_index + 1} of {event.total_ideas}..."
        )

        # Reset tracker for new idea
        tracker = self.query_one("#dim-tracker", DimensionTracker)
        tracker.covered = frozenset()
        tracker._completed_phase_count = 0

    def on_batch_idea_complete(self, event: BatchIdeaComplete) -> None:
        conv = self.query_one("#conversation", ConversationView)
        conv.add_status(
            f"Idea {event.idea_index + 1} complete.",
            severity="info",
        )
        tracker = self.query_one("#dim-tracker", DimensionTracker)
        tracker.mark_all_done()

    def on_batch_complete(self, event: BatchComplete) -> None:
        n = len(event.results)
        self.query_one("#batch-status-text", Static).update(
            f"Batch complete — {n} idea{'s' if n != 1 else ''} processed. "
            f"Results saved to {event.output_path}"
        )
        conv = self.query_one("#conversation", ConversationView)
        conv.add_status(
            f"[bold green]All {n} ideas processed.[/bold green] "
            f"Results saved to {event.output_path}",
        )
        self.dismiss(event.results)

    def on_pipeline_error(self, event: PipelineError) -> None:
        conv = self.query_one("#conversation", ConversationView)
        conv.add_error(event.error)
        self.query_one("#batch-status-text", Static).update(
            "Batch error — press Escape to quit."
        )

    def action_confirm_quit(self) -> None:
        if self._worker:
            self._worker.cancel()
        self.app.exit()
