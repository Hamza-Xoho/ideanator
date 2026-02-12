"""Background pipeline runners that bridge the synchronous ARISE pipeline
with Textual's async event loop.

PipelineWorker       — interactive mode (real user answers)
BatchPipelineWorker  — batch mode (simulated responses from a JSON file)

Both run in a background thread (via @work(thread=True)) and communicate
with the TUI main thread via post_message() (thread-safe).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from ideanator.config import Backend, get_backend_config
from ideanator.llm import OpenAILocalClient, create_server, preflight_check
from ideanator.pipeline import run_arise_for_idea, run_arise_interactive

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
    SynthesisComplete,
    UserPromptRequested,
    VaguenessResult,
)

logger = logging.getLogger(__name__)


# ── Interactive worker ────────────────────────────────────────────


class PipelineWorker:
    """Runs the ARISE pipeline in a background thread, posting Messages
    to a Textual widget for rendering.

    Usage:
        worker = PipelineWorker(screen)
        worker.run(idea, backend, model, server_url)  # blocking, call from thread
    """

    def __init__(self, target) -> None:
        """
        Args:
            target: A Textual Widget/Screen/App with post_message().
        """
        self._target = target

        # Synchronization for user input
        self._user_response: str = ""
        self._input_ready = threading.Event()
        self._cancelled = False

        # Phase tracking
        self._phases: list[str] = []
        self._current_phase_index: int = 0

    def submit_user_response(self, response: str) -> None:
        """Called from the main TUI thread when the user submits input."""
        self._user_response = response
        self._input_ready.set()

    def cancel(self) -> None:
        """Signal the worker to stop waiting for input."""
        self._cancelled = True
        self._input_ready.set()

    def _callback(self, event: str, data: str) -> str | None:
        """Pipeline callback — runs in the background thread.

        Posts thread-safe Messages to the TUI. For 'prompt_user', blocks
        until the main thread calls submit_user_response().
        """
        if self._cancelled:
            return None

        if event == "status":
            self._target.post_message(PipelineStatus(data))

        elif event == "vagueness":
            # Parse phase names from the formatted string
            # Format: "Covered: X/6 | Missing: ... | Phases: ANCHOR → REVEAL → SCOPE"
            phases: list[str] = []
            if "Phases: " in data:
                phases_part = data.split("Phases: ")[-1]
                phases = [p.strip().upper() for p in phases_part.split("→")]
            self._phases = phases
            self._target.post_message(VaguenessResult(text=data, phases=phases))

        elif event == "phase_start":
            self._target.post_message(
                PhaseStarted(
                    phase_label=data,
                    phase_index=self._current_phase_index,
                    total_phases=len(self._phases),
                )
            )
            self._current_phase_index += 1

        elif event == "interviewer":
            self._target.post_message(InterviewerMessage(text=data))

        elif event == "prompt_user":
            self._target.post_message(UserPromptRequested(phase_label=data))
            # Block until main thread provides the response
            self._input_ready.wait(timeout=600)
            self._input_ready.clear()
            if self._cancelled:
                return ""
            return self._user_response

        elif event == "generic_flag":
            self._target.post_message(GenericFlagDetected(question=data))

        elif event == "synthesis":
            # Handled via the return value of run_arise_interactive
            pass

        return None

    def run(self, idea: str, backend: Backend, model: str, server_url: str) -> None:
        """Execute the full pipeline. Call from a background thread."""
        cfg = get_backend_config(backend)
        resolved_model = model or cfg.default_model
        resolved_url = server_url or cfg.default_url

        try:
            if cfg.needs_server:
                server = create_server(backend, resolved_model)
                with server:
                    client = OpenAILocalClient(
                        base_url=resolved_url, model_id=resolved_model
                    )
                    self._execute(client, resolved_url, resolved_model, backend, idea)
            else:
                client = OpenAILocalClient(
                    base_url=resolved_url, model_id=resolved_model
                )
                self._execute(client, resolved_url, resolved_model, backend, idea)
        except Exception as e:
            logger.exception("Pipeline error")
            self._target.post_message(PipelineError(error=str(e)))

    def _execute(
        self,
        client: OpenAILocalClient,
        url: str,
        model: str,
        backend: Backend,
        idea: str,
    ) -> None:
        """Inner execution: preflight check + pipeline run."""
        if not preflight_check(url, model, backend):
            self._target.post_message(
                PipelineStatus(
                    "Warning: Pre-flight check failed. "
                    "The server may not be reachable. Proceeding anyway..."
                )
            )

        result = run_arise_interactive(client, idea, callback=self._callback)

        if not self._cancelled:
            self._target.post_message(
                SynthesisComplete(synthesis=result.synthesis, result=result)
            )


# ── Batch worker ──────────────────────────────────────────────────


class BatchPipelineWorker:
    """Runs the ARISE pipeline in batch mode — processes multiple ideas
    from a JSON file with LLM-simulated user responses.

    Usage:
        worker = BatchPipelineWorker(screen)
        worker.run(file_path, output_path, backend, model, server_url)
    """

    def __init__(self, target) -> None:
        self._target = target
        self._cancelled = False

        # Phase tracking (per-idea, reset between ideas)
        self._phases: list[str] = []
        self._current_phase_index: int = 0

    def cancel(self) -> None:
        self._cancelled = True

    def _callback(self, event: str, data: str) -> str | None:
        """Batch callback — no user prompting, shows simulated responses."""
        if self._cancelled:
            return None

        if event == "status":
            self._target.post_message(PipelineStatus(data))

        elif event == "vagueness":
            phases: list[str] = []
            if "Phases: " in data:
                phases_part = data.split("Phases: ")[-1]
                phases = [p.strip().upper() for p in phases_part.split("→")]
            self._phases = phases
            self._target.post_message(VaguenessResult(text=data, phases=phases))

        elif event == "phase_start":
            self._target.post_message(
                PhaseStarted(
                    phase_label=data,
                    phase_index=self._current_phase_index,
                    total_phases=len(self._phases),
                )
            )
            self._current_phase_index += 1

        elif event == "interviewer":
            self._target.post_message(InterviewerMessage(text=data))

        elif event == "user_sim":
            self._target.post_message(BatchSimulatedResponse(text=data))

        elif event == "generic_flag":
            self._target.post_message(GenericFlagDetected(question=data))

        elif event == "synthesis":
            pass  # handled via return value

        return None

    def run(
        self,
        file_path: str,
        output_path: str,
        backend: Backend,
        model: str,
        server_url: str,
    ) -> None:
        """Execute batch processing. Call from a background thread."""
        # ── Load and validate the ideas file ──
        path = Path(file_path)
        if not path.exists():
            self._target.post_message(
                PipelineError(error=f"File not found: {file_path}")
            )
            return

        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            self._target.post_message(
                PipelineError(error=f"Invalid JSON in '{file_path}': {e}")
            )
            return

        if not isinstance(data, dict) or "ideas" not in data:
            self._target.post_message(
                PipelineError(
                    error=f'Expected {{"ideas": [...]}} in \'{file_path}\'.'
                )
            )
            return

        ideas = data["ideas"]
        if not isinstance(ideas, list) or not ideas:
            self._target.post_message(
                PipelineError(error="No ideas found in input file.")
            )
            return

        for i, entry in enumerate(ideas):
            if not isinstance(entry, dict) or "content" not in entry:
                self._target.post_message(
                    PipelineError(
                        error=f'ideas[{i}] must be {{"content": "..."}}.'
                    )
                )
                return

        # ── Resolve backend config ──
        cfg = get_backend_config(backend)
        resolved_model = model or cfg.default_model
        resolved_url = server_url or cfg.default_url

        try:
            if cfg.needs_server:
                server = create_server(backend, resolved_model)
                with server:
                    client = OpenAILocalClient(
                        base_url=resolved_url, model_id=resolved_model
                    )
                    self._process_ideas(
                        client, ideas, output_path,
                        resolved_url, resolved_model, backend,
                    )
            else:
                client = OpenAILocalClient(
                    base_url=resolved_url, model_id=resolved_model
                )
                self._process_ideas(
                    client, ideas, output_path,
                    resolved_url, resolved_model, backend,
                )
        except Exception as e:
            logger.exception("Batch pipeline error")
            self._target.post_message(PipelineError(error=str(e)))

    def _process_ideas(
        self,
        client: OpenAILocalClient,
        ideas: list[dict],
        output_path: str,
        url: str,
        model: str,
        backend: Backend,
    ) -> None:
        """Process each idea sequentially, saving results incrementally."""
        if not preflight_check(url, model, backend):
            self._target.post_message(
                PipelineStatus(
                    "Warning: Pre-flight check failed. "
                    "The server may not be reachable. Proceeding anyway..."
                )
            )

        all_results: list[dict] = []

        for i, entry in enumerate(ideas):
            if self._cancelled:
                break

            idea = entry["content"]

            # Reset per-idea phase tracking
            self._phases = []
            self._current_phase_index = 0

            self._target.post_message(
                BatchIdeaStarted(
                    idea=idea, idea_index=i, total_ideas=len(ideas)
                )
            )

            result = run_arise_for_idea(
                client, idea, callback=self._callback
            )

            result_dict = _result_to_dict(result)
            all_results.append(result_dict)

            # Save incrementally (survives interruption)
            Path(output_path).write_text(
                json.dumps(all_results, indent=2, ensure_ascii=False)
            )

            self._target.post_message(
                BatchIdeaComplete(idea_index=i, result=result)
            )

        if not self._cancelled:
            self._target.post_message(
                BatchComplete(results=all_results, output_path=output_path)
            )


# ── Shared helpers ────────────────────────────────────────────────


def _result_to_dict(result) -> dict:
    """Convert an IdeaResult to a JSON-serializable dict."""
    return {
        "original_idea": result.original_idea,
        "timestamp": result.timestamp,
        "vagueness_assessment": result.vagueness_assessment,
        "phases_executed": result.phases_executed,
        "conversation": [
            {"phase": t.phase, "role": t.role, "content": t.content}
            for t in result.conversation
        ],
        "generic_flags": [
            {"phase": g.phase, "question": g.question, "flag": g.flag}
            for g in result.generic_flags
        ],
        "synthesis": result.synthesis,
    }
