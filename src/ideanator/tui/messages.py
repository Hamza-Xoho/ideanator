"""Custom Textual Message classes for pipeline callback events.

Each message maps to one event type from the ARISE pipeline's
 ProgressCallback. The PipelineWorker posts these from a background
thread; PipelineScreen handles them on the main thread.
"""

from __future__ import annotations

from textual.message import Message


# ── Interactive pipeline messages ─────────────────────────────────


class PipelineStatus(Message):
    """General status update (e.g., 'Scoring vagueness...')."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class VaguenessResult(Message):
    """Vagueness scoring complete — carries the formatted result string and phase list."""

    def __init__(self, text: str, phases: list[str]) -> None:
        super().__init__()
        self.text = text
        self.phases = phases


class PhaseStarted(Message):
    """A new ARISE phase has begun."""

    def __init__(self, phase_label: str, phase_index: int, total_phases: int) -> None:
        super().__init__()
        self.phase_label = phase_label
        self.phase_index = phase_index
        self.total_phases = total_phases


class InterviewerMessage(Message):
    """Interviewer reflection + questions to display."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class UserPromptRequested(Message):
    """Pipeline is waiting for user input."""

    def __init__(self, phase_label: str) -> None:
        super().__init__()
        self.phase_label = phase_label


class GenericFlagDetected(Message):
    """A question was flagged as too generic."""

    def __init__(self, question: str) -> None:
        super().__init__()
        self.question = question


class SynthesisComplete(Message):
    """Pipeline finished — synthesis and full result available."""

    def __init__(self, synthesis: str, result: object) -> None:
        super().__init__()
        self.synthesis = synthesis
        self.result = result


class PipelineError(Message):
    """Unrecoverable pipeline error."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


# ── Batch pipeline messages ───────────────────────────────────────


class BatchIdeaStarted(Message):
    """A new idea is being processed in batch mode."""

    def __init__(
        self, idea: str, idea_index: int, total_ideas: int
    ) -> None:
        super().__init__()
        self.idea = idea
        self.idea_index = idea_index
        self.total_ideas = total_ideas


class BatchSimulatedResponse(Message):
    """LLM-simulated user response in batch mode."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class BatchIdeaComplete(Message):
    """One idea finished processing in batch mode."""

    def __init__(self, idea_index: int, result: object) -> None:
        super().__init__()
        self.idea_index = idea_index
        self.result = result


class BatchComplete(Message):
    """All ideas in the batch have been processed."""

    def __init__(self, results: list, output_path: str) -> None:
        super().__init__()
        self.results = results
        self.output_path = output_path
