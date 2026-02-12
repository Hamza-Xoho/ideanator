"""Tests for TUI Message classes.

Each Message subclass should carry the correct attributes and be
instantiable without a running Textual app.
"""

from __future__ import annotations

import pytest

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


class TestInteractiveMessages:
    """Messages posted during interactive pipeline execution."""

    def test_pipeline_status(self):
        m = PipelineStatus("Starting…")
        assert m.text == "Starting…"

    def test_vagueness_result(self):
        m = VaguenessResult("high — 2 dimensions missing", ["anchor", "reveal"])
        assert m.text == "high — 2 dimensions missing"
        assert m.phases == ["anchor", "reveal"]

    def test_phase_started(self):
        m = PhaseStarted("anchor", 0, 3)
        assert m.phase_label == "anchor"
        assert m.phase_index == 0
        assert m.total_phases == 3

    def test_interviewer_message(self):
        m = InterviewerMessage("What problem?")
        assert m.text == "What problem?"

    def test_user_prompt_requested(self):
        m = UserPromptRequested("anchor")
        assert m.phase_label == "anchor"

    def test_generic_flag_detected(self):
        m = GenericFlagDetected("Tell me more")
        assert m.question == "Tell me more"

    def test_synthesis_complete(self):
        m = SynthesisComplete("Summary here", result=object())
        assert m.synthesis == "Summary here"
        assert m.result is not None

    def test_pipeline_error(self):
        m = PipelineError("Something broke")
        assert m.error == "Something broke"


class TestBatchMessages:
    """Messages posted during batch pipeline execution."""

    def test_batch_idea_started(self):
        m = BatchIdeaStarted("My idea", 0, 5)
        assert m.idea == "My idea"
        assert m.idea_index == 0
        assert m.total_ideas == 5

    def test_batch_simulated_response(self):
        m = BatchSimulatedResponse("Simulated answer")
        assert m.text == "Simulated answer"

    def test_batch_idea_complete(self):
        m = BatchIdeaComplete(2, result=object())
        assert m.idea_index == 2
        assert m.result is not None

    def test_batch_complete(self):
        m = BatchComplete([], "/tmp/out.json")
        assert m.output_path == "/tmp/out.json"
        assert m.results == []
