"""Tests for TUI Message classes.

Each Message subclass should carry the correct attributes and be
instantiable without a running Textual app.

Skipped entirely if textual is not installed.
"""

from __future__ import annotations

import pytest

try:
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

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

pytestmark = pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")


class TestInteractiveMessages:
    """Interactive pipeline message classes."""

    def test_pipeline_status(self):
        msg = PipelineStatus("Scoring vagueness...")
        assert msg.text == "Scoring vagueness..."

    def test_vagueness_result(self):
        phases = ["ANCHOR", "REVEAL", "SCOPE"]
        msg = VaguenessResult(text="Covered: 3/6", phases=phases)
        assert msg.text == "Covered: 3/6"
        assert msg.phases == ["ANCHOR", "REVEAL", "SCOPE"]

    def test_phase_started(self):
        msg = PhaseStarted(phase_label="ANCHOR", phase_index=0, total_phases=3)
        assert msg.phase_label == "ANCHOR"
        assert msg.phase_index == 0
        assert msg.total_phases == 3

    def test_interviewer_message(self):
        msg = InterviewerMessage(text="Tell me about your idea.")
        assert msg.text == "Tell me about your idea."

    def test_user_prompt_requested(self):
        msg = UserPromptRequested(phase_label="ANCHOR")
        assert msg.phase_label == "ANCHOR"

    def test_generic_flag_detected(self):
        msg = GenericFlagDetected(question="What is your target audience?")
        assert msg.question == "What is your target audience?"

    def test_synthesis_complete(self):
        result_obj = {"key": "value"}
        msg = SynthesisComplete(synthesis="Final output", result=result_obj)
        assert msg.synthesis == "Final output"
        assert msg.result == {"key": "value"}

    def test_pipeline_error(self):
        msg = PipelineError(error="Connection failed")
        assert msg.error == "Connection failed"


class TestBatchMessages:
    """Batch pipeline message classes."""

    def test_batch_idea_started(self):
        msg = BatchIdeaStarted(idea="My idea", idea_index=2, total_ideas=5)
        assert msg.idea == "My idea"
        assert msg.idea_index == 2
        assert msg.total_ideas == 5

    def test_batch_simulated_response(self):
        msg = BatchSimulatedResponse(text="Simulated answer")
        assert msg.text == "Simulated answer"

    def test_batch_idea_complete(self):
        msg = BatchIdeaComplete(idea_index=1, result={"done": True})
        assert msg.idea_index == 1
        assert msg.result == {"done": True}

    def test_batch_complete(self):
        results = [{"idea": "one"}, {"idea": "two"}]
        msg = BatchComplete(results=results, output_path="out.json")
        assert len(msg.results) == 2
        assert msg.output_path == "out.json"
