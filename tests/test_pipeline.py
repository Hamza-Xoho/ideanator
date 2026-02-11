"""Integration tests for the ARISE pipeline orchestration."""

import random

from tests.conftest import (
    MockLLMClient,
    MOCK_EXTRACT_RESPONSE,
    MOCK_SYNTHESIZE_RESPONSE,
    MOCK_VALIDATE_RESPONSE,
)

from ideanator.pipeline import run_arise_for_idea


def _make_refactor_responses():
    """Return the three refactoring stage mock responses."""
    return [MOCK_EXTRACT_RESPONSE, MOCK_SYNTHESIZE_RESPONSE, MOCK_VALIDATE_RESPONSE]


class TestRunAriseForIdea:
    def test_short_idea_runs_all_four_phases(self):
        """A short (<20 word) idea triggers safety net → all 4 phases run."""
        random.seed(42)
        client = MockLLMClient(
            responses=[
                # Vagueness: model says NONE but idea is short → safety net
                "NONE",
                # Anchor Q
                "[REFLECTION] Interesting.\n"
                "[QUESTION 1] What sparked this language idea?\n"
                "[QUESTION 2] Beginners or advanced?",
                # Anchor simulated response
                "I struggled learning Spanish in college.",
                # Reveal Q
                "[REFLECTION] College Spanish struggle.\n"
                "[QUESTION 1] What broke about language tools?\n"
                "[QUESTION 2] Speed or depth?",
                # Reveal simulated response
                "Tools felt robotic.",
                # Imagine Q
                "[REFLECTION] You want something human.\n"
                "[QUESTION 1] If perfect, what would it feel like?\n"
                "[QUESTION 2] Scale of 1-10?",
                # Imagine simulated response
                "Like talking to a patient friend.",
                # Scope Q
                "[REFLECTION] Great vision.\n"
                "[QUESTION 1] What could make this fail?\n"
                "[QUESTION 2] Smallest test version?",
                # Scope simulated response
                "Getting enough users to start.",
                # Legacy synthesis
                "[IDEA]: Language learning through conversation.\n"
                "[WHO]: College students\n"
                "[PROBLEM]: Robotic tools\n"
                "[MOTIVATION]: Personal struggle\n"
                "[VISION]: Like a patient friend\n"
                "[RISKS]: User acquisition\n"
                "[MVP]: Simple chat bot\n"
                "[DIFFERENTIATION]: Human-feeling",
                # Refactoring stages
                *_make_refactor_responses(),
            ]
        )

        idea = "I want to build a language learning app."
        result = run_arise_for_idea(client, idea)

        assert result.phases_executed == ["anchor", "reveal", "imagine", "scope"]

    def test_specified_idea_runs_anchor_and_scope_only(self):
        """A well-specified (long) idea where model says NONE → anchor + scope only."""
        random.seed(42)
        client = MockLLMClient(
            responses=[
                # Vagueness: NONE and idea is long → all covered
                "NONE",
                # Anchor Q
                "[REFLECTION] Detailed idea.\n"
                "[QUESTION 1] What sparked this?\n"
                "[QUESTION 2] Focus on A or B?",
                # Anchor simulated response
                "Personal experience drove this.",
                # Scope Q
                "[REFLECTION] Well thought out.\n"
                "[QUESTION 1] Biggest risk?\n"
                "[QUESTION 2] Smallest test?",
                # Scope simulated response
                "Technical complexity.",
                # Legacy synthesis
                "[IDEA]: Refined idea\n[WHO]: Target\n[PROBLEM]: Pain",
                # Refactoring stages
                *_make_refactor_responses(),
            ]
        )

        idea = (
            "I want to create a language learning app specifically for college "
            "students who struggle with conversational Spanish because current "
            "tools are too gamified and not focused on real dialogue practice."
        )
        result = run_arise_for_idea(client, idea)

        assert result.phases_executed == ["anchor", "scope"]

    def test_synthesis_is_always_produced(self):
        """Final result always contains a synthesis string."""
        random.seed(42)
        client = MockLLMClient(
            responses=[
                "NONE",  # vagueness (short idea → safety net)
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response.",
                "SYNTHESIS OUTPUT",
                # Refactoring stages
                *_make_refactor_responses(),
            ]
        )

        result = run_arise_for_idea(client, "I want to make an app.")
        assert result.synthesis == "SYNTHESIS OUTPUT"

    def test_refactored_output_is_produced(self):
        """Result includes three-stage refactored output."""
        random.seed(42)
        client = MockLLMClient(
            responses=[
                "NONE",  # vagueness (short idea → safety net)
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response.",
                "SYNTHESIS OUTPUT",
                # Refactoring stages
                *_make_refactor_responses(),
            ]
        )

        result = run_arise_for_idea(client, "I want to make an app.")

        assert result.refactored is not None
        assert result.refactored.one_liner != ""
        assert result.refactored.problem != ""
        assert result.refactored.solution != ""
        assert result.refactored.audience != ""
        assert result.refactored.differentiator != ""
        assert len(result.refactored.open_questions) > 0

    def test_refactored_includes_validation(self):
        """Refactored output includes validation results."""
        random.seed(42)
        client = MockLLMClient(
            responses=[
                "NONE",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response.",
                "SYNTHESIS OUTPUT",
                *_make_refactor_responses(),
            ]
        )

        result = run_arise_for_idea(client, "I want to make an app.")

        assert result.refactored.validation is not None
        assert result.refactored.validation.confidence >= 0.8
        assert result.refactored.validation.critique == "PASS"

    def test_refactored_includes_exploration_status(self):
        """Refactored output includes programmatic exploration status."""
        random.seed(42)
        client = MockLLMClient(
            responses=[
                "NONE",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response with enough words to count as substantive for testing.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response with enough words to count as substantive for testing.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response with enough words to count as substantive for testing.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Response with enough words to count as substantive for testing.",
                "SYNTHESIS OUTPUT",
                *_make_refactor_responses(),
            ]
        )

        result = run_arise_for_idea(client, "I want to make an app.")

        assert result.refactored.exploration_status is not None
        status = result.refactored.exploration_status
        # All 4 phases ran, so all dimensions should be at least partially explored
        assert status.motivation in ("well_explored", "partially_explored")
        assert status.audience in ("well_explored", "partially_explored")

    def test_generic_questions_are_flagged(self):
        """Questions with no keyword overlap are flagged as generic."""
        random.seed(42)
        client = MockLLMClient(
            responses=[
                "NONE",  # safety net triggers
                # Anchor: generic questions (no keywords from "fitness tracker")
                "[REFLECTION] Interesting.\n"
                "[QUESTION 1] What are your long-term goals?\n"
                "[QUESTION 2] Do you prefer working alone or in teams?",
                "I love running.",
                # Reveal
                "[REFLECTION] Runner.\n"
                "[QUESTION 1] What frustrates you about fitness tracking?\n"
                "[QUESTION 2] Speed or accuracy?",
                "Current apps are inaccurate.",
                # Imagine
                "[REFLECTION] Accuracy matters.\n"
                "[QUESTION 1] Perfect fitness tracker experience?\n"
                "[QUESTION 2] Scale?",
                "Seamless and accurate.",
                # Scope
                "[REFLECTION] Clear.\n"
                "[QUESTION 1] Biggest risk for the fitness tracker?\n"
                "[QUESTION 2] Smallest version?",
                "Hardware costs.",
                # Legacy synthesis
                "Summary.",
                # Refactoring stages
                *_make_refactor_responses(),
            ]
        )

        result = run_arise_for_idea(
            client, "I want to make a fitness tracker for runners."
        )

        # The anchor questions "long-term goals" and "working alone or teams"
        # have zero keyword overlap with "fitness tracker runners"
        assert len(result.generic_flags) > 0
        assert any("goals" in f.question for f in result.generic_flags)

    def test_conversation_log_format(self):
        """Conversation contains [Interviewer — ...] and [User] markers."""
        random.seed(42)
        client = MockLLMClient(
            responses=[
                "NONE",
                "[REFLECTION] R\n[QUESTION 1] Q1?\n[QUESTION 2] Q2?",
                "My response.",
                "[REFLECTION] R\n[QUESTION 1] Q1?\n[QUESTION 2] Q2?",
                "Another response.",
                "[REFLECTION] R\n[QUESTION 1] Q1?\n[QUESTION 2] Q2?",
                "Yet another.",
                "[REFLECTION] R\n[QUESTION 1] Q1?\n[QUESTION 2] Q2?",
                "Final response.",
                "Synthesis.",
                *_make_refactor_responses(),
            ]
        )

        result = run_arise_for_idea(client, "I want to build a cooking app.")

        # Check that interviewer turns exist
        interviewer_turns = [
            t for t in result.conversation if t.role == "interviewer"
        ]
        assert len(interviewer_turns) > 0

        # Check that simulated user turns exist
        user_turns = [
            t for t in result.conversation if t.role == "user_simulated"
        ]
        assert len(user_turns) > 0

    def test_anchor_receives_raw_idea_not_conversation(self):
        """The anchor phase should receive the raw idea as user_message."""
        random.seed(42)
        client = MockLLMClient(
            responses=[
                "PERSONAL_MOTIVATION\nTARGET_AUDIENCE\nCORE_PROBLEM\n"
                "SUCCESS_VISION\nCONSTRAINTS_RISKS\nDIFFERENTIATION",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Sim response.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Sim response.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Sim response.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Sim response.",
                "Synthesis.",
                *_make_refactor_responses(),
            ]
        )

        idea = "I want to build an amazing cooking assistant."
        run_arise_for_idea(client, idea)

        # Call index 1 = anchor phase question generation
        # (index 0 = vagueness assessment)
        anchor_call = client.calls[1]
        assert anchor_call["user_message"] == idea

        # Call index 3 = reveal phase question generation
        # (index 2 = anchor simulated response)
        reveal_call = client.calls[3]
        assert "Original idea:" in reveal_call["user_message"]

    def test_vagueness_assessment_in_result(self):
        """Result includes structured vagueness assessment."""
        random.seed(42)
        client = MockLLMClient(
            responses=[
                "CORE_PROBLEM\nDIFFERENTIATION",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Sim.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Sim.",
                "[REFLECTION] R\n[QUESTION 1] Q1\n[QUESTION 2] Q2",
                "Sim.",
                "Synth.",
                *_make_refactor_responses(),
            ]
        )

        result = run_arise_for_idea(
            client,
            "I want to build a meditation app for stressed college students.",
        )

        assert "dimensions" in result.vagueness_assessment
        assert "score" in result.vagueness_assessment
        assert result.vagueness_assessment["dimensions"]["core_problem"] is False
        assert result.vagueness_assessment["dimensions"]["differentiation"] is False
