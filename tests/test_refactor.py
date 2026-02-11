"""Tests for the three-stage idea refactoring engine."""

from __future__ import annotations

import json

from tests.conftest import (
    MockLLMClient,
    MOCK_EXTRACT_RESPONSE,
    MOCK_SYNTHESIZE_RESPONSE,
    MOCK_VALIDATE_RESPONSE,
)

from ideanator.models import (
    Contradiction,
    ExplorationStatus,
    ExtractedInsights,
    RefactoredIdea,
    ValidationResult,
)
from ideanator.refactor import (
    compute_exploration_status,
    detect_contradictions,
    extract,
    format_exploration_status,
    parse_synthesis_output,
    refactor_idea,
    synthesize,
    validate,
)
from ideanator.types import ConversationTurn


# ── Pydantic Model Tests ─────────────────────────────────────────────


class TestExtractedInsights:
    def test_defaults(self):
        """All fields have sensible defaults."""
        insights = ExtractedInsights()
        assert insights.problem == "NOT DISCUSSED"
        assert insights.audience == "NOT DISCUSSED"
        assert insights.key_phrases == []
        assert insights.contradictions == []
        assert insights.user_register == "casual"

    def test_from_json(self):
        """Can parse from JSON response."""
        data = json.loads(MOCK_EXTRACT_RESPONSE)
        insights = ExtractedInsights.model_validate(data)
        assert "too complicated" in insights.problem
        assert len(insights.key_phrases) == 3
        assert insights.user_register == "casual"

    def test_serialization_roundtrip(self):
        """Model can serialize to JSON and back."""
        insights = ExtractedInsights(
            problem="Test problem",
            audience="Test audience",
            key_phrases=["phrase1", "phrase2"],
        )
        data = json.loads(insights.model_dump_json())
        restored = ExtractedInsights.model_validate(data)
        assert restored.problem == "Test problem"
        assert len(restored.key_phrases) == 2


class TestValidationResult:
    def test_defaults(self):
        """Default validation has zero confidence."""
        v = ValidationResult()
        assert v.confidence == 0.0
        assert v.critique == ""

    def test_from_json(self):
        """Can parse from JSON response."""
        data = json.loads(MOCK_VALIDATE_RESPONSE)
        v = ValidationResult.model_validate(data)
        assert v.confidence == 0.85
        assert v.critique == "PASS"
        assert v.faithfulness.supported_count == 5
        assert v.completeness.problem is True
        assert v.sycophancy.severity == "none"

    def test_confidence_bounds(self):
        """Confidence must be between 0 and 1."""
        v = ValidationResult(confidence=0.5)
        assert 0.0 <= v.confidence <= 1.0


# ── Stage 1: Extract Tests ───────────────────────────────────────────


class TestExtract:
    def test_extract_parses_json_response(self):
        """Extract stage parses a well-formed JSON response."""
        client = MockLLMClient(responses=[MOCK_EXTRACT_RESPONSE])
        insights = extract(client, "some transcript")

        assert "too complicated" in insights.problem
        assert len(insights.key_phrases) == 3
        assert insights.user_register == "casual"

    def test_extract_handles_code_fenced_json(self):
        """Extract handles JSON wrapped in markdown code fences."""
        fenced = f"```json\n{MOCK_EXTRACT_RESPONSE}\n```"
        client = MockLLMClient(responses=[fenced])
        insights = extract(client, "some transcript")

        assert "too complicated" in insights.problem

    def test_extract_fallback_on_bad_json(self):
        """Extract falls back gracefully when JSON is malformed."""
        client = MockLLMClient(responses=["Not valid JSON at all"])
        insights = extract(client, "some transcript")

        # Should return an ExtractedInsights with defaults
        assert isinstance(insights, ExtractedInsights)

    def test_extract_uses_correct_temperature(self):
        """Extract call uses low temperature (0.3) for deterministic output."""
        client = MockLLMClient(responses=[MOCK_EXTRACT_RESPONSE])
        extract(client, "transcript")

        assert client.calls[0]["temperature"] == 0.3


# ── Stage 2: Synthesize Tests ────────────────────────────────────────


class TestSynthesize:
    def test_synthesize_returns_text(self):
        """Synthesize returns the raw text from the LLM."""
        client = MockLLMClient(responses=[MOCK_SYNTHESIZE_RESPONSE])
        insights = ExtractedInsights(problem="Test", audience="Users")

        result = synthesize(client, insights, "transcript")
        assert "[ONE-LINER]" in result
        assert "[PROBLEM]" in result

    def test_synthesize_includes_banned_words_in_prompt(self):
        """Synthesize prompt includes banned words from YAML config."""
        client = MockLLMClient(responses=["output"])
        insights = ExtractedInsights()

        synthesize(client, insights, "transcript")

        system_prompt = client.calls[0]["system_prompt"]
        # Should contain some of the banned words in the prompt
        assert "innovative" in system_prompt or "robust" in system_prompt

    def test_synthesize_with_critique(self):
        """Synthesize includes critique in system prompt during self-refine."""
        client = MockLLMClient(responses=["refined output"])
        insights = ExtractedInsights()

        synthesize(client, insights, "transcript", critique="Fix the sycophancy")

        system_prompt = client.calls[0]["system_prompt"]
        assert "Fix the sycophancy" in system_prompt

    def test_synthesize_uses_medium_temperature(self):
        """Synthesize uses temperature 0.5 for creative-but-grounded output."""
        client = MockLLMClient(responses=["output"])
        insights = ExtractedInsights()

        synthesize(client, insights, "transcript")
        assert client.calls[0]["temperature"] == 0.5


# ── Stage 3: Validate Tests ──────────────────────────────────────────


class TestValidate:
    def test_validate_parses_json(self):
        """Validate parses a well-formed JSON response."""
        client = MockLLMClient(responses=[MOCK_VALIDATE_RESPONSE])
        result = validate(client, "statement", "transcript")

        assert result.confidence == 0.85
        assert result.critique == "PASS"

    def test_validate_fallback_on_bad_json(self):
        """Validate falls back with moderate confidence on bad JSON."""
        client = MockLLMClient(responses=["Not valid JSON"])
        result = validate(client, "statement", "transcript")

        assert isinstance(result, ValidationResult)
        assert result.confidence == 0.6  # fallback confidence

    def test_validate_uses_low_temperature(self):
        """Validate uses low temperature (0.2) for deterministic checking."""
        client = MockLLMClient(responses=[MOCK_VALIDATE_RESPONSE])
        validate(client, "statement", "transcript")

        assert client.calls[0]["temperature"] == 0.2


# ── Exploration Status Tests ─────────────────────────────────────────


class TestExplorationStatus:
    def test_all_phases_well_explored(self):
        """All dimensions well-explored when all phases ran with substantive responses."""
        conversation = [
            ConversationTurn(phase="anchor", role="interviewer", content="Q"),
            ConversationTurn(
                phase="anchor",
                role="user_simulated",
                content="I care about this because I struggled with it myself for years and years "
                        "and I really want to help other people who face the same frustrating situation."
            ),
            ConversationTurn(phase="reveal", role="interviewer", content="Q"),
            ConversationTurn(
                phase="reveal",
                role="user_simulated",
                content="The problem is that current tools are way too complicated for normal people "
                        "to use and they end up giving up before they even get started with the basics."
            ),
            ConversationTurn(phase="imagine", role="interviewer", content="Q"),
            ConversationTurn(
                phase="imagine",
                role="user_simulated",
                content="If it worked perfectly it would feel like talking to a patient friend who "
                        "understands you and guides you step by step without making you feel stupid."
            ),
            ConversationTurn(phase="scope", role="interviewer", content="Q"),
            ConversationTurn(
                phase="scope",
                role="user_simulated",
                content="The biggest risk is getting enough users to start with and keeping them "
                        "engaged long term because there are so many alternatives already out there."
            ),
        ]
        phases = ["anchor", "reveal", "imagine", "scope"]

        status = compute_exploration_status(conversation, phases)

        assert status.motivation == "well_explored"
        assert status.problem == "well_explored"

    def test_missing_phase_not_explored(self):
        """Dimensions are not_explored when their phase didn't run."""
        conversation = [
            ConversationTurn(phase="anchor", role="interviewer", content="Q"),
            ConversationTurn(phase="anchor", role="user", content="Short."),
            ConversationTurn(phase="scope", role="interviewer", content="Q"),
            ConversationTurn(phase="scope", role="user", content="Short."),
        ]
        phases = ["anchor", "scope"]

        status = compute_exploration_status(conversation, phases)

        assert status.problem == "not_explored"  # Reveal didn't run

    def test_thin_response_partially_explored(self):
        """Short user responses result in partially_explored status."""
        conversation = [
            ConversationTurn(phase="anchor", role="interviewer", content="Q"),
            ConversationTurn(phase="anchor", role="user_simulated", content="Yes."),
        ]
        phases = ["anchor"]

        status = compute_exploration_status(conversation, phases)

        assert status.motivation == "partially_explored"

    def test_format_exploration_status(self):
        """Format function produces readable output with emoji labels."""
        status = ExplorationStatus(
            problem="well_explored",
            audience="partially_explored",
            solution="not_explored",
            differentiation="well_explored",
            motivation="partially_explored",
        )

        formatted = format_exploration_status(status)
        assert "Well-explored" in formatted
        assert "Partially explored" in formatted
        assert "Not yet explored" in formatted


# ── Contradiction Detection Tests ────────────────────────────────────


class TestContradictionDetection:
    def test_no_contradictions_in_consistent_responses(self):
        """No contradictions detected when responses are consistent."""
        conversation = [
            ConversationTurn(phase="anchor", role="user", content="I love simple tools."),
            ConversationTurn(phase="scope", role="user", content="Simplicity is key for me."),
        ]
        result = detect_contradictions(conversation)
        assert len(result) == 0

    def test_single_turn_no_contradictions(self):
        """Single user turn can't have contradictions."""
        conversation = [
            ConversationTurn(phase="anchor", role="user", content="I want X."),
        ]
        result = detect_contradictions(conversation)
        assert len(result) == 0


# ── Synthesis Output Parsing Tests ───────────────────────────────────


class TestParseSynthesisOutput:
    def test_parses_all_sections(self):
        """Parses all six sections from well-formed output."""
        idea = parse_synthesis_output(MOCK_SYNTHESIZE_RESPONSE)

        assert "simplified tool" in idea.one_liner
        assert "too complicated" in idea.problem
        assert "stripped-down" in idea.solution or "core workflow" in idea.solution
        assert "tried existing tools" in idea.audience or "overwhelmed" in idea.audience
        assert "simplicity" in idea.differentiator
        assert len(idea.open_questions) == 3

    def test_fallback_on_unparseable_output(self):
        """Falls back gracefully when output doesn't match expected format."""
        idea = parse_synthesis_output("Just a plain text response with no sections.")

        assert idea.one_liner != ""  # Should use raw text as fallback
        assert isinstance(idea, RefactoredIdea)

    def test_empty_input(self):
        """Handles empty input gracefully."""
        idea = parse_synthesis_output("")
        assert isinstance(idea, RefactoredIdea)


# ── Full Pipeline Tests ──────────────────────────────────────────────


class TestRefactorIdea:
    def test_full_pipeline_produces_output(self):
        """Full three-stage pipeline produces a RefactoredIdea."""
        client = MockLLMClient(
            responses=[
                MOCK_EXTRACT_RESPONSE,
                MOCK_SYNTHESIZE_RESPONSE,
                MOCK_VALIDATE_RESPONSE,
            ]
        )
        conversation = [
            ConversationTurn(phase="anchor", role="interviewer", content="Q"),
            ConversationTurn(
                phase="anchor",
                role="user_simulated",
                content="I care deeply about making tools simpler for everyone to use in their daily work."
            ),
            ConversationTurn(phase="scope", role="interviewer", content="Q"),
            ConversationTurn(
                phase="scope",
                role="user_simulated",
                content="The risk is that people won't switch from the tools they already know and use."
            ),
        ]

        result = refactor_idea(
            client=client,
            transcript="Test transcript",
            conversation=conversation,
            phases_executed=["anchor", "scope"],
        )

        assert isinstance(result, RefactoredIdea)
        assert result.one_liner != ""
        assert result.validation is not None
        assert result.exploration_status is not None
        assert result.refinement_rounds == 0  # confidence >= 0.8, no refinement needed

    def test_self_refine_loop_triggers(self):
        """Self-refine triggers when validation confidence is below threshold."""
        low_confidence = json.dumps({
            "faithfulness": {"supported_count": 2, "implied_count": 1,
                             "unsupported_count": 2, "unsupported_claims": ["claim1", "claim2"]},
            "completeness": {"problem": True, "audience": False, "solution": True,
                             "differentiation": False, "missing": ["audience", "differentiation"]},
            "sycophancy": {"flags": ["too polished"], "severity": "mild"},
            "confidence": 0.5,
            "critique": "Missing audience and differentiation. Too polished."
        })

        client = MockLLMClient(
            responses=[
                MOCK_EXTRACT_RESPONSE,       # Stage 1: Extract
                MOCK_SYNTHESIZE_RESPONSE,    # Stage 2: Synthesize (first attempt)
                low_confidence,              # Stage 3: Validate (fails)
                MOCK_SYNTHESIZE_RESPONSE,    # Stage 2: Synthesize (refinement 1)
                MOCK_VALIDATE_RESPONSE,      # Stage 3: Validate (passes)
            ]
        )

        result = refactor_idea(
            client=client,
            transcript="Test transcript",
            conversation=[],
            phases_executed=["anchor", "scope"],
        )

        assert result.refinement_rounds == 1
        assert result.validation.confidence >= 0.8

    def test_pipeline_makes_correct_number_of_calls(self):
        """Pipeline makes exactly 3 LLM calls when validation passes."""
        client = MockLLMClient(
            responses=[
                MOCK_EXTRACT_RESPONSE,
                MOCK_SYNTHESIZE_RESPONSE,
                MOCK_VALIDATE_RESPONSE,
            ]
        )

        refactor_idea(
            client=client,
            transcript="Test transcript",
            conversation=[],
            phases_executed=["anchor"],
        )

        # 3 calls: extract + synthesize + validate
        assert client.call_count == 3
