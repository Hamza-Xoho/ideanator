"""Tests for the inverted vagueness scorer."""

from tests.conftest import MockLLMClient

from ideanator.scorer import assess_vagueness
from ideanator.types import Dimension


class TestAssessVagueness:
    def test_all_dimensions_missing(self):
        """Model lists all 6 dimensions → all marked False."""
        client = MockLLMClient(
            responses=[
                "PERSONAL_MOTIVATION\nTARGET_AUDIENCE\nCORE_PROBLEM\n"
                "SUCCESS_VISION\nCONSTRAINTS_RISKS\nDIFFERENTIATION"
            ]
        )
        idea = "I want to build a platform for connecting people with mentors."
        dims, raw = assess_vagueness(client, idea)

        for dim in Dimension:
            assert dims.coverage[dim] is False
        assert dims.covered_count == 0

    def test_no_dimensions_missing_long_idea(self):
        """Model says NONE for a 25+ word idea → all stay True."""
        client = MockLLMClient(responses=["NONE"])
        idea = (
            "I want to create a language learning app for college students "
            "who struggle with conversational Spanish because I tried Duolingo "
            "and found it too gamified and not focused enough."
        )
        dims, raw = assess_vagueness(client, idea)

        for dim in Dimension:
            assert dims.coverage[dim] is True
        assert dims.covered_count == 6

    def test_safety_net_triggers_for_short_idea(self):
        """Model says NONE for <20 word idea → overrides to all missing."""
        client = MockLLMClient(responses=["NONE"])
        idea = "I want to make a budgeting app."
        dims, raw = assess_vagueness(client, idea)

        for dim in Dimension:
            assert dims.coverage[dim] is False
        assert dims.covered_count == 0

    def test_safety_net_does_not_trigger_at_threshold(self):
        """Model says NONE for exactly 20 words → safety net does NOT trigger."""
        client = MockLLMClient(responses=["NONE"])
        # Exactly 20 words
        idea = (
            "I want to create a very specific and detailed language learning "
            "application for college students who need conversational practice daily"
        )
        assert len(idea.split()) == 20
        dims, raw = assess_vagueness(client, idea)

        for dim in Dimension:
            assert dims.coverage[dim] is True

    def test_partial_dimensions_missing(self):
        """Model lists only some dimensions → only those are False."""
        client = MockLLMClient(
            responses=["CORE_PROBLEM\nDIFFERENTIATION"]
        )
        idea = "I want to create a language learning app for college freshmen."
        dims, raw = assess_vagueness(client, idea)

        assert dims.coverage[Dimension.CORE_PROBLEM] is False
        assert dims.coverage[Dimension.DIFFERENTIATION] is False
        assert dims.coverage[Dimension.PERSONAL_MOTIVATION] is True
        assert dims.coverage[Dimension.TARGET_AUDIENCE] is True
        assert dims.coverage[Dimension.SUCCESS_VISION] is True
        assert dims.coverage[Dimension.CONSTRAINTS_RISKS] is True

    def test_case_insensitive_parsing(self):
        """Dimension names are matched case-insensitively."""
        client = MockLLMClient(
            responses=["personal_motivation\nTarget_Audience"]
        )
        idea = "I want to build a tool for freelancers to track their time."
        dims, raw = assess_vagueness(client, idea)

        assert dims.coverage[Dimension.PERSONAL_MOTIVATION] is False
        assert dims.coverage[Dimension.TARGET_AUDIENCE] is False

    def test_returns_raw_response(self):
        """The raw LLM output string is returned alongside dimensions."""
        raw_text = "CORE_PROBLEM\nSUCCESS_VISION"
        client = MockLLMClient(responses=[raw_text])
        idea = "I want to build a fitness tracker for marathon runners."
        dims, raw = assess_vagueness(client, idea)

        assert raw == raw_text

    def test_uses_correct_temperature_and_tokens(self):
        """Vagueness assessment uses decision temperature (0.0) and tokens (200)."""
        client = MockLLMClient(responses=["NONE"])
        idea = "I want to build something meaningful for a specific group of struggling teachers."
        assess_vagueness(client, idea)

        assert client.calls[0]["temperature"] == 0.0
        assert client.calls[0]["max_tokens"] == 200
