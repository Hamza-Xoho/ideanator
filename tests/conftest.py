"""Shared test fixtures for the ideanator test suite."""

from __future__ import annotations

import pytest

from ideanator.prompts import clear_cache


class MockLLMClient:
    """
    A mock LLM client that returns predetermined responses in sequence.

    Tracks all calls for assertion in tests.
    """

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.calls: list[dict] = []

    def call(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.6,
        max_tokens: int = 300,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


@pytest.fixture
def mock_client():
    """Create a MockLLMClient with default responses."""
    return MockLLMClient(
        responses=[
            # Vagueness assessment — lists all as missing
            "PERSONAL_MOTIVATION\nTARGET_AUDIENCE\nCORE_PROBLEM\n"
            "SUCCESS_VISION\nCONSTRAINTS_RISKS\nDIFFERENTIATION",
            # Anchor phase response
            "[REFLECTION] This sounds interesting.\n"
            "[QUESTION 1] What made you think of this?\n"
            "[QUESTION 2] Are you thinking more about A or B?",
            # Simulated user response (for anchor)
            "I've always been passionate about this area.",
            # Reveal phase response
            "[REFLECTION] You care deeply about this.\n"
            "[QUESTION 1] What's broken about current solutions?\n"
            "[QUESTION 2] Is it more about cost or quality?",
            # Simulated user response (for reveal)
            "The current tools are too complicated.",
            # Imagine phase response
            "[REFLECTION] Simplicity is your north star.\n"
            "[QUESTION 1] If this worked perfectly, what would it feel like?\n"
            "[QUESTION 2] On a scale of 1-10, how clear is your vision?",
            # Simulated user response (for imagine)
            "It would feel effortless and intuitive.",
            # Scope phase response
            "[REFLECTION] You've come a long way.\n"
            "[QUESTION 1] What could make this fail?\n"
            "[QUESTION 2] What's the smallest version you could test?",
            # Simulated user response (for scope)
            "I think adoption would be the hardest part.",
            # Synthesis
            "[IDEA]: A simplified tool for the domain\n"
            "[WHO]: Frustrated users\n"
            "[PROBLEM]: Current tools too complex\n"
            "[MOTIVATION]: Personal experience\n"
            "[VISION]: Effortless experience\n"
            "[RISKS]: Adoption\n"
            "[MVP]: Simple prototype\n"
            "[DIFFERENTIATION]: Simplicity focus",
        ]
    )


@pytest.fixture
def sample_idea():
    """A sample idea string for testing."""
    return "I want to build an app that helps people learn languages."


@pytest.fixture
def short_idea():
    """A very short idea (under 20 words) for safety net testing."""
    return "I want to make a budgeting app."


@pytest.fixture
def detailed_idea():
    """A detailed idea with many dimensions covered."""
    return (
        "I want to create a language learning app specifically for college students "
        "who struggle with conversational Spanish. I've tried Duolingo myself and "
        "found it too gamified and not focused enough on real conversations. "
        "Success would look like students being able to hold a 5-minute conversation "
        "after 30 days. The biggest risk is that students won't stick with it. "
        "Unlike Duolingo, this would focus purely on conversation practice."
    )


@pytest.fixture(autouse=True)
def _clear_prompt_cache():
    """Clear the prompt cache before each test to ensure isolation."""
    clear_cache()
    yield
    clear_cache()
