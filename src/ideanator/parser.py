"""Response parsing, thinking-model stripping, and anti-generic question checking."""

from __future__ import annotations

import re

from ideanator.types import ParsedResponse


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from thinking-model output.

    Handles both closed tags and unclosed tags (model hit token limit
    mid-thought and never emitted </think>). Must be called before any
    structured output parsing.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)  # unclosed tag
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_structured_response(raw_text: str) -> ParsedResponse:
    """Extract [REFLECTION], [QUESTION 1], [QUESTION 2] from model output.

    Uses a tiered approach:
    1. Strip any <think> blocks first.
    2. Try strict regex (exact tag casing).
    3. Fall back to case-insensitive fuzzy regex (handles wrong casing,
       alternative delimiters like **REFLECTION**, <REFLECTION>, etc.).
    4. If nothing matches, return raw text as the 'clean' output.
    """
    cleaned = strip_thinking(raw_text)

    reflection, question_1, question_2 = _parse_strict(cleaned)

    if not reflection and not question_1:
        reflection, question_1, question_2 = _parse_fuzzy(cleaned)

    clean_parts = []
    if reflection:
        clean_parts.append(reflection)
    if question_1:
        clean_parts.append(question_1)
    if question_2:
        clean_parts.append(question_2)

    clean = "\n\n".join(clean_parts) if clean_parts else cleaned

    return ParsedResponse(
        reflection=reflection,
        question_1=question_1,
        question_2=question_2,
        raw=raw_text,
        clean=clean,
    )


def _parse_strict(text: str) -> tuple[str, str, str]:
    """Strict parsing: exact [TAG] casing."""
    reflection = ""
    question_1 = ""
    question_2 = ""

    m = re.search(r"\[REFLECTION\]\s*(.+?)(?=\[QUESTION|\Z)", text, re.DOTALL)
    if m:
        reflection = m.group(1).strip()

    m = re.search(
        r"\[QUESTION 1\]\s*(.+?)(?=\[QUESTION 2\]|\Z)", text, re.DOTALL
    )
    if m:
        question_1 = m.group(1).strip()

    m = re.search(r"\[QUESTION 2\]\s*(.+?)(?=\[|\Z)", text, re.DOTALL)
    if m:
        question_2 = m.group(1).strip()

    return reflection, question_1, question_2


def _parse_fuzzy(text: str) -> tuple[str, str, str]:
    """Fuzzy parsing: case-insensitive, handles alternative delimiters.

    Matches [Reflection], <REFLECTION>, **Reflection**, etc.
    """
    reflection = ""
    question_1 = ""
    question_2 = ""

    m = re.search(
        r"(?:\[|<|\*\*)\s*reflection\s*(?:\]|>|\*\*)\s*:?\s*(.*?)"
        r"(?=(?:\[|<|\*\*)\s*question|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        reflection = m.group(1).strip()

    matches = re.findall(
        r"(?:\[|<|\*\*)\s*question\s*\d*\s*(?:\]|>|\*\*)\s*:?\s*(.*?)"
        r"(?=(?:\[|<|\*\*)\s*question\s*\d|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if len(matches) >= 1:
        question_1 = matches[0].strip()
    if len(matches) >= 2:
        question_2 = matches[1].strip()

    return reflection, question_1, question_2


# Exact stop words from the original ARISE v2 implementation.
STOP_WORDS = frozenset(
    {
        "want",
        "make",
        "create",
        "build",
        "develop",
        "design",
        "that",
        "helps",
        "people",
        "with",
        "their",
        "them",
        "this",
        "would",
        "platform",
        "allows",
        "from",
        "about",
        "have",
        "will",
        "into",
        "your",
        "they",
        "could",
        "should",
        "more",
        "most",
        "also",
    }
)


def is_question_generic(question: str, idea: str) -> bool:
    """Check if a question is too generic (could apply to ANY idea).

    Heuristic: extract 4+ character keywords from the idea (minus stop
    words), then check if the question contains any of them. Zero overlap
    means the question is generic.
    """
    if not question or not idea:
        return True

    idea_words = {w.lower().strip(".,!?") for w in idea.split() if len(w) >= 4}
    idea_keywords = idea_words - STOP_WORDS

    if not idea_keywords:
        return False  # Can't assess if idea has no qualifying keywords

    question_lower = question.lower()
    matches = sum(1 for kw in idea_keywords if kw in question_lower)
    return matches == 0
