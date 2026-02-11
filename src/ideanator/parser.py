"""Response parsing and anti-generic question checking."""

from __future__ import annotations

import re

from ideanator.types import ParsedResponse


def parse_structured_response(raw_text: str) -> ParsedResponse:
    """
    Extract [REFLECTION], [QUESTION 1], [QUESTION 2] from model output.

    Falls back to raw text as the 'clean' output if the structured format
    was not followed by the model.
    """
    reflection = ""
    question_1 = ""
    question_2 = ""

    reflection_match = re.search(
        r"\[REFLECTION\]\s*(.+?)(?=\[QUESTION|\Z)", raw_text, re.DOTALL
    )
    q1_match = re.search(
        r"\[QUESTION 1\]\s*(.+?)(?=\[QUESTION 2\]|\Z)", raw_text, re.DOTALL
    )
    q2_match = re.search(
        r"\[QUESTION 2\]\s*(.+?)(?=\[|\Z)", raw_text, re.DOTALL
    )

    if reflection_match:
        reflection = reflection_match.group(1).strip()
    if q1_match:
        question_1 = q1_match.group(1).strip()
    if q2_match:
        question_2 = q2_match.group(1).strip()

    clean_parts = []
    if reflection:
        clean_parts.append(reflection)
    if question_1:
        clean_parts.append(question_1)
    if question_2:
        clean_parts.append(question_2)

    clean = "\n\n".join(clean_parts) if clean_parts else raw_text

    return ParsedResponse(
        reflection=reflection,
        question_1=question_1,
        question_2=question_2,
        raw=raw_text,
        clean=clean,
    )


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
    """
    Check if a question is too generic (could apply to ANY idea).

    Heuristic: extract 4+ character keywords from the idea (minus stop
    words), then check if the question contains any of them. Zero overlap
    means the question is generic.
    """
    idea_words = {w.lower().strip(".,!?") for w in idea.split() if len(w) >= 4}
    idea_keywords = idea_words - STOP_WORDS
    question_lower = question.lower()
    matches = sum(1 for kw in idea_keywords if kw in question_lower)
    return matches == 0
