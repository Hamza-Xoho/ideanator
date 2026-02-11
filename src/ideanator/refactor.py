"""Three-stage idea refactoring engine: Extract → Synthesize → Validate.

Implements the pipeline described in the refactoring design document:
- Stage 1 (Extract): Parses conversation into structured dimensions with citations
- Stage 2 (Synthesize): Chain-of-density adapted synthesis with banned words
- Stage 3 (Validate): Faithfulness, completeness, and sycophancy checks

If validation confidence < 0.8, triggers a self-refine loop back to Stage 2.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ideanator.llm import LLMClient
from ideanator.models import (
    Contradiction,
    ExplorationStatus,
    ExtractedInsights,
    RefactoredIdea,
    ValidationResult,
)
from ideanator.types import (
    ConversationTurn,
    Dimension,
    PHASE_DIMENSION_MAP,
    Phase,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# Maximum self-refine iterations before accepting result as-is
_MAX_REFINE_ROUNDS = 2

# Confidence threshold — below this triggers self-refine loop
_CONFIDENCE_THRESHOLD = 0.8


# ── Prompt Loading ────────────────────────────────────────────────────


@lru_cache(maxsize=4)
def _load_stage_config(stage: str) -> dict[str, Any]:
    """Load and cache a stage YAML config file."""
    path = _PROMPTS_DIR / f"{stage}.yml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clear_refactor_cache() -> None:
    """Clear cached prompt configs (for testing)."""
    _load_stage_config.cache_clear()


def _get_banned_words(cfg: dict[str, Any]) -> str:
    """Extract banned phrases as a comma-separated string."""
    phrases = cfg.get("anti_patterns", {}).get("banned_phrases", [])
    return ", ".join(str(p) for p in phrases)


# ── Stage 1: Extract ──────────────────────────────────────────────────


def extract(client: LLMClient, transcript: str) -> ExtractedInsights:
    """Parse the conversation transcript into structured dimensions.

    Uses low temperature (0.3) for deterministic extraction.
    Organizes thematically, cites conversation turns.
    """
    cfg = _load_stage_config("extract")
    settings = cfg["model"]["settings"]

    raw = client.call(
        system_prompt=cfg["system_prompt"],
        user_message=cfg["user_template"].format(transcript=transcript),
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )

    return _parse_extraction(raw)


def _parse_extraction(raw: str) -> ExtractedInsights:
    """Parse the LLM extraction response into an ExtractedInsights model.

    Tries JSON parsing first, falls back to best-effort text parsing.
    """
    # Try to extract JSON from the response
    json_str = _extract_json(raw)
    if json_str:
        try:
            data = json.loads(json_str)
            return ExtractedInsights.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("JSON extraction parse failed: %s", e)

    # Fallback: best-effort text parsing
    return _extract_from_text(raw)


def _extract_json(text: str) -> str | None:
    """Extract a JSON object from text, handling markdown code fences."""
    # Try code fence first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)

    # Try bare JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)

    return None


def _extract_from_text(raw: str) -> ExtractedInsights:
    """Best-effort text extraction when JSON parsing fails."""
    insights = ExtractedInsights()

    field_patterns = {
        "problem": r"(?:problem|pain)[:\s]*(.+?)(?=\n[a-z]|\Z)",
        "audience": r"(?:audience|who|target)[:\s]*(.+?)(?=\n[a-z]|\Z)",
        "solution": r"(?:solution|what|approach)[:\s]*(.+?)(?=\n[a-z]|\Z)",
        "differentiation": r"(?:differentiation|different|unique)[:\s]*(.+?)(?=\n[a-z]|\Z)",
        "motivation": r"(?:motivation|why|cares)[:\s]*(.+?)(?=\n[a-z]|\Z)",
    }

    for field_name, pattern in field_patterns.items():
        m = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
        if m:
            setattr(insights, field_name, m.group(1).strip())

    return insights


# ── Stage 2: Synthesize ──────────────────────────────────────────────


def synthesize(
    client: LLMClient,
    insights: ExtractedInsights,
    transcript: str,
    critique: str | None = None,
) -> str:
    """Transform extracted insights into a refined idea statement.

    Uses chain-of-density adapted prompting with banned word enforcement.
    If critique is provided (from self-refine loop), includes it in the prompt.
    """
    cfg = _load_stage_config("synthesize")
    settings = cfg["model"]["settings"]

    banned = _get_banned_words(cfg)
    system = cfg["system_prompt"].replace("{banned_words}", banned)

    if critique:
        system += (
            f"\n\nPREVIOUS ATTEMPT CRITIQUE — address these issues:\n{critique}\n"
            "Fix the specific problems identified above while preserving what was good."
        )

    user_msg = cfg["user_template"].format(
        insights=insights.model_dump_json(indent=2),
        transcript=transcript,
    )

    raw = client.call(
        system_prompt=system,
        user_message=user_msg,
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )

    return raw


# ── Stage 3: Validate ────────────────────────────────────────────────


def validate(client: LLMClient, statement: str, transcript: str) -> ValidationResult:
    """Check faithfulness, completeness, and sycophancy of the refined statement.

    Returns a ValidationResult with confidence score.
    confidence < 0.8 signals the pipeline should self-refine.
    """
    cfg = _load_stage_config("validate")
    settings = cfg["model"]["settings"]

    raw = client.call(
        system_prompt=cfg["system_prompt"],
        user_message=cfg["user_template"].format(
            statement=statement,
            transcript=transcript,
        ),
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )

    return _parse_validation(raw)


def _parse_validation(raw: str) -> ValidationResult:
    """Parse the LLM validation response into a ValidationResult model."""
    json_str = _extract_json(raw)
    if json_str:
        try:
            data = json.loads(json_str)
            return ValidationResult.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Validation JSON parse failed: %s", e)

    # Fallback: assume moderate confidence, pass through raw as critique
    return ValidationResult(
        confidence=0.6,
        critique=f"Could not parse validation output. Raw: {raw[:200]}",
    )


# ── Exploration Status (Programmatic) ────────────────────────────────


def compute_exploration_status(
    conversation: list[ConversationTurn],
    phases_executed: list[str],
) -> ExplorationStatus:
    """Compute exploration status programmatically from conversation structure.

    NOT delegated to LLM self-assessment — computed by analyzing:
    - Which phases ran (and what dimensions they cover)
    - How many follow-up exchanges occurred per dimension
    - Whether user responses contained substantive content

    Labels:
    - well_explored: Phase ran AND user gave substantive response (>20 words)
    - partially_explored: Phase ran but user response was thin (<20 words)
    - not_explored: Relevant phase did not run
    """
    status = ExplorationStatus()

    # Map dimensions to the phases that cover them
    dim_phase_map: dict[str, list[str]] = {
        "motivation": ["anchor"],
        "audience": ["anchor", "reveal"],
        "problem": ["reveal"],
        "solution": ["imagine", "scope"],
        "differentiation": ["scope"],
    }

    # Count user response words per phase
    phase_user_words: dict[str, int] = {}
    for turn in conversation:
        if turn.role in ("user", "user_simulated"):
            word_count = len(turn.content.split())
            phase_user_words[turn.phase] = (
                phase_user_words.get(turn.phase, 0) + word_count
            )

    phases_set = set(phases_executed)

    for dim, relevant_phases in dim_phase_map.items():
        ran_phases = [p for p in relevant_phases if p in phases_set]
        if not ran_phases:
            setattr(status, dim, "not_explored")
        else:
            total_words = sum(phase_user_words.get(p, 0) for p in ran_phases)
            if total_words >= 20:
                setattr(status, dim, "well_explored")
            else:
                setattr(status, dim, "partially_explored")

    return status


# ── Contradiction Detection ──────────────────────────────────────────


def detect_contradictions(
    conversation: list[ConversationTurn],
) -> list[Contradiction]:
    """Detect potential contradictions in user responses across phases.

    Resolution hierarchy from the design doc:
    1. Prefer later answers (thinking evolved)
    2. Prefer more specific answers
    3. Synthesize both where valid
    4. Flag irreconcilable contradictions for user review

    This function identifies candidates; the extract stage's LLM call
    also flags contradictions for a more nuanced analysis.
    """
    user_turns = [
        t for t in conversation if t.role in ("user", "user_simulated")
    ]

    if len(user_turns) < 2:
        return []

    contradictions: list[Contradiction] = []

    # Simple heuristic: check for negation patterns between turns
    negation_pairs = [
        ("not", "yes"), ("don't", "do"), ("can't", "can"),
        ("won't", "will"), ("isn't", "is"), ("no", "yes"),
    ]

    for i, earlier in enumerate(user_turns):
        for later in user_turns[i + 1:]:
            earlier_lower = earlier.content.lower()
            later_lower = later.content.lower()

            for neg, pos in negation_pairs:
                # Check if one turn negates and the other affirms
                # with overlapping topic words
                earlier_words = set(earlier_lower.split())
                later_words = set(later_lower.split())
                shared_topic = earlier_words & later_words - {
                    "i", "the", "a", "to", "is", "it", "and", "or", "but",
                    "my", "that", "this", "in", "of", "for", "on", "with",
                }

                if shared_topic and (
                    (neg in earlier_lower and pos in later_lower)
                    or (pos in earlier_lower and neg in later_lower)
                ):
                    contradictions.append(
                        Contradiction(
                            earlier=earlier.content[:100],
                            later=later.content[:100],
                            turns=f"{earlier.phase} vs {later.phase}",
                        )
                    )
                    break

    return contradictions


# ── Output Parsing ───────────────────────────────────────────────────


def parse_synthesis_output(raw: str) -> RefactoredIdea:
    """Parse the structured synthesis output into a RefactoredIdea.

    Expected sections: [ONE-LINER], [PROBLEM], [SOLUTION], [AUDIENCE],
    [DIFFERENTIATOR], [OPEN QUESTIONS].
    Falls back gracefully if parsing fails.
    """
    idea = RefactoredIdea(raw_synthesis=raw)

    section_map = {
        "one_liner": r"\[ONE[- ]?LINER\]\s*:?\s*(.+?)(?=\[|$)",
        "problem": r"\[PROBLEM\]\s*:?\s*(.+?)(?=\[|$)",
        "solution": r"\[SOLUTION\]\s*:?\s*(.+?)(?=\[|$)",
        "audience": r"\[AUDIENCE\]\s*:?\s*(.+?)(?=\[|$)",
        "differentiator": r"\[DIFFERENTIATOR\]\s*:?\s*(.+?)(?=\[|$)",
    }

    for field_name, pattern in section_map.items():
        m = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        if m:
            setattr(idea, field_name, m.group(1).strip())

    # Parse open questions
    oq_match = re.search(
        r"\[OPEN QUESTIONS?\]\s*:?\s*(.+?)(?=\[|$)", raw, re.DOTALL | re.IGNORECASE
    )
    if oq_match:
        questions_text = oq_match.group(1).strip()
        questions = re.findall(r"[-•*]\s*(.+)", questions_text)
        idea.open_questions = [q.strip() for q in questions if q.strip()]

    # If parsing failed entirely, use raw as the one-liner
    if not idea.one_liner and not idea.problem:
        idea.one_liner = raw[:200] if raw else ""

    return idea


# ── Exploration Status Formatting ────────────────────────────────────


_STATUS_LABELS = {
    "well_explored": "\u2705 Well-explored",
    "partially_explored": "\u26a0\ufe0f Partially explored",
    "not_explored": "\U0001f532 Not yet explored",
}


def format_exploration_status(status: ExplorationStatus) -> str:
    """Format exploration status with qualitative labels for display."""
    lines = []
    for field in ["problem", "audience", "solution", "differentiation", "motivation"]:
        value = getattr(status, field)
        label = _STATUS_LABELS.get(value, value)
        lines.append(f"  {field.title()}: {label}")
    return "\n".join(lines)


# ── Main Pipeline ────────────────────────────────────────────────────


def refactor_idea(
    client: LLMClient,
    transcript: str,
    conversation: list[ConversationTurn],
    phases_executed: list[str],
    callback: Any = None,
) -> RefactoredIdea:
    """Run the three-stage refactoring pipeline: Extract → Synthesize → Validate.

    If validation confidence < 0.8, triggers a self-refine loop (up to 2 rounds).

    Args:
        client: LLM client for all model calls.
        transcript: The full conversation log text.
        conversation: Structured conversation turns for programmatic analysis.
        phases_executed: List of phase names that ran.
        callback: Optional progress callback.

    Returns:
        RefactoredIdea with structured output and pipeline metadata.
    """
    _emit(callback, "status", "Stage 1: Extracting structured insights...")

    # Stage 1: Extract
    insights = extract(client, transcript)
    _emit(callback, "status", f"Extracted {len(insights.key_phrases)} key phrases, "
          f"{len(insights.contradictions)} contradictions")

    # Stage 2: Synthesize
    _emit(callback, "status", "Stage 2: Synthesizing refined statement...")
    raw_synthesis = synthesize(client, insights, transcript)

    # Stage 3: Validate
    _emit(callback, "status", "Stage 3: Validating faithfulness and completeness...")
    validation = validate(client, raw_synthesis, transcript)

    refinement_rounds = 0

    # Self-refine loop if confidence < threshold
    while (
        validation.confidence < _CONFIDENCE_THRESHOLD
        and refinement_rounds < _MAX_REFINE_ROUNDS
    ):
        refinement_rounds += 1
        _emit(
            callback,
            "status",
            f"Self-refine round {refinement_rounds}: confidence {validation.confidence:.2f} "
            f"< {_CONFIDENCE_THRESHOLD} — revising...",
        )

        raw_synthesis = synthesize(
            client, insights, transcript, critique=validation.critique
        )
        validation = validate(client, raw_synthesis, transcript)

    if validation.confidence >= _CONFIDENCE_THRESHOLD:
        _emit(callback, "status",
              f"Validation passed (confidence: {validation.confidence:.2f})")
    else:
        _emit(callback, "status",
              f"Accepted after {refinement_rounds} rounds "
              f"(confidence: {validation.confidence:.2f})")

    # Parse structured output
    idea = parse_synthesis_output(raw_synthesis)

    # Compute exploration status programmatically
    exploration = compute_exploration_status(conversation, phases_executed)

    # Detect contradictions programmatically
    programmatic_contradictions = detect_contradictions(conversation)

    # Merge LLM-detected and programmatic contradictions
    all_contradictions = list(insights.contradictions) + programmatic_contradictions

    # Populate metadata
    idea.extracted_insights = insights
    idea.validation = validation
    idea.exploration_status = exploration
    idea.contradictions_found = all_contradictions
    idea.refinement_rounds = refinement_rounds
    idea.raw_synthesis = raw_synthesis

    return idea


def _emit(callback: Any, event: str, data: str) -> None:
    """Send a progress event if callback is present."""
    if callback:
        callback(event, data)
