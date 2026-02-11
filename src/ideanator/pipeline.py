"""ARISE pipeline orchestration — the core idea development loop."""

from __future__ import annotations

import logging
from datetime import datetime

from ideanator.config import TEMPERATURES, TOKENS
from ideanator.llm import LLMClient
from ideanator.parser import is_question_generic, parse_structured_response
from ideanator.phases import build_phase_prompt, determine_phases
from ideanator.prompts import get_simulated_user_prompt, get_synthesis_prompt
from ideanator.scorer import assess_vagueness
from ideanator.types import (
    ConversationTurn,
    GenericFlag,
    IdeaResult,
    Phase,
    PHASE_DIMENSION_MAP,
    PHASE_LABELS,
)

logger = logging.getLogger(__name__)


def run_arise_for_idea(
    client: LLMClient,
    idea: str,
    callback: _ProgressCallback | None = None,
) -> IdeaResult:
    """
    Execute the full ARISE pipeline for a single idea.

    Args:
        client: LLM client to use for all model calls.
        idea: The raw idea text to develop.
        callback: Optional callback for progress reporting.

    Returns:
        IdeaResult with complete conversation, assessment, and synthesis.
    """
    _emit(callback, "status", "Scoring vagueness (inverted prompt)...")

    # Step 1: Vagueness Calibration
    dims, raw_score = assess_vagueness(client, idea)
    uncovered = dims.uncovered_labels()
    phases = determine_phases(dims)

    _emit(
        callback,
        "vagueness",
        f"Covered: {dims.score_str} | Missing: {', '.join(uncovered) or 'None'} | "
        f"Phases: {' → '.join(p.value for p in phases)}",
    )

    result = IdeaResult(
        original_idea=idea,
        timestamp=datetime.now().isoformat(),
        vagueness_assessment={
            "dimensions": {d.value: v for d, v in dims.coverage.items()},
            "score": dims.score_str,
            "uncovered": uncovered,
            "raw_response": raw_score,
        },
        phases_executed=[],
        conversation=[],
        generic_flags=[],
        synthesis="",
    )

    # Step 2: Multi-Phase ARISE Loop
    conversation_log = f"Original idea: {idea}\n"

    for phase in phases:
        phase_label = PHASE_LABELS[phase]
        _emit(callback, "phase_start", phase_label)

        current_uncovered = dims.uncovered_labels()
        system_prompt = build_phase_prompt(phase, conversation_log, current_uncovered)

        # Key behavior: anchor uses raw idea as user message;
        # all other phases use full conversation log
        user_msg = idea if phase == Phase.ANCHOR else conversation_log

        raw_response = client.call(
            system_prompt=system_prompt,
            user_message=user_msg,
            temperature=TEMPERATURES.questioning,
            max_tokens=TOKENS.question,
        )

        # Parse structured response
        parsed = parse_structured_response(raw_response)
        display_text = parsed.clean
        _emit(callback, "interviewer", display_text)

        # Anti-generic check for each question
        for q_attr in ("question_1", "question_2"):
            q = getattr(parsed, q_attr, "")
            if q and is_question_generic(q, idea):
                result.generic_flags.append(
                    GenericFlag(phase=phase.value, question=q)
                )
                _emit(callback, "generic_flag", q)

        conversation_log += f"\n[Interviewer — {phase_label}]:\n{display_text}\n"
        result.conversation.append(
            ConversationTurn(
                phase=phase.value,
                role="interviewer",
                content=display_text,
                parsed=parsed,
            )
        )

        # Simulate user response
        sim_prompt = get_simulated_user_prompt().format(original_idea=idea)
        sim_response = client.call(
            system_prompt=sim_prompt,
            user_message=display_text,
            temperature=TEMPERATURES.simulation,
            max_tokens=TOKENS.simulation,
        )
        _emit(callback, "user_sim", sim_response)

        conversation_log += f"\n[User]:\n{sim_response}\n"
        result.conversation.append(
            ConversationTurn(
                phase=phase.value,
                role="user_simulated",
                content=sim_response,
            )
        )

        # Update dimension coverage (phase-based, not response-based)
        for dim in PHASE_DIMENSION_MAP.get(phase, []):
            dims.coverage[dim] = True

        result.phases_executed.append(phase.value)

    # Step 3: Synthesis
    _emit(callback, "status", "Synthesizing...")
    synth_prompt = get_synthesis_prompt().format(conversation=conversation_log)
    synthesis = client.call(
        system_prompt=synth_prompt,
        user_message="Please synthesize now.",
        temperature=TEMPERATURES.synthesis,
        max_tokens=TOKENS.synthesis,
    )
    result.synthesis = synthesis
    _emit(callback, "synthesis", synthesis)

    return result


def run_arise_interactive(
    client: LLMClient,
    idea: str,
    callback: _ProgressCallback | None = None,
) -> IdeaResult:
    """
    Execute the ARISE pipeline interactively — real user answers questions.

    Instead of simulating user responses, prompts the user via the callback.
    Falls back to simulated mode if no callback with 'prompt_user' is available.
    """
    _emit(callback, "status", "Scoring vagueness (inverted prompt)...")

    # Step 1: Vagueness Calibration
    dims, raw_score = assess_vagueness(client, idea)
    uncovered = dims.uncovered_labels()
    phases = determine_phases(dims)

    _emit(
        callback,
        "vagueness",
        f"Covered: {dims.score_str} | Missing: {', '.join(uncovered) or 'None'} | "
        f"Phases: {' → '.join(p.value for p in phases)}",
    )

    result = IdeaResult(
        original_idea=idea,
        timestamp=datetime.now().isoformat(),
        vagueness_assessment={
            "dimensions": {d.value: v for d, v in dims.coverage.items()},
            "score": dims.score_str,
            "uncovered": uncovered,
            "raw_response": raw_score,
        },
        phases_executed=[],
        conversation=[],
        generic_flags=[],
        synthesis="",
    )

    # Step 2: Multi-Phase ARISE Loop
    conversation_log = f"Original idea: {idea}\n"

    for phase in phases:
        phase_label = PHASE_LABELS[phase]
        _emit(callback, "phase_start", phase_label)

        current_uncovered = dims.uncovered_labels()
        system_prompt = build_phase_prompt(phase, conversation_log, current_uncovered)

        user_msg = idea if phase == Phase.ANCHOR else conversation_log

        raw_response = client.call(
            system_prompt=system_prompt,
            user_message=user_msg,
            temperature=TEMPERATURES.questioning,
            max_tokens=TOKENS.question,
        )

        parsed = parse_structured_response(raw_response)
        display_text = parsed.clean
        _emit(callback, "interviewer", display_text)

        # Anti-generic check
        for q_attr in ("question_1", "question_2"):
            q = getattr(parsed, q_attr, "")
            if q and is_question_generic(q, idea):
                result.generic_flags.append(
                    GenericFlag(phase=phase.value, question=q)
                )

        conversation_log += f"\n[Interviewer — {phase_label}]:\n{display_text}\n"
        result.conversation.append(
            ConversationTurn(
                phase=phase.value,
                role="interviewer",
                content=display_text,
                parsed=parsed,
            )
        )

        # Get real user response via callback
        user_response = _prompt_user(callback, phase_label)

        conversation_log += f"\n[User]:\n{user_response}\n"
        result.conversation.append(
            ConversationTurn(
                phase=phase.value,
                role="user",
                content=user_response,
            )
        )

        # Update dimension coverage
        for dim in PHASE_DIMENSION_MAP.get(phase, []):
            dims.coverage[dim] = True

        result.phases_executed.append(phase.value)

    # Step 3: Synthesis
    _emit(callback, "status", "Synthesizing...")
    synth_prompt = get_synthesis_prompt().format(conversation=conversation_log)
    synthesis = client.call(
        system_prompt=synth_prompt,
        user_message="Please synthesize now.",
        temperature=TEMPERATURES.synthesis,
        max_tokens=TOKENS.synthesis,
    )
    result.synthesis = synthesis
    _emit(callback, "synthesis", synthesis)

    return result


# ── Callback helpers ───────────────────────────────────────────────────

from typing import Callable

_ProgressCallback = Callable[[str, str], str | None]


def _emit(callback: _ProgressCallback | None, event: str, data: str) -> None:
    """Send a progress event to the callback, if present."""
    if callback:
        callback(event, data)


def _prompt_user(callback: _ProgressCallback | None, phase_label: str) -> str:
    """Request user input via the callback."""
    if callback:
        result = callback("prompt_user", phase_label)
        if result:
            return result
    return input("\nYour response: ")
