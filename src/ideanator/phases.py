"""Phase determination and prompt building for the ARISE pipeline."""

from __future__ import annotations

import random

from ideanator.prompts import get_example_pool, get_phase_prompt_template
from ideanator.types import Dimension, DimensionCoverage, Phase


def determine_phases(dims: DimensionCoverage) -> list[Phase]:
    """
    Determine which ARISE phases to run based on dimension coverage.

    Anchor and Scope ALWAYS run. Reveal runs if core_problem or
    target_audience are missing. Imagine runs if success_vision is missing.
    """
    phases = [Phase.ANCHOR]  # ALWAYS: no user types their personal story unprompted

    if (
        not dims.coverage[Dimension.CORE_PROBLEM]
        or not dims.coverage[Dimension.TARGET_AUDIENCE]
    ):
        phases.append(Phase.REVEAL)

    if not dims.coverage[Dimension.SUCCESS_VISION]:
        phases.append(Phase.IMAGINE)

    phases.append(Phase.SCOPE)  # ALWAYS: nobody states constraints in initial ideas

    return phases


def get_example_for_phase(phase: Phase, pool: dict | None = None) -> dict:
    """Randomly select one few-shot example for the given ARISE phase."""
    examples_pool = pool or get_example_pool()
    examples = examples_pool.get(phase.value, examples_pool["anchor"])
    return random.choice(examples)


_DEFAULT_NEEDS: dict[Phase, str] = {
    Phase.ANCHOR: "their personal motivation",
    Phase.REVEAL: "the deeper problem",
    Phase.IMAGINE: "their success vision",
    Phase.SCOPE: "constraints and risks",
}


def build_phase_prompt(
    phase: Phase,
    conversation_log: str,
    uncovered: list[str],
    example_pool: dict | None = None,
    prompts: dict | None = None,
) -> str:
    """
    Build the system prompt for a given ARISE phase.

    Selects a random few-shot example, determines the 'still_need' string,
    and formats the phase template with all dynamic parts.
    """
    ex = get_example_for_phase(phase, example_pool)
    still_need = ", ".join(uncovered[:3]) if uncovered else _DEFAULT_NEEDS[phase]

    template = get_phase_prompt_template(phase.value, prompts)
    return template.format(
        still_need=still_need,
        example_user=ex["user"],
        example_response=ex["response"],
        conversation=conversation_log,
    )
