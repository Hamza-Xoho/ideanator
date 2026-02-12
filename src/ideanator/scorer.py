"""Inverted vagueness assessment for ideas."""

from __future__ import annotations

import logging

from ideanator.config import TEMPERATURES, TOKENS, VAGUENESS_WORD_THRESHOLD
from ideanator.llm import LLMClient
from ideanator.prompts import get_vagueness_prompt
from ideanator.types import Dimension, DimensionCoverage

logger = logging.getLogger(__name__)


def assess_vagueness(
    client: LLMClient,
    idea: str,
    vagueness_prompt: str | None = None,
) -> tuple[DimensionCoverage, str]:
    """
    Inverted vagueness assessment: asks what is MISSING, not what is present.

    Returns a DimensionCoverage (True = covered, False = missing) and the
    raw LLM output string.

    The inverted approach counteracts RLHF sycophancy: small models tend to
    say YES to "Does this have X?" but are happy to list what's MISSING.

    Safety net: if the model says "NONE" but the idea is under the word
    threshold, all dimensions are overridden to missing (short ideas cannot
    possibly cover all 6 dimensions).
    """
    prompt = vagueness_prompt or get_vagueness_prompt()
    dims = DimensionCoverage()  # Starts with all dimensions True

    raw = client.call(
        system_prompt=prompt,
        user_message=idea,
        temperature=TEMPERATURES.decision,
        max_tokens=TOKENS.decision,
    )

    # Parse: any dimension NAME found in output = it is missing
    raw_upper = raw.upper()
    for dim in Dimension:
        if dim.value.upper() in raw_upper:
            dims.coverage[dim] = False

    # SAFETY NET: if model says NONE but idea is clearly vague
    # (under word threshold), override to all-missing
    word_count = len(idea.split())
    if "NONE" in raw_upper and word_count < VAGUENESS_WORD_THRESHOLD:
        logger.debug(
            "Safety net triggered: NONE response for %d-word idea (threshold: %d)",
            word_count,
            VAGUENESS_WORD_THRESHOLD,
        )
        dims.mark_all_missing()

    logger.debug("Vagueness score: %s", dims.score_str)
    return dims, raw
