"""Core data structures for the ARISE pipeline.

All types use Pydantic for validation and serialization.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Phase(str, Enum):
    """The four ARISE questioning phases."""

    ANCHOR = "anchor"
    REVEAL = "reveal"
    IMAGINE = "imagine"
    SCOPE = "scope"


class Dimension(str, Enum):
    """The six vagueness dimensions assessed for each idea."""

    PERSONAL_MOTIVATION = "personal_motivation"
    TARGET_AUDIENCE = "target_audience"
    CORE_PROBLEM = "core_problem"
    SUCCESS_VISION = "success_vision"
    CONSTRAINTS_RISKS = "constraints_risks"
    DIFFERENTIATION = "differentiation"


DIMENSION_LABELS: dict[Dimension, str] = {
    Dimension.PERSONAL_MOTIVATION: "their personal motivation and story",
    Dimension.TARGET_AUDIENCE: "who specifically this is for",
    Dimension.CORE_PROBLEM: "the specific pain point being solved",
    Dimension.SUCCESS_VISION: "what success looks like concretely",
    Dimension.CONSTRAINTS_RISKS: "potential risks and what could go wrong",
    Dimension.DIFFERENTIATION: "what makes this different from alternatives",
}

# Maps each phase to the dimensions it covers upon completion.
PHASE_DIMENSION_MAP: dict[Phase, list[Dimension]] = {
    Phase.ANCHOR: [Dimension.PERSONAL_MOTIVATION, Dimension.TARGET_AUDIENCE],
    Phase.REVEAL: [Dimension.CORE_PROBLEM],
    Phase.IMAGINE: [Dimension.SUCCESS_VISION],
    Phase.SCOPE: [Dimension.CONSTRAINTS_RISKS, Dimension.DIFFERENTIATION],
}

PHASE_LABELS: dict[Phase, str] = {
    Phase.ANCHOR: "Phase 1 — ANCHOR (Personal Reality)",
    Phase.REVEAL: "Phase 2 — REVEAL (Deeper Job)",
    Phase.IMAGINE: "Phase 3 — IMAGINE (Ideal Outcome)",
    Phase.SCOPE: "Phase 4 — SCOPE (Boundaries & Risks)",
}


class DimensionCoverage(BaseModel):
    """Tracks which of the 6 dimensions are covered vs missing."""

    coverage: dict[Dimension, bool] = Field(
        default_factory=lambda: {d: True for d in Dimension}
    )

    @property
    def covered_count(self) -> int:
        return sum(1 for v in self.coverage.values() if v)

    @property
    def score_str(self) -> str:
        return f"{self.covered_count}/6"

    def uncovered_labels(self) -> list[str]:
        return [DIMENSION_LABELS[k] for k, v in self.coverage.items() if not v]

    def mark_all_missing(self) -> None:
        self.coverage = {d: False for d in Dimension}

    def mark_covered(self, dims: list[Dimension]) -> None:
        for d in dims:
            self.coverage[d] = True


class ParsedResponse(BaseModel):
    """Structured output from parsing an ARISE phase response."""

    reflection: str = ""
    question_1: str = ""
    question_2: str = ""
    raw: str = ""
    clean: str = ""


class GenericFlag(BaseModel):
    """Records a question flagged as too generic."""

    phase: str
    question: str
    flag: str = "GENERIC — could apply to any idea"


class ConversationTurn(BaseModel):
    """A single turn in the ARISE conversation."""

    phase: str
    role: str  # "interviewer" or "user_simulated"
    content: str
    parsed: ParsedResponse | None = None


class IdeaResult(BaseModel):
    """Complete result of running the ARISE pipeline for one idea."""

    original_idea: str
    timestamp: str
    vagueness_assessment: dict[str, Any] = Field(default_factory=dict)
    phases_executed: list[str]
    conversation: list[ConversationTurn]
    generic_flags: list[GenericFlag] = Field(default_factory=list)
    synthesis: str
    # Three-stage refactoring output (populated when refactoring engine runs)
    refactored: Any = None

    model_config = {"arbitrary_types_allowed": True}
