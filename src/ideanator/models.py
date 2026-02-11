"""Pydantic models for the three-stage idea refactoring pipeline.

These models enforce structural correctness between pipeline stages
and enable programmatic validation of intermediate outputs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Contradiction(BaseModel):
    """A contradiction found between different parts of the conversation."""

    earlier: str = Field(description="What the user said first")
    later: str = Field(description="What the user said later")
    turns: str = Field(default="", description="Turn references, e.g. 'Turn 2 vs Turn 5'")


class ExtractedInsights(BaseModel):
    """Structured extraction from the conversation transcript.

    This is the output of Stage 1 (Extract) and the input to Stage 2 (Synthesize).
    Every field should cite conversation turns for auditability.
    """

    problem: str = Field(default="NOT DISCUSSED", description="The specific pain point")
    audience: str = Field(default="NOT DISCUSSED", description="Who this is for")
    solution: str = Field(default="NOT DISCUSSED", description="What the idea does")
    differentiation: str = Field(default="NOT DISCUSSED", description="What makes it different")
    motivation: str = Field(default="NOT DISCUSSED", description="Why this person cares")
    key_phrases: list[str] = Field(default_factory=list, description="User's exact words")
    contradictions: list[Contradiction] = Field(default_factory=list)
    user_register: str = Field(default="casual", description="casual|formal|technical")
    unresolved: list[str] = Field(default_factory=list, description="Unanswered questions")


class FaithfulnessResult(BaseModel):
    """Faithfulness check: each claim verified against the transcript."""

    supported_count: int = 0
    implied_count: int = 0
    unsupported_count: int = 0
    unsupported_claims: list[str] = Field(default_factory=list)


class CompletenessResult(BaseModel):
    """Completeness check: are the four must-have dimensions covered?"""

    problem: bool = False
    audience: bool = False
    solution: bool = False
    differentiation: bool = False
    missing: list[str] = Field(default_factory=list)


class SycophancyResult(BaseModel):
    """Sycophancy check: did the refinement over-polish the user's words?"""

    flags: list[str] = Field(default_factory=list)
    severity: str = Field(default="none", description="none|mild|significant")


class ValidationResult(BaseModel):
    """Complete validation output from Stage 3.

    confidence < 0.8 triggers a self-refine loop back to Stage 2.
    """

    faithfulness: FaithfulnessResult = Field(default_factory=FaithfulnessResult)
    completeness: CompletenessResult = Field(default_factory=CompletenessResult)
    sycophancy: SycophancyResult = Field(default_factory=SycophancyResult)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    critique: str = Field(default="", description="What needs fixing, or 'PASS'")


class ExplorationStatus(BaseModel):
    """Programmatic exploration status for each dimension.

    Computed from conversation structure, NOT from LLM self-assessment.
    Labels: well_explored, partially_explored, not_explored.
    """

    problem: str = "not_explored"
    audience: str = "not_explored"
    solution: str = "not_explored"
    differentiation: str = "not_explored"
    motivation: str = "not_explored"


class RefactoredIdea(BaseModel):
    """Complete output of the three-stage refactoring pipeline."""

    one_liner: str = ""
    problem: str = ""
    solution: str = ""
    audience: str = ""
    differentiator: str = ""
    open_questions: list[str] = Field(default_factory=list)

    # Pipeline metadata
    extracted_insights: ExtractedInsights | None = None
    validation: ValidationResult | None = None
    exploration_status: ExplorationStatus | None = None
    contradictions_found: list[Contradiction] = Field(default_factory=list)
    refinement_rounds: int = 0
    raw_synthesis: str = ""
