"""Tests for phase determination and prompt building."""

import random

from ideanator.phases import build_phase_prompt, determine_phases, get_example_for_phase
from ideanator.types import Dimension, DimensionCoverage, Phase


class TestDeterminePhases:
    def test_all_dimensions_missing(self):
        """All missing → anchor, reveal, imagine, scope."""
        dims = DimensionCoverage()
        dims.mark_all_missing()
        phases = determine_phases(dims)
        assert phases == [Phase.ANCHOR, Phase.REVEAL, Phase.IMAGINE, Phase.SCOPE]

    def test_all_dimensions_covered(self):
        """All covered → anchor + scope only (both always run)."""
        dims = DimensionCoverage()  # Defaults to all True
        phases = determine_phases(dims)
        assert phases == [Phase.ANCHOR, Phase.SCOPE]

    def test_missing_core_problem_adds_reveal(self):
        dims = DimensionCoverage()
        dims.coverage[Dimension.CORE_PROBLEM] = False
        phases = determine_phases(dims)
        assert Phase.REVEAL in phases

    def test_missing_target_audience_adds_reveal(self):
        dims = DimensionCoverage()
        dims.coverage[Dimension.TARGET_AUDIENCE] = False
        phases = determine_phases(dims)
        assert Phase.REVEAL in phases

    def test_missing_success_vision_adds_imagine(self):
        dims = DimensionCoverage()
        dims.coverage[Dimension.SUCCESS_VISION] = False
        phases = determine_phases(dims)
        assert Phase.IMAGINE in phases

    def test_anchor_always_first(self):
        dims = DimensionCoverage()
        dims.mark_all_missing()
        phases = determine_phases(dims)
        assert phases[0] == Phase.ANCHOR

    def test_scope_always_last(self):
        dims = DimensionCoverage()
        dims.mark_all_missing()
        phases = determine_phases(dims)
        assert phases[-1] == Phase.SCOPE

    def test_order_preserved(self):
        """Phase order is always anchor → reveal → imagine → scope."""
        dims = DimensionCoverage()
        dims.mark_all_missing()
        phases = determine_phases(dims)

        phase_indices = {p: i for i, p in enumerate(phases)}
        if Phase.REVEAL in phase_indices and Phase.IMAGINE in phase_indices:
            assert phase_indices[Phase.REVEAL] < phase_indices[Phase.IMAGINE]

    def test_missing_constraints_does_not_add_extra_phase(self):
        """constraints_risks missing doesn't add a new phase (scope always runs)."""
        dims = DimensionCoverage()
        dims.coverage[Dimension.CONSTRAINTS_RISKS] = False
        phases = determine_phases(dims)
        assert phases == [Phase.ANCHOR, Phase.SCOPE]


class TestGetExampleForPhase:
    def test_returns_dict_with_user_and_response(self):
        random.seed(42)
        example = get_example_for_phase(Phase.ANCHOR)
        assert "user" in example
        assert "response" in example

    def test_anchor_examples_exist(self):
        random.seed(42)
        example = get_example_for_phase(Phase.ANCHOR)
        assert len(example["user"]) > 0
        assert len(example["response"]) > 0

    def test_falls_back_to_anchor_for_unknown(self):
        """If phase key not found, falls back to anchor examples."""
        from ideanator.prompts import get_example_pool

        pool = get_example_pool()
        pool_with_missing = {k: v for k, v in pool.items() if k != "scope"}
        random.seed(42)
        example = get_example_for_phase(Phase.SCOPE, pool=pool_with_missing)
        # Should return an anchor example as fallback
        assert "user" in example


class TestBuildPhasePrompt:
    def test_anchor_prompt_contains_still_need(self):
        random.seed(42)
        prompt = build_phase_prompt(
            Phase.ANCHOR,
            "Original idea: test\n",
            ["their personal motivation and story"],
        )
        assert "their personal motivation and story" in prompt

    def test_anchor_prompt_contains_example(self):
        random.seed(42)
        prompt = build_phase_prompt(
            Phase.ANCHOR,
            "Original idea: test\n",
            ["their personal motivation and story"],
        )
        assert "[REFLECTION]" in prompt
        assert "[QUESTION 1]" in prompt

    def test_reveal_prompt_contains_conversation(self):
        random.seed(42)
        conv = "Original idea: test\n[Interviewer]: some question\n"
        prompt = build_phase_prompt(
            Phase.REVEAL,
            conv,
            ["the specific pain point being solved"],
        )
        assert conv in prompt

    def test_empty_uncovered_uses_default(self):
        random.seed(42)
        prompt = build_phase_prompt(
            Phase.ANCHOR,
            "Original idea: test\n",
            [],
        )
        assert "their personal motivation" in prompt

    def test_uncovered_truncated_to_three(self):
        random.seed(42)
        uncovered = [
            "their personal motivation and story",
            "who specifically this is for",
            "the specific pain point being solved",
            "what success looks like concretely",
        ]
        prompt = build_phase_prompt(
            Phase.ANCHOR,
            "Original idea: test\n",
            uncovered,
        )
        # Only first 3 should appear in still_need
        assert "what success looks like concretely" not in prompt
