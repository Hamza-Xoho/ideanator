"""Tests for prompt loading and content integrity."""

from ideanator.prompts import (
    get_example_pool,
    get_phase_prompt_template,
    get_simulated_user_prompt,
    get_synthesis_prompt,
    get_vagueness_prompt,
    load_prompts,
)


class TestLoadPrompts:
    def test_loads_without_error(self):
        prompts = load_prompts()
        assert isinstance(prompts, dict)

    def test_has_required_top_level_keys(self):
        prompts = load_prompts()
        required = {
            "vagueness_prompt",
            "phase_prompts",
            "simulated_user_prompt",
            "synthesis_prompt",
            "example_pool",
        }
        assert required.issubset(prompts.keys())


class TestVaguenessPrompt:
    def test_contains_inverted_framing(self):
        prompt = get_vagueness_prompt()
        assert "MISSING" in prompt

    def test_contains_does_not_count_guards(self):
        prompt = get_vagueness_prompt()
        assert "does NOT count" in prompt

    def test_contains_none_instruction(self):
        prompt = get_vagueness_prompt()
        assert "NONE" in prompt

    def test_contains_all_six_dimensions(self):
        prompt = get_vagueness_prompt()
        dimensions = [
            "PERSONAL_MOTIVATION",
            "TARGET_AUDIENCE",
            "CORE_PROBLEM",
            "SUCCESS_VISION",
            "CONSTRAINTS_RISKS",
            "DIFFERENTIATION",
        ]
        for dim in dimensions:
            assert dim in prompt, f"Missing dimension: {dim}"


class TestPhasePrompts:
    def test_all_four_phases_present(self):
        for phase in ("anchor", "reveal", "imagine", "scope"):
            template = get_phase_prompt_template(phase)
            assert len(template) > 0, f"Empty template for {phase}"

    def test_anchor_has_required_format_markers(self):
        template = get_phase_prompt_template("anchor")
        assert "[REFLECTION]" in template
        assert "[QUESTION 1]" in template
        assert "[QUESTION 2]" in template

    def test_anchor_has_placeholders(self):
        template = get_phase_prompt_template("anchor")
        assert "{still_need}" in template
        assert "{example_user}" in template
        assert "{example_response}" in template

    def test_reveal_has_conversation_placeholder(self):
        template = get_phase_prompt_template("reveal")
        assert "{conversation}" in template

    def test_imagine_has_miracle_question(self):
        template = get_phase_prompt_template("imagine")
        assert "miracle" in template.lower() or "PERFECTLY" in template

    def test_scope_has_smallest_version(self):
        template = get_phase_prompt_template("scope")
        assert "smallest version" in template


class TestSimulatedUserPrompt:
    def test_has_original_idea_placeholder(self):
        prompt = get_simulated_user_prompt()
        assert "{original_idea}" in prompt

    def test_contains_rules(self):
        prompt = get_simulated_user_prompt()
        assert "3-5 sentences" in prompt

    def test_contains_authenticity_guidance(self):
        prompt = get_simulated_user_prompt()
        assert "product manager" in prompt.lower()


class TestSynthesisPrompt:
    def test_has_conversation_placeholder(self):
        prompt = get_synthesis_prompt()
        assert "{conversation}" in prompt

    def test_contains_all_headers(self):
        prompt = get_synthesis_prompt()
        headers = [
            "[IDEA]",
            "[WHO]",
            "[PROBLEM]",
            "[MOTIVATION]",
            "[VISION]",
            "[RISKS]",
            "[MVP]",
            "[DIFFERENTIATION]",
        ]
        for header in headers:
            assert header in prompt, f"Missing header: {header}"

    def test_contains_not_explored_instruction(self):
        prompt = get_synthesis_prompt()
        assert "Not yet explored" in prompt


class TestExamplePool:
    def test_has_all_four_phases(self):
        pool = get_example_pool()
        for phase in ("anchor", "reveal", "imagine", "scope"):
            assert phase in pool, f"Missing phase in example pool: {phase}"

    def test_anchor_has_three_examples(self):
        pool = get_example_pool()
        assert len(pool["anchor"]) == 3

    def test_reveal_has_two_examples(self):
        pool = get_example_pool()
        assert len(pool["reveal"]) == 2

    def test_examples_have_user_and_response(self):
        pool = get_example_pool()
        for phase, examples in pool.items():
            for i, ex in enumerate(examples):
                assert "user" in ex, f"{phase}[{i}] missing 'user'"
                assert "response" in ex, f"{phase}[{i}] missing 'response'"

    def test_examples_contain_structured_format(self):
        pool = get_example_pool()
        for phase, examples in pool.items():
            for ex in examples:
                assert "[REFLECTION]" in ex["response"], (
                    f"{phase} example missing [REFLECTION]"
                )
