"""Tests for response parsing, thinking-model stripping, and anti-generic checking."""

from ideanator.parser import (
    is_question_generic,
    parse_structured_response,
    strip_thinking,
    _parse_strict,
    _parse_fuzzy,
)


# ── Thinking model stripping ─────────────────────────────────────────


class TestStripThinking:
    def test_removes_closed_think_block(self):
        text = "<think>Some reasoning here.</think>The actual answer."
        assert strip_thinking(text) == "The actual answer."

    def test_removes_multiline_think_block(self):
        text = (
            "<think>\nI need to think about this.\n"
            "Let me consider the options.\n</think>\n"
            "[REFLECTION] Here is my reflection."
        )
        assert "[REFLECTION] Here is my reflection." in strip_thinking(text)
        assert "<think>" not in strip_thinking(text)

    def test_removes_unclosed_think_block(self):
        text = "<think>Model hit token limit mid-thought and never closed"
        assert strip_thinking(text) == ""

    def test_preserves_content_without_think_tags(self):
        text = "[REFLECTION] Normal output.\n[QUESTION 1] A question?"
        assert strip_thinking(text) == text

    def test_collapses_excess_newlines(self):
        text = "<think>reasoning</think>\n\n\n\nContent here."
        result = strip_thinking(text)
        assert "\n\n\n" not in result
        assert "Content here." in result

    def test_empty_think_block(self):
        text = "<think>\n\n</think>[REFLECTION] After empty thinking."
        assert "[REFLECTION] After empty thinking." in strip_thinking(text)


# ── Strict parsing ───────────────────────────────────────────────────


class TestParseStructuredResponse:
    def test_well_formatted_response(self):
        raw = (
            "[REFLECTION] This is a reflection.\n"
            "[QUESTION 1] What is your experience?\n"
            "[QUESTION 2] Are you thinking A or B?"
        )
        result = parse_structured_response(raw)

        assert result.reflection == "This is a reflection."
        assert result.question_1 == "What is your experience?"
        assert result.question_2 == "Are you thinking A or B?"
        assert result.raw == raw

    def test_missing_question_2(self):
        raw = (
            "[REFLECTION] A reflection.\n"
            "[QUESTION 1] What happened?"
        )
        result = parse_structured_response(raw)

        assert result.reflection == "A reflection."
        assert result.question_1 == "What happened?"
        assert result.question_2 == ""

    def test_completely_unstructured(self):
        raw = "This is just free text with no brackets at all."
        result = parse_structured_response(raw)

        assert result.reflection == ""
        assert result.question_1 == ""
        assert result.question_2 == ""
        assert result.clean == raw

    def test_clean_joins_parsed_parts(self):
        raw = (
            "[REFLECTION] Ref.\n"
            "[QUESTION 1] Q1?\n"
            "[QUESTION 2] Q2?"
        )
        result = parse_structured_response(raw)
        assert result.clean == "Ref.\n\nQ1?\n\nQ2?"

    def test_clean_falls_back_to_raw_when_nothing_parsed(self):
        raw = "Just some text."
        result = parse_structured_response(raw)
        assert result.clean == raw

    def test_extra_whitespace_handling(self):
        raw = (
            "[REFLECTION]   Lots of space.  \n\n"
            "[QUESTION 1]   Spaced question?  \n"
            "[QUESTION 2]   Another?  "
        )
        result = parse_structured_response(raw)

        assert result.reflection == "Lots of space."
        assert result.question_1 == "Spaced question?"
        assert result.question_2 == "Another?"

    def test_multiline_reflection(self):
        raw = (
            "[REFLECTION] First line.\n"
            "Second line of reflection.\n"
            "[QUESTION 1] A question?"
        )
        result = parse_structured_response(raw)
        assert "First line." in result.reflection
        assert "Second line" in result.reflection


# ── Thinking model + parsing integration ─────────────────────────────


class TestThinkingModelParsing:
    def test_strips_think_then_parses(self):
        raw = (
            "<think>Let me analyze this idea carefully.\n"
            "The user wants to build something.\n</think>\n"
            "[REFLECTION] Interesting idea.\n"
            "[QUESTION 1] What triggered this?\n"
            "[QUESTION 2] A or B?"
        )
        result = parse_structured_response(raw)

        assert result.reflection == "Interesting idea."
        assert result.question_1 == "What triggered this?"
        assert result.question_2 == "A or B?"
        assert result.raw == raw  # Raw preserves original including <think>

    def test_strips_unclosed_think_falls_back(self):
        raw = "<think>Model ran out of tokens mid-thought"
        result = parse_structured_response(raw)
        assert result.clean == ""
        assert result.reflection == ""


# ── Fuzzy parsing ────────────────────────────────────────────────────


class TestFuzzyParsing:
    def test_case_insensitive_tags(self):
        raw = (
            "[Reflection] Mixed case reflection.\n"
            "[Question 1] First question?\n"
            "[Question 2] Second question?"
        )
        result = parse_structured_response(raw)
        assert result.reflection == "Mixed case reflection."
        assert result.question_1 == "First question?"
        assert result.question_2 == "Second question?"

    def test_bold_delimiters(self):
        raw = (
            "**Reflection** Bold reflection here.\n"
            "**Question 1** Bold Q1?\n"
            "**Question 2** Bold Q2?"
        )
        result = parse_structured_response(raw)
        assert result.reflection == "Bold reflection here."
        assert result.question_1 == "Bold Q1?"
        assert result.question_2 == "Bold Q2?"

    def test_angle_bracket_delimiters(self):
        raw = (
            "<Reflection> Angle bracket.\n"
            "<Question 1> Q1 angle?\n"
            "<Question 2> Q2 angle?"
        )
        result = parse_structured_response(raw)
        assert result.reflection == "Angle bracket."
        assert result.question_1 == "Q1 angle?"
        assert result.question_2 == "Q2 angle?"

    def test_colon_after_tag(self):
        raw = (
            "[Reflection]: Colon style.\n"
            "[Question 1]: Q1 colon?\n"
            "[Question 2]: Q2 colon?"
        )
        result = parse_structured_response(raw)
        assert result.reflection == "Colon style."
        assert result.question_1 == "Q1 colon?"
        assert result.question_2 == "Q2 colon?"

    def test_fuzzy_only_invoked_when_strict_fails(self):
        """If strict parsing works, fuzzy is not used (ensures priority)."""
        raw = (
            "[REFLECTION] Strict reflection.\n"
            "[QUESTION 1] Strict Q1?\n"
            "[QUESTION 2] Strict Q2?"
        )
        strict_r, strict_q1, strict_q2 = _parse_strict(raw)
        assert strict_r == "Strict reflection."
        assert strict_q1 == "Strict Q1?"
        assert strict_q2 == "Strict Q2?"


# ── Generic question detection ────────────────────────────────────────


class TestIsQuestionGeneric:
    def test_question_with_idea_keyword_is_not_generic(self):
        idea = "I want to build an app that helps people learn languages."
        question = "What's your experience learning languages?"
        assert is_question_generic(question, idea) is False

    def test_question_without_keywords_is_generic(self):
        idea = "I want to build an app that helps people learn languages."
        question = "What are your goals for the future?"
        assert is_question_generic(question, idea) is True

    def test_stop_words_do_not_count(self):
        idea = "I want to create a platform that helps people."
        # All qualifying words are stop words — can't assess, returns False
        question = "How do you plan to help people through this platform?"
        assert is_question_generic(question, idea) is False

    def test_all_stop_words_idea_with_one_keyword(self):
        idea = "I want to create a platform that helps people with budgeting."
        # "budgeting" is the only non-stop keyword
        question = "What are your long-term goals?"
        assert is_question_generic(question, idea) is True

    def test_short_words_in_idea_are_ignored(self):
        idea = "I want to do AI for art."
        # "want", "art" — "art" is only 3 chars, under the 4-char threshold
        question = "What kind of creative work interests you?"
        assert is_question_generic(question, idea) is True

    def test_case_insensitive_matching(self):
        idea = "I want to build a LANGUAGE learning tool."
        question = "What's your experience with language education?"
        assert is_question_generic(question, idea) is False

    def test_punctuation_stripped_from_idea_words(self):
        idea = "I want to help students, especially freshmen."
        question = "What challenges do freshmen face?"
        assert is_question_generic(question, idea) is False

    def test_empty_question_is_generic(self):
        assert is_question_generic("", "some idea") is True

    def test_empty_idea_is_generic(self):
        assert is_question_generic("some question", "") is True
