"""Tests for response parsing and anti-generic question checking."""

from ideanator.parser import is_question_generic, parse_structured_response


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
        # "platform", "people", "helps", "create" are all stop words
        question = "How do you plan to help people through this platform?"
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
