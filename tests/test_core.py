"""Live Assistant — Core logic tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFastFilter:
    def test_short_text_returns_none(self):
        from question_detector import fast_filter
        assert fast_filter("hi") is None

    def test_long_text_returns_none(self):
        from question_detector import fast_filter
        text = " ".join(["word"] * 100)
        assert fast_filter(text) is None

    def test_direct_question_strong(self):
        from question_detector import fast_filter
        result = fast_filter("can you tell me about your experience with Python?")
        assert result == "strong"

    def test_question_mark_weak(self):
        from question_detector import fast_filter
        result = fast_filter("you worked at a startup before right?")
        assert result == "weak"

    def test_statement_returns_none(self):
        from question_detector import fast_filter
        result = fast_filter("today we are going to discuss the architecture of distributed systems")
        assert result is None


class TestDetectQuestionType:
    def test_technical(self):
        from responder import detect_question_type
        assert detect_question_type("Write a function to reverse a linked list") == "technical"

    def test_behavioral(self):
        from responder import detect_question_type
        assert detect_question_type("Tell me about a time you led a team") == "behavioral"

    def test_algorithm_is_technical(self):
        from responder import detect_question_type
        assert detect_question_type("Implement a binary search algorithm") == "technical"

    def test_leadership_is_behavioral(self):
        from responder import detect_question_type
        assert detect_question_type("What's your greatest weakness?") == "behavioral"
