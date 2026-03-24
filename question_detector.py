"""Two-stage question detection: fast heuristic filter + LLM classifier.

Key design: avoid false positives from lecture content. Only trigger on
questions directly addressed to the user (interviewee).
"""

import subprocess
import time


# Stage 1: Patterns that strongly indicate a question TO the interviewee
# These must be at the START of a sentence or utterance
DIRECT_PATTERNS_START = [
    "tell me about", "tell us about",
    "walk me through", "walk us through",
    "can you tell", "could you tell", "would you tell",
    "can you describe", "could you describe",
    "can you explain", "could you explain",
    "what is your", "what's your", "what are your",
    "what do you think", "what would you do",
    "how do you", "how would you", "how did you",
    "why do you", "why did you", "why would you",
    "where do you see",
    "have you ever", "have you worked", "have you used",
    "do you have experience", "do you have any",
    "are you familiar", "are you comfortable",
    "describe a time", "describe a situation",
    "give me an example", "give us an example",
    "why should we hire", "why are you interested",
    "what attracted you", "what motivates you",
    "what experience do you",
    "do you have any questions",
    "write a", "implement a", "code a",
    "explain the difference", "explain how",
    "what is the time complexity", "what is the space complexity",
    "what is the difference between",
    "solve this", "solve the",
]

# Patterns that can appear anywhere in the text
DIRECT_PATTERNS_ANY = [
    "tell me about yourself",
    "walk me through your",
    "your greatest", "your biggest",
    "your strength", "your weakness",
    "why this company",
    "why this role",
    "where do you see yourself",
    "what salary",
    "when can you start",
]

# Definitely NOT a question to the interviewee
LECTURE_INDICATORS = [
    "what we", "what i", "what i'", "what the ",
    "how the ", "how we ", "how i ",
    "why the ", "why we ", "why i ",
    "that's what", "that's why", "that's how",
    "here's what", "here's why", "here's how",
    "so what we", "and what we",
    "let me", "let's", "let us",
    "i think", "i believe", "i would say",
    "we can", "we should", "we need",
    "there's a question", "the question is",
    "this is about", "this is how",
    "you know", "you might", "you see",
    "we're going", "we'll",
    "the problem", "the issue",
    "so basically", "so essentially",
]


def fast_filter(text: str) -> str | None:
    """Stage 1: Quick heuristic check.

    Returns "strong", "weak", or None.
    Much stricter than before — defaults to None (not a question).
    """
    lower = text.lower().strip()

    # Too short or too long — not a single question
    if len(lower) < 15 or len(lower.split()) > 80:
        return None

    # Check strong patterns FIRST (direct questions override lecture patterns)
    for pattern in DIRECT_PATTERNS_START:
        if lower.startswith(pattern):
            return "strong"

    # Then check lecture/statement indicators
    for pattern in LECTURE_INDICATORS:
        if lower.startswith(pattern):
            return None

    # Check patterns that can be anywhere
    for pattern in DIRECT_PATTERNS_ANY:
        if pattern in lower:
            return "strong"

    # Question mark at the end + short enough to be a real question
    if lower.endswith("?") and len(lower.split()) < 30:
        return "weak"

    return None


def llm_classify(text: str, ollama_model: str) -> bool:
    """Stage 2: Use Ollama to confirm if this is a direct question to the interviewee."""
    # Truncate to avoid huge prompts
    truncated = text[:300]
    prompt = (
        f'Is this a direct question being asked TO a person in an interview? '
        f'(NOT a lecture statement, NOT rhetorical, NOT the speaker talking to themselves) '
        f'Reply ONLY "YES" or "NO".\n\n"{truncated}"'
    )
    try:
        result = subprocess.run(
            ["ollama", "run", ollama_model, prompt],
            capture_output=True, text=True, timeout=10,
        )
        answer = result.stdout.strip().upper()
        return answer.startswith("YES")
    except Exception:
        return False


class QuestionDetector:
    """Detects real interview questions from transcript stream."""

    def __init__(self, ollama_model: str = "gemma3:4b"):
        self.ollama_model = ollama_model
        self.sentence_buffer = ""
        self.last_speech_time = 0
        self.silence_gap = 2.0  # seconds of silence = sentence boundary

    def feed(self, text: str, timestamp: float | None = None) -> str | None:
        """Feed a transcribed chunk. Returns question text if detected, else None."""
        now = timestamp or time.time()

        # On silence gap, evaluate previous buffer and start fresh
        if self.sentence_buffer and (now - self.last_speech_time) > self.silence_gap:
            result = self._evaluate(self.sentence_buffer)
            self.sentence_buffer = text
            self.last_speech_time = now
            return result

        # Accumulate
        if self.sentence_buffer:
            self.sentence_buffer += " " + text
        else:
            self.sentence_buffer = text
        self.last_speech_time = now

        # Don't let buffer grow too large — evaluate in chunks
        word_count = len(self.sentence_buffer.split())
        if word_count > 40:
            # Only evaluate the LAST ~30 words (most recent utterance)
            words = self.sentence_buffer.split()
            recent = " ".join(words[-30:])
            result = self._evaluate(recent)
            self.sentence_buffer = ""
            return result

        return None

    def flush(self) -> str | None:
        """Force evaluate the buffer (called on silence)."""
        if self.sentence_buffer and len(self.sentence_buffer.split()) > 5:
            # Only look at last ~30 words
            words = self.sentence_buffer.split()
            text = " ".join(words[-30:]) if len(words) > 30 else self.sentence_buffer
            result = self._evaluate(text)
            self.sentence_buffer = ""
            return result
        self.sentence_buffer = ""
        return None

    def _evaluate(self, text: str) -> str | None:
        """Run 2-stage detection."""
        signal = fast_filter(text)

        if signal is None:
            return None

        if signal == "strong":
            # Even strong signals get LLM verification to reduce false positives
            if llm_classify(text, self.ollama_model):
                return text
            return None

        # "weak": always verify with LLM
        if llm_classify(text, self.ollama_model):
            return text

        return None
