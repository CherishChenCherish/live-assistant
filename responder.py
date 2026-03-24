"""Generate spoken English responses via Ollama with context-aware prompting.

Supports two response modes:
- BEHAVIORAL: conversational spoken answers (STAR format)
- TECHNICAL: code + explanation (algorithms, system design, SQL, etc.)
"""

import re
import subprocess

# --- Question Type Detection ---
TECHNICAL_KEYWORDS = [
    # Coding
    "implement", "write code", "write a function", "code this",
    "algorithm", "data structure", "time complexity", "space complexity",
    "big o", "optimize this", "binary search", "sort", "hash map", "linked list",
    "tree", "graph", "dynamic programming", "recursion", "bfs", "dfs",
    "two pointer", "sliding window", "stack", "queue", "heap",
    # Languages & tools
    "python", "java", "javascript", "typescript", "sql", "react",
    "api", "restful", "rest api", "http", "database", "query",
    # System design
    "system design", "design a", "architecture", "scalab", "load balanc",
    "caching", "microservice", "distributed", "database schema",
    "how would you build", "how would you design",
    # ML/AI specific
    "neural network", "model", "training", "gradient", "loss function",
    "overfitting", "regularization", "cross-validation", "feature engineer",
    "precision", "recall", "f1", "auc", "roc",
    "transformer", "attention", "embedding", "fine-tun",
    "pandas", "numpy", "pytorch", "tensorflow", "scikit",
    # Concepts
    "explain the difference between", "what is the difference",
    "how does.*work", "what happens when",
    "runtime", "memory", "thread", "process", "deadlock", "race condition",
]


def detect_question_type(text: str) -> str:
    """Classify question as 'technical' or 'behavioral'."""
    lower = text.lower()
    for kw in TECHNICAL_KEYWORDS:
        if kw in lower:
            return "technical"
    # Regex patterns
    if re.search(r"write\s+(a\s+)?(function|method|class|program|script|code)", lower):
        return "technical"
    if re.search(r"(implement|solve|compute|calculate)\s+", lower):
        return "technical"
    return "behavioral"


# --- Prompt Templates ---

SYSTEM_PROMPT_BASE = """You are a real-time interview coach helping Cherish Chen ace her interview.

## User Profile
Name: Cherish Chen
- Yale School of Management (current MBA)
- AI/ML Engineer: Python, deep learning, NLP, computer vision
- Founded an AI startup
- Previously at Gate.io (cryptocurrency exchange) — data analysis, quantitative trading
- University of Washington — Informatics + Economics double major
- Bilingual Chinese/English
- Strong in: Python, SQL, pandas, scikit-learn, PyTorch, React, data visualization

{context_section}"""

BEHAVIORAL_RULES = """
## Response Rules (Behavioral/General)
1. Output ONLY the spoken response — no labels, no markdown headers, no bullet points
2. Natural spoken English — contractions OK, minimal filler
3. Length: 3-6 sentences. Concise but substantive
4. Start with a direct answer, then support with specifics
5. For "tell me about a time" questions, use STAR: Situation → Task → Action → Result
6. Sound confident and genuine, not rehearsed
7. Reference specific experiences from profile/context when relevant
8. Do NOT invent experiences"""

TECHNICAL_RULES = """
## Response Rules (Technical)
You MUST use EXACTLY these three headers. Do NOT skip any.

VERBAL:
I would approach this using a sliding window technique with a hash map to track characters.

CODE:
def solution():
    pass

EXPLAIN:
This runs in O(n) time and O(min(n,m)) space where m is the charset size.

IMPORTANT FORMATTING:
- Start with "VERBAL:" then 2-3 spoken sentences (NO code here)
- Then "CODE:" then the actual code (NO ``` backticks, just raw code)
- Then "EXPLAIN:" then 1-2 sentences on complexity/trade-offs
- Default to Python unless specified otherwise
- Code must be clean and correct
- For system design: describe components in VERBAL, show key code in CODE
- For SQL: put the query in CODE"""

RESPONSE_PROMPT = """Recent conversation:
{conversation}

Question detected:
"{question}"

Question type: {question_type}

{previous_section}

Generate your response:"""


def build_system_prompt(context_materials: str = "", question_type: str = "behavioral") -> str:
    if context_materials:
        context_section = f"## Background Materials\n{context_materials}"
    else:
        context_section = "(No additional context materials loaded)"

    base = SYSTEM_PROMPT_BASE.format(context_section=context_section)
    rules = TECHNICAL_RULES if question_type == "technical" else BEHAVIORAL_RULES
    return base + rules


def build_response_prompt(
    question: str,
    conversation_lines: list[dict],
    previous_responses: list[dict] | None = None,
    question_type: str = "behavioral",
) -> str:
    conv = "\n".join(
        f"[{line['time']}] {line['text']}" for line in conversation_lines
    )

    prev_section = ""
    if previous_responses:
        prev_parts = [f"Q: {r['question']}\nA: {r['response']}" for r in previous_responses[-3:]]
        prev_section = "Previous responses (do NOT repeat):\n" + "\n---\n".join(prev_parts)

    return RESPONSE_PROMPT.format(
        conversation=conv or "(no prior conversation)",
        question=question,
        question_type=question_type.upper(),
        previous_section=prev_section,
    )


def _call_ollama(ollama_model: str, prompt: str, timeout: int = 60) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", ollama_model, prompt],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[Response timed out — press Regenerate to retry]"
    except Exception as e:
        return f"[Error: {e}]"


def generate_response(
    ollama_model: str,
    question: str,
    conversation_lines: list[dict],
    context_materials: str = "",
    previous_responses: list[dict] | None = None,
) -> dict:
    """Generate a response. Returns dict with type, verbal, code, explain fields.

    For behavioral: {"type": "behavioral", "verbal": "...", "code": None, "explain": None}
    For technical:  {"type": "technical", "verbal": "...", "code": "...", "explain": "..."}
    """
    q_type = detect_question_type(question)

    system = build_system_prompt(context_materials, q_type)
    user = build_response_prompt(question, conversation_lines, previous_responses, q_type)
    full_prompt = f"{system}\n\n---\n\n{user}"

    raw = _call_ollama(ollama_model, full_prompt)

    if q_type == "technical":
        return _parse_technical_response(raw)
    else:
        verbal = _clean_behavioral(raw)
        if len(verbal) > 400:
            sentences = re.split(r'(?<=[.!?])\s+', verbal)
            verbal = " ".join(sentences[:5])
        return {
            "type": "behavioral",
            "verbal": verbal,
            "code": None,
            "explain": None,
        }


def regenerate_response(
    ollama_model: str,
    question: str,
    conversation_lines: list[dict],
    context_materials: str = "",
    previous_responses: list[dict] | None = None,
    previous_attempt: str = "",
) -> dict:
    q_type = detect_question_type(question)

    system = build_system_prompt(context_materials, q_type)
    user = build_response_prompt(question, conversation_lines, previous_responses, q_type)
    if previous_attempt:
        user += f"\n\nPrevious attempt (try a DIFFERENT approach):\n{previous_attempt}"
    full_prompt = f"{system}\n\n---\n\n{user}"

    raw = _call_ollama(ollama_model, full_prompt)

    if q_type == "technical":
        return _parse_technical_response(raw)
    else:
        return {
            "type": "behavioral",
            "verbal": _clean_behavioral(raw),
            "code": None,
            "explain": None,
        }


def _parse_technical_response(raw: str) -> dict:
    """Parse a technical response into verbal / code / explain sections."""
    verbal = ""
    code = ""
    explain = ""

    # Strategy 1: Structured VERBAL/CODE/EXPLAIN headers
    sections = re.split(r'\n?(?:VERBAL|CODE|EXPLAIN)\s*:\s*\n?', raw, flags=re.IGNORECASE)

    if len(sections) >= 4:
        verbal = _clean_behavioral(sections[1].strip())
        code = sections[2].strip()
        explain = _clean_behavioral(sections[3].strip())
    elif len(sections) >= 3:
        verbal = _clean_behavioral(sections[1].strip())
        code = sections[2].strip()
    else:
        # Strategy 2: Extract ```code blocks```
        code_blocks = re.findall(r'```[\w]*\n(.*?)```', raw, re.DOTALL)
        if code_blocks:
            code = "\n\n".join(block.strip() for block in code_blocks)
            outside = re.sub(r'```[\w]*\n.*?```', '|||CB|||', raw, flags=re.DOTALL)
            text_parts = [p.strip() for p in outside.split('|||CB|||') if p.strip()]
            all_text = _clean_behavioral("\n".join(text_parts))
            sentences = re.split(r'(?<=[.!?])\s+', all_text)
            if len(sentences) > 3:
                verbal = " ".join(sentences[:3])
                explain = " ".join(sentences[3:])
            else:
                verbal = all_text
        else:
            # Strategy 3: Look for code-like lines (def/class/import/SELECT/for/if with indentation)
            lines = raw.split("\n")
            code_lines, text_lines = [], []
            in_code = False
            for line in lines:
                is_code_line = (
                    line.startswith("    ") or line.startswith("\t") or
                    re.match(r'^(def |class |import |from |SELECT |INSERT |CREATE |for |if |while |return )', line.strip())
                )
                if is_code_line:
                    in_code = True
                    code_lines.append(line)
                elif in_code and line.strip() == "":
                    code_lines.append(line)
                else:
                    in_code = False
                    text_lines.append(line)
            if code_lines:
                code = "\n".join(code_lines).strip()
                verbal = _clean_behavioral("\n".join(text_lines))
            else:
                verbal = _clean_behavioral(raw)

    # Clean code: remove ``` fences if still present
    code = re.sub(r'^```[\w]*\n?', '', code)
    code = re.sub(r'\n?```$', '', code)

    # Ensure verbal doesn't contain code
    if verbal and ('def ' in verbal or 'import ' in verbal or '```' in verbal):
        # Verbal got contaminated with code — extract just the natural text
        clean_lines = [l for l in verbal.split('\n')
                       if not l.strip().startswith(('def ', 'class ', 'import ', 'from ', '```', '#!', '    '))]
        verbal = _clean_behavioral("\n".join(clean_lines))

    # Cap verbal length
    if verbal and len(verbal) > 400:
        sentences = re.split(r'(?<=[.!?])\s+', verbal)
        verbal = " ".join(sentences[:3])

    # Fallback verbal
    if not verbal or len(verbal) < 10:
        verbal = "Let me walk you through my approach for this problem."

    # Clean code fences
    code = re.sub(r'^```[\w]*\n?', '', code)
    code = re.sub(r'\n?```$', '', code)

    return {
        "type": "technical",
        "verbal": verbal or "(approach explanation pending)",
        "code": code or None,
        "explain": explain or None,
    }


def _clean_behavioral(text: str) -> str:
    """Remove LLM artifacts from behavioral response."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith("**") and s.endswith("**"):
            continue
        if s.lower().startswith(("here's", "here is", "response:", "answer:", "sure,")):
            for sep in [":", ","]:
                if sep in s:
                    after = s.split(sep, 1)[1].strip()
                    if after:
                        cleaned.append(after)
                    break
            continue
        if s.startswith("*") and not s.startswith("* "):
            s = s.strip("*").strip()
        cleaned.append(s)
    return " ".join(cleaned)
