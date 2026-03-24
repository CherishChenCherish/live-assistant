"""Load background context materials from files (.txt, .md, .pdf, .docx)."""

from pathlib import Path


def load_file(filepath: str) -> str:
    """Read a file and return its text content."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Context file not found: {filepath}")

    suffix = p.suffix.lower()

    if suffix in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        return _load_pdf(p)

    if suffix == ".docx":
        return _load_docx(p)

    # Fallback: try reading as text
    return p.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    import fitz  # pymupdf
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def _load_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_context(file_paths: list[str] | None, text_snippets: list[str] | None) -> str:
    """Load all context materials into a single string.

    Args:
        file_paths: List of file paths to load
        text_snippets: List of raw text snippets (from --context-text)

    Returns:
        Combined context string, or empty string if nothing provided
    """
    parts = []

    if file_paths:
        for fp in file_paths:
            try:
                content = load_file(fp)
                name = Path(fp).name
                parts.append(f"=== {name} ===\n{content.strip()}")
            except Exception as e:
                parts.append(f"=== {Path(fp).name} === [Error: {e}]")

    if text_snippets:
        for snippet in text_snippets:
            parts.append(f"=== Additional Info ===\n{snippet.strip()}")

    return "\n\n".join(parts)


def context_summary(file_paths: list[str] | None, text_snippets: list[str] | None) -> str:
    """Return a short summary of loaded context for UI display."""
    items = []
    if file_paths:
        items.extend(Path(fp).name for fp in file_paths)
    if text_snippets:
        items.append(f"{len(text_snippets)} text snippet(s)")
    return ", ".join(items) if items else "None"
