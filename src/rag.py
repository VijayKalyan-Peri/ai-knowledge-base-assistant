from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class Passage:
    source: str
    text: str


def words(text: str) -> set[str]:
    """Return lowercase word tokens, ignoring punctuation."""
    return {token.lower() for token in TOKEN_RE.findall(text)}


def load_passages(docs_dir: Path, max_chars: int = 700) -> list[Passage]:
    """Load Markdown files and split them into paragraph-sized passages."""
    passages: list[Passage] = []
    for path in sorted(docs_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        for paragraph in re.split(r"\n\s*\n", content):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            # Keep chunks bounded; split long paragraphs on sentence boundaries.
            pieces = re.split(r"(?<=[.!?])\s+", paragraph)
            chunk = ""
            for piece in pieces:
                if chunk and len(chunk) + len(piece) + 1 > max_chars:
                    passages.append(Passage(path.name, chunk))
                    chunk = ""
                chunk = f"{chunk} {piece}".strip()
            if chunk:
                passages.append(Passage(path.name, chunk))
    return passages


def search(query: str, passages: list[Passage], limit: int = 3) -> list[tuple[float, Passage]]:
    """Rank passages by fraction of query words that they contain."""
    query_words = words(query)
    if not query_words:
        return []
    ranked = []
    for passage in passages:
        passage_words = words(passage.text)
        score = len(query_words & passage_words) / len(query_words)
        if score > 0:
            ranked.append((score, passage))
    return sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]
