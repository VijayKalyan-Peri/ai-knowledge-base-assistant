from __future__ import annotations

import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "the",
    "to", "was", "what", "when", "where", "which", "with",
}


@dataclass(frozen=True)
class Passage:
    source: str
    text: str


def words(text: str) -> list[str]:
    """Normalize text into lowercase tokens and remove common stop words."""
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOP_WORDS
    ]


def _append_chunks(
    passages: list[Passage], source: str, content: str, max_chars: int
) -> None:
    """Split text into sentence-based chunks while retaining its source label."""
    content = re.sub(r"\s+", " ", content).strip()
    if not content:
        return
    sentences = re.split(r"(?<=[.!?])\s+", content)
    chunk = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            if chunk:
                passages.append(Passage(source, chunk))
                chunk = ""
            passages.append(Passage(source, sentence[:max_chars]))
            sentence = sentence[max_chars:]
        if chunk and len(chunk) + len(sentence) + 1 > max_chars:
            passages.append(Passage(source, chunk))
            chunk = ""
        chunk = f"{chunk} {sentence}".strip()
    if chunk:
        passages.append(Passage(source, chunk))


def _load_pdf(path: Path, max_chars: int, progress_callback=None) -> list[Passage]:
    """Extract selectable text from each PDF page and preserve page labels."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support needs pypdf. Install it with: "
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        warnings.warn(f"Skipping unreadable PDF {path.name}: {exc}")
        return []

    if reader.is_encrypted:
        warnings.warn(f"Skipping password-protected PDF {path.name}.")
        return []

    passages: list[Passage] = []
    pages_with_text = 0
    total_pages = len(reader.pages)
    for page_number, page in enumerate(reader.pages, start=1):
        if progress_callback:
            progress_callback("extract", page_number, total_pages, path.name)
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            warnings.warn(f"Could not extract page {page_number} of {path.name}: {exc}")
            continue
        if not text.strip():
            continue
        pages_with_text += 1
        _append_chunks(passages, f"{path.name}, page {page_number}", text, max_chars)

    if pages_with_text == 0:
        warnings.warn(
            f"No selectable text found in {path.name}; it may be scanned and need OCR."
        )
    return passages


def load_document(path: Path, max_chars: int = 1200, progress_callback=None) -> list[Passage]:
    """Extract one supported document into source-labelled passages."""
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path, max_chars, progress_callback)

    if path.suffix.lower() != ".md":
        return []
    if progress_callback:
        progress_callback("extract", 1, 1, path.name)
    content = path.read_text(encoding="utf-8")
    content = re.sub(r"(?m)^\s{0,3}#{1,6}\s+.*$", "", content)
    passages: list[Passage] = []
    for paragraph in re.split(r"\n\s*\n", content):
        _append_chunks(passages, path.name, paragraph, max_chars)
    return passages


def load_passages(docs_dir: Path, max_chars: int = 1200) -> list[Passage]:
    """Load top-level Markdown and PDF files and split them into passages."""
    passages: list[Passage] = []
    supported_files = sorted(
        path for path in docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".pdf"}
    )
    for path in supported_files:
        passages.extend(load_document(path, max_chars))
    return passages

def search_bm25(
    query: str,
    passages: list[Passage],
    limit: int = 3,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[float, Passage]]:
    """Rank passages by BM25, a lexical relevance-ranking formula."""
    query_terms = words(query)
    if not query_terms or not passages:
        return []

    tokenized_passages = [words(passage.text) for passage in passages]
    document_frequencies: Counter[str] = Counter()
    for tokens in tokenized_passages:
        document_frequencies.update(set(tokens))

    average_length = sum(map(len, tokenized_passages)) / len(tokenized_passages)
    if average_length == 0:
        return []

    ranked: list[tuple[float, Passage]] = []
    total_passages = len(passages)
    for passage, tokens in zip(passages, tokenized_passages):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            doc_frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (total_passages - doc_frequency + 0.5) / (doc_frequency + 0.5)
            )
            length_norm = frequency + k1 * (
                1 - b + b * len(tokens) / average_length
            )
            score += inverse_document_frequency * frequency * (k1 + 1) / length_norm
        if score > 0:
            ranked.append((score, passage))
    return sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must be non-empty and have equal dimensions.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def search_semantic(
    passages: list[Passage],
    passage_vectors: list[list[float]],
    query_vector: list[float],
    limit: int = 3,
) -> list[tuple[float, Passage]]:
    """Rank passages by cosine similarity between their embeddings and the query."""
    if len(passages) != len(passage_vectors):
        raise ValueError("Each passage must have one matching embedding vector.")
    ranked = [
        (_cosine_similarity(query_vector, vector), passage)
        for passage, vector in zip(passages, passage_vectors)
    ]
    return sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]


def search_hybrid(
    query: str,
    passages: list[Passage],
    passage_vectors: list[list[float]],
    query_vector: list[float],
    limit: int = 3,
    rrf_k: int = 60,
) -> list[tuple[float, Passage]]:
    """Fuse BM25 and semantic rankings with Reciprocal Rank Fusion (RRF)."""
    if len(passages) != len(passage_vectors):
        raise ValueError("Each passage must have one matching embedding vector.")
    candidate_count = min(len(passages), max(20, limit * 10))
    bm25_results = search_bm25(query, passages, limit=candidate_count)
    semantic_results = search_semantic(
        passages, passage_vectors, query_vector, limit=candidate_count
    )

    passage_by_key = {(passage.source, passage.text): passage for passage in passages}
    fused_scores: defaultdict[tuple[str, str], float] = defaultdict(float)
    for ranking in (bm25_results, semantic_results):
        for rank, (_, passage) in enumerate(ranking, start=1):
            key = (passage.source, passage.text)
            fused_scores[key] += 1 / (rrf_k + rank)

    ranked_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)[:limit]
    return [(fused_scores[key], passage_by_key[key]) for key in ranked_keys]


def search(
    query: str,
    passages: list[Passage],
    limit: int = 3,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[float, Passage]]:
    """Backward-compatible alias for the current BM25 baseline."""
    return search_bm25(query, passages, limit=limit, k1=k1, b=b)
