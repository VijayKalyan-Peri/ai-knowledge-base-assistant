from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.embeddings import embed_texts
from src.rag import (
    Passage,
    load_passages,
    search_bm25,
    search_hybrid,
    search_semantic,
)


def matches_expected(passage: Passage, example: dict[str, str]) -> bool:
    """Match the expected source and a distinctive snippet from its passage."""
    return (
        passage.source == example["expected_source"]
        and example["expected_contains"].casefold() in passage.text.casefold()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure retrieval against labeled example questions.")
    parser.add_argument("--k", type=int, default=3, help="Number of top results to consider (default: 3)")
    parser.add_argument("--mode", choices=("bm25", "semantic", "hybrid"), default="bm25")
    parser.add_argument("--server-url", default="http://localhost:11434")
    parser.add_argument("--embedding-model", default="embeddinggemma")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    project_root = Path(__file__).resolve().parents[1]
    examples_path = project_root / "evaluation" / "retrieval_questions.json"
    examples = json.loads(examples_path.read_text(encoding="utf-8"))["questions"]
    passages = load_passages(project_root / "docs")
    if not examples:
        parser.error("Evaluation set contains no questions")

    passage_vectors: list[list[float]] = []
    query_vectors: list[list[float]] = []
    if args.mode != "bm25":
        passage_vectors = embed_texts(
            [passage.text for passage in passages], args.server_url, args.embedding_model
        )
        query_vectors = embed_texts(
            [example["question"] for example in examples],
            args.server_url,
            args.embedding_model,
        )

    reciprocal_ranks: list[float] = []
    hits = 0
    for index, example in enumerate(examples):
        if args.mode == "bm25":
            results = search_bm25(example["question"], passages, limit=args.k)
        elif args.mode == "semantic":
            results = search_semantic(
                passages, passage_vectors, query_vectors[index], limit=args.k
            )
        else:
            results = search_hybrid(
                example["question"],
                passages,
                passage_vectors,
                query_vectors[index],
                limit=args.k,
            )

        matching_rank = next(
            (rank for rank, (_, passage) in enumerate(results, start=1)
             if matches_expected(passage, example)),
            None,
        )
        if matching_rank is None:
            reciprocal_ranks.append(0.0)
            outcome = f"MISS in top {args.k}"
        else:
            hits += 1
            reciprocal_ranks.append(1.0 / matching_rank)
            outcome = f"hit at rank {matching_rank}"
        print(f"{outcome}: {example['question']}")

    recall_at_k = hits / len(examples)
    mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"\nMode: {args.mode}")
    print(f"Questions: {len(examples)}")
    print(f"Recall@{args.k}: {recall_at_k:.2f} ({hits}/{len(examples)})")
    print(f"MRR:       {mean_reciprocal_rank:.2f}")
    print("Scores describe this small synthetic sample, not general or book-specific quality.")


if __name__ == "__main__":
    main()
