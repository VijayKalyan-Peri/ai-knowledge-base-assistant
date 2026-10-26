from __future__ import annotations

import argparse
from pathlib import Path

from src.answer import generate_answer
from src.rag import load_passages, search


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a Markdown/PDF knowledge base and optionally generate a grounded answer.")
    parser.add_argument("question", help="A question to search for in docs/")
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Ask the configured LLM to answer using the retrieved passages",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    passages = load_passages(project_root / "docs")
    if not passages:
        print("No passages found. Add Markdown documents to docs/.")
        return

    results = search(args.question, passages)
    if not results:
        print("No matching passages found. Try different words or add relevant docs.")
        return

    print("Retrieved passages (BM25 keyword search):\n")
    for rank, (score, passage) in enumerate(results, start=1):
        print(f"{rank}. {passage.source} | score={score:.2f}")
        print(f"   {passage.text}\n")

    if not args.answer:
        print("Retrieval only. Add --answer to generate a response with your configured LLM.")
        return

    try:
        answer = generate_answer(args.question, results, project_root)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"LLM error: {exc}\n")

    print(f"Answer:\n{answer}")
    print("\nRetrieved sources: " + ", ".join(dict.fromkeys(p.source for _, p in results)))


if __name__ == "__main__":
    main()
