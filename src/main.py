from __future__ import annotations

import argparse
from pathlib import Path

from src.rag import load_passages, search


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the project Markdown knowledge base.")
    parser.add_argument("question", help="A question to search for in docs/")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    passages = load_passages(project_root / "docs")
    if not passages:
        print("No passages found. Add Markdown documents to the docs folder.")
        return

    results = search(args.question, passages)
    if not results:
        print("No matching passages found. Try different words or add relevant docs.")
        return

    print("Retrieved passages (word-overlap baseline):\n")
    for rank, (score, passage) in enumerate(results, start=1):
        print(f"{rank}. {passage.source} | score={score:.2f}")
        print(f"   {passage.text}\n")
    print("This milestone retrieves evidence; it does not generate an LLM answer yet.")


if __name__ == "__main__":
    main()
