# AI Knowledge Base Assistant

A beginner-friendly retrieval-augmented generation (RAG) learning project. Milestone 1 searches a small Markdown knowledge base and shows the source passages for each question. It uses Python's standard library, so this first milestone needs no API key or model download.

## Requirements

- Python 3.10 or newer
- Git

## Run on Windows PowerShell

From this folder:

```powershell
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m src.main "How long are returns accepted?"
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe -m src.main "How long are returns accepted?"` instead. This avoids changing the machine's execution policy.

Add public or synthetic Markdown files under `docs\`, then ask questions about them. The demo uses word overlap retrieval as a transparent baseline; it is not a production search system and does not call an LLM yet.

## Next milestones

1. This local retrieval baseline with source snippets.
2. Add an LLM adapter and compare a hosted API with a local model using the same retrieved context.
3. Replace word overlap with embedding-based retrieval and add evaluation examples.

Never add confidential company or customer data, API keys, `.env` files, or model credentials to this repository.
