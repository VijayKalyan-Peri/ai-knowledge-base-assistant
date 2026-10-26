# AI Knowledge Base Assistant

A local, beginner-friendly Retrieval-Augmented Generation (RAG) app. Add PDF or Markdown documents, search them with keywords (BM25), meaning-based embeddings, or both, and ask a local Ollama model to answer from the retrieved passages.

The app is designed for learning and personal documents. It is not a production document-security or access-control system.

## What you need

- Python 3.10 or newer
- Git
- [Ollama](https://ollama.com/download) installed and running locally
- Enough disk space and memory for the Ollama models you choose. Smaller models are a practical starting point on a laptop.

## 1. Download the project

Open Command Prompt, PowerShell, or a terminal and run:

```shell
git clone https://github.com/VijayKalyan-Peri/ai-knowledge-base-assistant.git
cd ai-knowledge-base-assistant
```

## 2. Create the Python environment

### Windows

Run these commands from the project folder:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `python` is not recognized, install Python from [python.org](https://www.python.org/downloads/) and reopen the terminal. The project requires Python 3.10 or newer.

### macOS or Linux

Run these commands from the project folder:

```shell
python3 --version
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Install Ollama models

The app uses one model to write answers and a separate embedding model to search by meaning. In a terminal, download both with Ollama:

```shell
ollama pull qwen3:4b
ollama pull embeddinggemma
ollama list
```

`qwen3:4b` is an example answer model; you can use another chat model installed in Ollama. `embeddinggemma` is the example embedding model. The model names must appear in `ollama list` before the app can select them. See the [Qwen3 model page](https://ollama.com/library/qwen3) and [EmbeddingGemma model page](https://ollama.com/library/embeddinggemma).

**Windows note:** If `ollama` is not recognized, reopen the terminal after installing Ollama. If it is still not recognized, run the executable directly in Command Prompt:

```cmd
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" pull qwen3:4b
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" pull embeddinggemma
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" list
```

Ollama should be running and reachable at `http://localhost:11434`. The app checks this address by default.

## 4. Add your documents

Place `.pdf` and `.md` files directly in the project's `docs/` folder. The app scans that folder (not subfolders). You can start with the included synthetic `return-policy.md` example.

- Text PDFs retain the PDF filename and page number in retrieved sources.
- Scanned or image-only PDFs need OCR before this app can read them.
- Password-protected or unreadable PDFs are skipped.
- PDFs in `docs/` are ignored by Git so personal books and documents are not pushed accidentally. Markdown files are not automatically ignored, so keep private Markdown documents out of commits.

## 5. Start the app

Run the command for your operating system from the project folder.

### Windows

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\app.py --server.address 127.0.0.1 --server.port 8501
```

### macOS or Linux

If you activated `.venv` in step 2:

```shell
python -m streamlit run src/app.py --server.address 127.0.0.1 --server.port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501) in your browser. Keep the terminal and Ollama running while you use the app. To stop Streamlit, press **Ctrl+C** in its terminal.

## 6. Ask questions

In the app's sidebar:

1. Select an **Answer model** (for example, `qwen3:4b`).
2. Select an **Embedding model** (for example, `embeddinggemma`).
3. Choose a retrieval method:
   - **BM25** matches words and is useful for exact terms, names, and numbers.
   - **Semantic** uses embeddings to find passages with related meaning.
   - **Hybrid** combines both rankings and is the default recommendation.
4. Type a question. The answer is generated from the retrieved passages, which are available under **Retrieved passages**.

BM25 can run without an embedding model. An answer model is needed to generate a written answer; without one, the app can still show retrieved passages.

## How document indexing works

When the app starts, it checks the documents against a local SQLite index at `.cache/knowledge_index.sqlite3`. Each passage (up to 1,200 characters) gets one embedding. The first index build can take a while for a large book, and the app shows progress.

The index is saved locally and excluded from Git. Unchanged documents reuse their saved vectors. New or changed documents are indexed; deleted documents are removed. If you add, edit, or delete files while the app is open, click **Sync documents** in the sidebar (or refresh the app) to update the index. Changing the embedding model requires rebuilding the vectors.

Each question is embedded when you search. The app compares that query vector with the saved document vectors; it does not recreate the document embeddings for every question. Generating the answer still takes time because the local answer model runs on your computer.

## Troubleshooting

- **Ollama is not connected:** Open the Ollama app and make sure it is running. The app uses `http://localhost:11434` by default.
- **No model appears in the app:** Run `ollama list` and confirm you have pulled both an answer model and an embedding model. On Windows, use the full executable path above if needed.
- **Indexing seems slow:** The first index build processes every passage. Watch the progress indicator; later runs reuse the saved SQLite index.
- **A document change is not reflected:** Click **Sync documents** after the file has finished copying into `docs/`.
- **A PDF produces no passages:** It may be scanned, encrypted, or contain text in a format the PDF parser cannot extract. OCR scanned documents first.
- **The app page does not open:** Check that Streamlit is still running and visit `http://127.0.0.1:8501`.

## Optional command-line use

The command-line tool performs BM25 keyword retrieval without opening the web app. From the project folder:

### Windows

```powershell
.\.venv\Scripts\python.exe -m src.main "When can customers request a return?"
```

### macOS or Linux

```shell
python -m src.main "When can customers request a return?"
```

To also generate an answer, copy `.env.example` to `.env` and fill in `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY`. For local Ollama, use `http://localhost:11434/v1`, the name of an installed chat model such as `qwen3:4b`, and `ollama` as the placeholder API key. Then add `--answer` to the command. Keep `.env` private; it is ignored by Git.
## Evaluate retrieval

The included evaluation questions are for the synthetic return policy only; they do not measure quality on your book. Run these commands from the project folder.

### Windows

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --mode bm25
.\.venv\Scripts\python.exe -m src.evaluate --mode semantic --embedding-model embeddinggemma
.\.venv\Scripts\python.exe -m src.evaluate --mode hybrid --embedding-model embeddinggemma
```

### macOS or Linux

```shell
python -m src.evaluate --mode bm25
python -m src.evaluate --mode semantic --embedding-model embeddinggemma
python -m src.evaluate --mode hybrid --embedding-model embeddinggemma
```

The evaluator reports Recall@k and mean reciprocal rank (MRR) for the small labeled sample. Add questions with expected sources from your own public or synthetic documents before drawing conclusions about their retrieval quality. See [evaluation/README.md](evaluation/README.md) and [LEARNING.md](LEARNING.md) for the evaluation details and RAG concepts.

## Privacy and Git

The Streamlit app sends document passages and questions to the Ollama server address shown in the sidebar; by default this is the local server at `localhost`. The Streamlit interface is configured for local Ollama. The optional command-line tool can also use an OpenAI-compatible hosted endpoint configured in `.env`; that provider receives the question and retrieved passages and may charge for usage.

Do not add confidential company or customer data, API keys, or private documents to this learning project. `.env` and PDFs in `docs/` are ignored by Git; `.env.example` contains placeholders only. Review `git status` before committing, especially if you add Markdown documents.
