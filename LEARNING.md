# Grounded answer generation

The `--answer` option sends the question and retrieved passages to an OpenAI-compatible Chat Completions endpoint. The configured model is asked to answer only from those passages and to cite their filenames. The application also prints the actual retrieved sources separately.

## Configure one provider

Copy `.env.example` to `.env` and edit the settings. `.env` is ignored by Git. Never paste an API key into source code or commit it. The same program can be pointed to a hosted API or a local Ollama server by changing `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY`.

Then run:

```powershell
.\.venv\Scripts\python.exe -m src.main "When can customers request a return?" --answer
```

The model and retrieved context are sent to the configured endpoint. Use synthetic/public data with a hosted provider. For a local Ollama server, its documentation says the local OpenAI-compatible endpoint is `http://localhost:11434/v1/`; the API key field is required by compatible clients but ignored by the local server, so `ollama` is a placeholder value. Install and download a model separately before using that option.

## Concepts, use, and bottlenecks

- **Retrieval vs generation:** BM25 selects evidence; the LLM turns that evidence into a readable response. The model should not be treated as the source of truth.
- **Grounding prompt:** The system instruction asks the model to use only the supplied passages and say when they do not answer the question. This reduces unsupported answers but cannot guarantee they never happen.
- **Source handling:** The model is asked to cite filenames, while the application prints the sources actually passed to it. Later we can track individual passage IDs for more precise citations.
- **Latency and cost:** A hosted request adds network time and may be billed by tokens. A local request avoids sending prompts over the network but uses laptop memory/compute and can respond more slowly.
- **Context size:** Sending more passages can help coverage but increases latency and token use. Top-k, chunk size, and model context limits are trade-offs.
- **Prompt injection:** Retrieved files are untrusted input. The prompt tells the model not to follow instructions inside documents, but this is not a security boundary. Production systems need stronger isolation and adversarial evaluation.
- **Provider compatibility:** This project uses the shared `/chat/completions` shape and basic `model`/`messages` fields. Providers can differ in supported parameters and response formats, so the adapter may need provider-specific handling.

## Interview questions

1. **What makes this RAG?** It retrieves relevant passages from an external knowledge base and supplies them as context to a language model that generates the answer.
2. **How do you reduce hallucinations?** Improve retrieval, constrain the prompt to evidence, return sources, abstain when evidence is missing, and evaluate groundedness. A prompt alone cannot guarantee correctness.
3. **What is the hosted vs local trade-off?** Hosted models simplify compute and may offer stronger performance but send data to a provider and can incur cost/latency. Local models keep inference on-device but are bounded by laptop resources and may be slower or less capable.
4. **How would you compare the two fairly?** Use identical questions, retrieved passages, and answer instructions; measure answer correctness/groundedness, latency, and cost, and record the model and settings.
5. **What are key production risks?** Secret leakage, sensitive data leaving the machine, prompt injection, unsupported claims, access-control mistakes, and unbounded latency/cost.

Never use confidential company or customer data in this learning project.

## PDF document ingestion

The loader uses pypdf to extract selectable text page by page and keeps the PDF filename and page number on each passage. BM25 then treats the extracted text like Markdown text. This preserves a basic source locator, but PDF text extraction may disrupt columns, tables, reading order, or hyphenated words. pypdf does not OCR scanned pages; those need an OCR step before ingestion. Password-protected PDFs are skipped. The app scans only files directly inside docs/, not nested folders. PDFs are ignored by Git by default so personal documents are less likely to be pushed accidentally.

## Chat user interface

The Streamlit page is a local browser client backed by Python on the same laptop. It discovers local Ollama models with the Ollama tags endpoint, lets the learner ask questions, runs BM25 retrieval, and sends the retrieved passages to the selected local model. It caches extracted passages because reparsing a large book for every UI rerun would waste time. The Refresh button clears that cache after documents change. st.session_state keeps the visible chat history across Streamlit reruns, but this milestone deliberately sends each question independently; prior turns are not model context. This is useful to distinguish chat history in the UI from conversational memory in the LLM.

## Semantic and hybrid retrieval

An embedding model maps each passage and query to a numeric vector. Cosine similarity ranks vectors that point in similar directions, which can match related wording even without exact shared terms. This project requests local vectors from Ollama's /api/embed endpoint, so document text stays on the local server. BM25 remains useful for exact names, numbers, and phrases. Hybrid mode combines the two ranked lists with Reciprocal Rank Fusion (RRF), which adds rank-based scores rather than combining BM25 and cosine values with incompatible scales.

### Persistent document index

The app stores passages and float32 vectors in `.cache/knowledge_index.sqlite3`, a local SQLite database excluded from Git. On startup, it compares the documents in `docs/` with the saved document records. A new or changed document is extracted, split into 1,200-character passages, embedded once in batches, then saved. A deleted document and its passages are removed. A content hash is checked when file size or modification time changes, so touching a file without changing its contents does not trigger re-embedding. The **Sync documents** button forces a reconciliation while the app is open. The app keeps a read-only index snapshot in memory for quick repeated searches.

The first index build is still expensive and shows progress; later questions embed only the question. Changing the embedding model invalidates and rebuilds the index because vectors from different models are not comparable. The index is loaded into memory at search time; for very large collections, memory use and the linear scan over all vectors become bottlenecks, at which point a vector database or approximate-nearest-neighbor index may help. The current evaluation set covers only the fictional return policy, so it cannot establish search quality on the book. Add book questions with correct PDF page/source snippets and compare Recall@k and MRR.

**Interview question: How do you avoid embedding every document on every query?**
Persist each document passage and vector in a local index. Track each source file with a content hash, re-embed only new or changed files, remove records for deleted files, and embed only the query at request time. The trade-offs are storage, index invalidation when the embedding model changes, and search cost as the collection grows.
