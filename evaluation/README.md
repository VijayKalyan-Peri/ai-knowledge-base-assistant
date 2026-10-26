# Retrieval evaluation set

`retrieval_questions.json` pairs each sample question with its expected source file and a distinctive phrase from the relevant passage. The examples currently cover only the synthetic return policy, not the book.

Run the BM25 baseline:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --mode bm25
```

After installing an Ollama embedding model, compare semantic and hybrid retrieval:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --mode semantic --embedding-model embeddinggemma
.\.venv\Scripts\python.exe -m src.evaluate --mode hybrid --embedding-model embeddinggemma
```

The evaluator reports Recall@k and mean reciprocal rank (MRR). Add representative book questions with expected PDF page sources and distinctive answer snippets before drawing conclusions about book retrieval.
