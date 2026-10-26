from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import streamlit as st

from src.answer import generate_answer
from src.embeddings import embed_texts
from src.index_store import index_generation, load_index_snapshot, sync_documents
from src.rag import Passage, load_passages, search_bm25, search_hybrid, search_semantic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
DATABASE_PATH = PROJECT_ROOT / ".cache" / "knowledge_index.sqlite3"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
CHUNK_SIZE = 1200


def document_files() -> list[Path]:
    """List supported files directly inside docs/."""
    if not DOCS_DIR.exists():
        return []
    return sorted(
        path for path in DOCS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".md"}
    )


def document_signature(files: list[Path]) -> tuple[tuple[str, int, int], ...]:
    """Fast session-level change check; the indexer hashes files before updating them."""
    return tuple(
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in files
    )


@st.cache_data(show_spinner=False)
def get_passages(docs_path: str, signature: tuple[tuple[str, int, int], ...]) -> list[Passage]:
    """Cache document extraction for keyword-only use when no embedding model exists."""
    del signature  # Included in the cache key so changed files refresh the passages.
    return load_passages(Path(docs_path), max_chars=CHUNK_SIZE)


@st.cache_resource(show_spinner=False)
def get_index_snapshot(
    database_path: str, embedding_model: str, generation: int
) -> tuple[list[Passage], list[list[float]]]:
    """Keep the persisted SQLite index in memory for fast repeated searches."""
    del generation  # Included in the cache key; a changed index creates a new snapshot.
    return load_index_snapshot(Path(database_path))


@st.cache_data(ttl=10, show_spinner=False)
def get_ollama_models(server_url: str) -> tuple[list[str], str | None]:
    """Read locally installed Ollama models from its local tags endpoint."""
    try:
        with urlopen(f"{server_url.rstrip('/')}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = [model["name"] for model in payload.get("models", []) if model.get("name")]
        return names, None
    except (OSError, URLError, json.JSONDecodeError, KeyError) as exc:
        return [], str(exc)


def main() -> None:
    st.set_page_config(page_title="Knowledge Base Assistant", page_icon="📚", layout="wide")
    st.title("📚 AI Knowledge Base Assistant")
    st.caption("Ask questions about your local PDF and Markdown documents.")

    files = document_files()
    signature = document_signature(files)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Local setup")
        server_url = st.text_input("Ollama server", value=DEFAULT_OLLAMA_URL)
        model_names, model_error = get_ollama_models(server_url)
        answer_models = [name for name in model_names if "embed" not in name.lower()]
        embedding_models = [name for name in model_names if "embed" in name.lower()]

        if answer_models:
            selected_model = st.selectbox("Answer model", answer_models)
        else:
            selected_model = None
            st.warning("No local answer model found. Install one in Ollama.")

        if embedding_models:
            selected_embedding_model = st.selectbox("Embedding model", embedding_models)
            retrieval_mode = st.selectbox(
                "Retrieval method", ["BM25", "Semantic", "Hybrid (BM25 + semantic)"], index=2
            )
            st.caption("Hybrid combines keyword and meaning-based rankings.")
        else:
            selected_embedding_model = None
            retrieval_mode = "BM25"
            st.warning("Install an embedding model to enable semantic and hybrid search.")
            st.code('"%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe" pull embeddinggemma')
            if model_error:
                st.caption("Check that Ollama is running at the server address above.")

        st.divider()
        st.subheader(f"Documents ({len(files)})")
        if files:
            for path in files:
                st.caption(f"• {path.name}")
        else:
            st.info("Add PDF or Markdown files directly to the project's docs folder.")

        sync_requested = st.button("Sync documents", use_container_width=True)
        if selected_embedding_model:
            sync_key = (selected_embedding_model, signature)
            needs_sync = sync_requested or st.session_state.get("_synced_documents") != sync_key
            sync_succeeded = st.session_state.get("_sync_success_key") == sync_key
            if needs_sync:
                progress = st.progress(0, text="Checking document files...")

                def show_sync_progress(stage: str, done: int, total: int, detail: str) -> None:
                    ratio = done / total if total else 1.0
                    if stage == "hash":
                        label = f"Checking file: {detail}"
                    elif stage == "extract":
                        label = f"Reading {detail}: page {done:,} of {total:,}"
                    elif stage == "embed":
                        label = f"Embedding {detail}: passage {done:,} of {total:,}"
                    else:
                        label = "Syncing document index..."
                    progress.progress(min(max(ratio, 0.0), 1.0), text=label)

                try:
                    summary = sync_documents(
                        DOCS_DIR,
                        DATABASE_PATH,
                        server_url,
                        selected_embedding_model,
                        progress_callback=show_sync_progress,
                    )
                    st.session_state["_synced_documents"] = sync_key
                    st.session_state["_sync_success_key"] = sync_key
                    sync_succeeded = True
                    if summary.added or summary.updated or summary.deleted:
                        get_index_snapshot.clear()
                    progress.progress(1.0, text="Document index is up to date.")
                    st.caption(
                        f"Index ready: {summary.added} added, {summary.updated} updated, "
                        f"{summary.deleted} removed, {summary.unchanged} unchanged; "
                        f"{summary.passages:,} passages stored."
                    )
                except Exception as exc:
                    sync_succeeded = False
                    # Avoid retrying a failed sync on every Streamlit rerun; the button retries it.
                    st.session_state["_synced_documents"] = sync_key
                    st.session_state.pop("_sync_success_key", None)
                    st.error(f"Document indexing failed: {exc}. Use Sync documents to retry.")
                finally:
                    progress.empty()
            else:
                st.caption("Saved document embeddings are ready; unchanged files are reused.")
            if sync_succeeded:
                generation = index_generation(DATABASE_PATH)
                get_index_snapshot(str(DATABASE_PATH), selected_embedding_model, generation)
        else:
            sync_succeeded = False
            st.caption("Keyword search is available. Install an embedding model to build the saved vector index.")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.info(
        "The document index is synchronized when the app opens and when you click Sync documents. "
        "New or changed files are embedded once; deleted files are removed from the index."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("evidence"):
                with st.expander("Retrieved passages"):
                    for source, text in message["evidence"]:
                        st.markdown(f"**{source}**")
                        st.write(text)

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    answer = ""
    evidence: list[tuple[str, str]] = []
    with st.chat_message("assistant"):
        try:
            if retrieval_mode == "BM25":
                if selected_embedding_model and sync_succeeded:
                    generation = index_generation(DATABASE_PATH)
                    passages, _ = get_index_snapshot(
                        str(DATABASE_PATH), selected_embedding_model, generation
                    )
                else:
                    passages = get_passages(str(DOCS_DIR), signature)
                with st.spinner("Searching by keyword..."):
                    results = search_bm25(question, passages, limit=3)
            else:
                if not selected_embedding_model:
                    raise RuntimeError("No embedding model is installed in Ollama.")
                if not sync_succeeded:
                    raise RuntimeError("The saved document index is not ready. Fix Ollama and click Sync documents.")
                generation = index_generation(DATABASE_PATH)
                passages, passage_vectors = get_index_snapshot(
                    str(DATABASE_PATH), selected_embedding_model, generation
                )
                with st.spinner("Searching the saved document embeddings..."):
                    query_vector = embed_texts(
                        [question], server_url, selected_embedding_model
                    )[0]
                    if retrieval_mode == "Semantic":
                        results = search_semantic(
                            passages, passage_vectors, query_vector, limit=3
                        )
                    else:
                        results = search_hybrid(
                            question,
                            passages,
                            passage_vectors,
                            query_vector,
                            limit=3,
                        )
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            return

        if not results:
            answer = "I could not find a matching passage in the documents. Try different keywords."
        else:
            evidence = [(passage.source, passage.text) for _, passage in results]
            if selected_model is None:
                answer = "I found relevant passages, but no local Ollama answer model is available. The passages are shown below."
            else:
                with st.spinner(f"Generating an answer with {selected_model}..."):
                    try:
                        answer = generate_answer(
                            question,
                            results,
                            PROJECT_ROOT,
                            base_url_override=f"{server_url.rstrip('/')}/v1",
                            model_override=selected_model,
                            api_key_override="ollama",
                        )
                    except (ValueError, RuntimeError) as exc:
                        answer = f"I found relevant passages, but answer generation failed: {exc}"

        st.caption(f"Retrieved with {retrieval_mode}.")
        st.markdown(answer)
        if evidence:
            with st.expander("Retrieved passages"):
                for source, text in evidence:
                    st.markdown(f"**{source}**")
                    st.write(text)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "evidence": evidence}
    )


if __name__ == "__main__":
    main()
