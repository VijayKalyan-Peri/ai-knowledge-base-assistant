from __future__ import annotations

import hashlib
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.embeddings import embed_texts
from src.rag import Passage, load_document

CHUNK_SIZE = 1200
SCHEMA_VERSION = "1"
ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class SyncSummary:
    added: int
    updated: int
    deleted: int
    unchanged: int
    passages: int


def _connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            passage_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS passages (
            document_path TEXT NOT NULL REFERENCES documents(path) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            source TEXT NOT NULL,
            text TEXT NOT NULL,
            vector BLOB NOT NULL,
            dimensions INTEGER NOT NULL,
            PRIMARY KEY (document_path, ordinal)
        );
        """
    )
    return connection


def _metadata(connection: sqlite3.Connection, key: str, default: str = "") -> str:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def _set_metadata(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _bump_generation(connection: sqlite3.Connection) -> None:
    generation = int(_metadata(connection, "generation", "0")) + 1
    _set_metadata(connection, "generation", str(generation))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_documents(
    docs_dir: Path,
    database_path: Path,
    server_url: str,
    embedding_model: str,
    progress_callback: ProgressCallback | None = None,
) -> SyncSummary:
    """Reconcile SQLite's saved vectors with the current PDF/Markdown documents."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".md"}
    )
    connection = _connect(database_path)
    try:
        stored_model = _metadata(connection, "embedding_model")
        stored_schema = _metadata(connection, "schema_version")
        if stored_model and (stored_model != embedding_model or stored_schema != SCHEMA_VERSION):
            connection.execute("DELETE FROM documents")
            _bump_generation(connection)
        _set_metadata(connection, "embedding_model", embedding_model)
        _set_metadata(connection, "schema_version", SCHEMA_VERSION)
        _set_metadata(connection, "chunk_size", str(CHUNK_SIZE))
        connection.commit()

        existing_rows = connection.execute(
            "SELECT path, content_hash, file_size, mtime_ns FROM documents"
        ).fetchall()
        existing = {row[0]: row[1:] for row in existing_rows}
        current_names = {path.name for path in files}
        deleted_names = sorted(set(existing) - current_names)
        for name in deleted_names:
            connection.execute("DELETE FROM documents WHERE path = ?", (name,))
        if deleted_names:
            _bump_generation(connection)
            connection.commit()

        added = updated = unchanged = 0
        for file_number, path in enumerate(files, start=1):
            stat = path.stat()
            old = existing.get(path.name)
            if old and old[1] == stat.st_size and old[2] == stat.st_mtime_ns:
                unchanged += 1
                continue

            if progress_callback:
                progress_callback("hash", file_number, len(files), path.name)
            content_hash = _file_hash(path)
            if old and old[0] == content_hash:
                connection.execute(
                    "UPDATE documents SET file_size = ?, mtime_ns = ? WHERE path = ?",
                    (stat.st_size, stat.st_mtime_ns, path.name),
                )
                connection.commit()
                unchanged += 1
                continue

            passages = load_document(
                path,
                max_chars=CHUNK_SIZE,
                progress_callback=(
                    (lambda stage, done, total, detail: progress_callback(stage, done, total, detail))
                    if progress_callback else None
                ),
            )
            if progress_callback:
                progress_callback("embed", 0, len(passages), path.name)
            vectors = embed_texts(
                [passage.text for passage in passages],
                server_url,
                embedding_model,
                batch_size=32,
                progress_callback=(
                    (lambda done, total: progress_callback("embed", done, total, path.name))
                    if progress_callback else None
                ),
            )
            if len(passages) != len(vectors):
                raise RuntimeError(f"Embedding count did not match passages for {path.name}.")

            try:
                connection.execute("DELETE FROM documents WHERE path = ?", (path.name,))
                connection.execute(
                    "INSERT INTO documents(path, content_hash, file_size, mtime_ns, passage_count) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (path.name, content_hash, stat.st_size, stat.st_mtime_ns, len(passages)),
                )
                for ordinal, (passage, vector) in enumerate(zip(passages, vectors)):
                    vector_blob = struct.pack(f"<{len(vector)}f", *vector)
                    connection.execute(
                        "INSERT INTO passages(document_path, ordinal, source, text, vector, dimensions) "
                        "VALUES(?, ?, ?, ?, ?, ?)",
                        (path.name, ordinal, passage.source, passage.text, vector_blob, len(vector)),
                    )
                _bump_generation(connection)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

            if old:
                updated += 1
            else:
                added += 1

        total_passages = connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        return SyncSummary(added, updated, len(deleted_names), unchanged, total_passages)
    finally:
        connection.close()


def index_generation(database_path: Path) -> int:
    if not database_path.exists():
        return 0
    connection = _connect(database_path)
    try:
        return int(_metadata(connection, "generation", "0"))
    finally:
        connection.close()


def load_index_snapshot(database_path: Path) -> tuple[list[Passage], list[list[float]]]:
    """Load stored passages and float32 vectors for retrieval; creates no embeddings."""
    if not database_path.exists():
        return [], []
    connection = _connect(database_path)
    try:
        rows = connection.execute(
            "SELECT p.source, p.text, p.vector, p.dimensions "
            "FROM passages AS p JOIN documents AS d ON d.path = p.document_path "
            "ORDER BY p.document_path, p.ordinal"
        ).fetchall()
    finally:
        connection.close()

    passages: list[Passage] = []
    vectors: list[list[float]] = []
    for source, text, vector_blob, dimensions in rows:
        passages.append(Passage(source=source, text=text))
        vectors.append(list(struct.unpack(f"<{dimensions}f", vector_blob)))
    return passages, vectors
