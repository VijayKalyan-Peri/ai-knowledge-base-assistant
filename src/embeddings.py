from __future__ import annotations

import json
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def embed_texts(
    texts: list[str],
    server_url: str,
    model: str,
    batch_size: int = 64,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Create local vector embeddings in bounded batches through Ollama."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not texts:
        return []

    endpoint = f"{server_url.rstrip('/')}/api/embed"
    embeddings: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        body = json.dumps({"model": model, "input": batch, "truncate": True}).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(
                f"Ollama embedding request failed (HTTP {exc.code}): {details} "
                f"Check that '{model}' is installed and supports embeddings."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {server_url}: {exc.reason}"
            ) from exc

        batch_vectors = payload.get("embeddings")
        if not isinstance(batch_vectors, list) or len(batch_vectors) != len(batch):
            raise RuntimeError("Ollama returned an unexpected embedding response.")
        embeddings.extend(batch_vectors)
        if progress_callback:
            progress_callback(len(embeddings), total)
    return embeddings
