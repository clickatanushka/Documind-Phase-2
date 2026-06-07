"""
core/embeddings.py
Phase 1 — text → OpenAI embeddings

Phase 3 hook: add semantic cache check BEFORE calling OpenAI.
              If cache hit, return cached embedding + flag.
"""

from __future__ import annotations
import os
import time
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from core.ingestion import Chunk

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536   # dimensions for text-embedding-3-small
BATCH_SIZE      = 100    # OpenAI allows up to 2048 inputs per call


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)


def embed_texts(texts: list[str], client: OpenAI | None = None) -> list[list[float]]:
    """
    Embed a list of strings. Batches automatically.
    Returns list of float vectors, same order as input.

    Phase 3: before calling OpenAI, check semantic cache.
             Cache structure: {vector → cached_response}
    """
    if client is None:
        client = get_client()

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        # response.data is sorted by index
        batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embeddings)

        # Polite rate limiting for large batches
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.05)

    return all_embeddings


def embed_chunks(chunks: list["Chunk"], client: OpenAI | None = None) -> list[list[float]]:
    """Convenience wrapper: embed the .text field of each Chunk."""
    texts = [c.text for c in chunks]
    return embed_texts(texts, client)


def embed_query(query: str, client: OpenAI | None = None) -> list[float]:
    """
    Embed a single query string.
    IMPORTANT: must use the same model as embed_chunks — otherwise
    query vectors live in a different space and similarity search breaks.

    Phase 3: check semantic cache here first.
    """
    return embed_texts([query], client)[0]
