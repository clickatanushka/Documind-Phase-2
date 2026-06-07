"""
core/vector_store.py
Phase 1 — Pinecone index management + upsert + delete

Phase 2 hook: upsert_chunks() will be called from a Celery worker,
              not the request thread.
Phase 3 hook: add metadata filters (e.g., doc_name) to query_index()
              for scoped retrieval.
"""

from __future__ import annotations
import os
import uuid
from typing import TYPE_CHECKING

from pinecone import Pinecone, ServerlessSpec

if TYPE_CHECKING:
    from core.ingestion import Chunk

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "documind")
CLOUD      = os.getenv("PINECONE_CLOUD", "aws")
REGION     = os.getenv("PINECONE_REGION", "us-east-1")
DIMENSION  = 1536   # matches text-embedding-3-small
METRIC     = "cosine"
UPSERT_BATCH = 100


def get_index():
    """Return a Pinecone Index object, creating the index if it doesn't exist."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not set in environment.")

    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )

    return pc.Index(INDEX_NAME)


def upsert_chunks(
    chunks: list["Chunk"],
    embeddings: list[list[float]],
) -> int:
    """
    Upsert (chunk, embedding) pairs into Pinecone.
    Returns number of vectors upserted.

    Vector ID format: {doc_name}_{chunk_index}
    This makes it easy to delete all chunks for a document.
    """
    index = get_index()

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vec_id = f"{chunk.doc_name}_{chunk.chunk_index}"
        vectors.append({
            "id": vec_id,
            "values": embedding,
            "metadata": chunk.to_pinecone_metadata(),
        })

    # Batch upsert
    for i in range(0, len(vectors), UPSERT_BATCH):
        batch = vectors[i : i + UPSERT_BATCH]
        index.upsert(vectors=batch)

    return len(vectors)


def query_index(
    query_embedding: list[float],
    top_k: int = 5,
    filter_doc: str | None = None,   # Phase 3: filter by document name
) -> list[dict]:
    """
    Similarity search. Returns list of metadata dicts for top_k results.

    Phase 3: add `filter={"doc_name": filter_doc}` for scoped retrieval.
    """
    index = get_index()

    query_kwargs: dict = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
    }

    # Phase 3: metadata filtering
    if filter_doc:
        query_kwargs["filter"] = {"doc_name": {"$eq": filter_doc}}

    response = index.query(**query_kwargs)

    results = []
    for match in response.matches:
        results.append({
            "id": match.id,
            "score": round(match.score, 4),
            "text": match.metadata.get("text", ""),
            "doc_name": match.metadata.get("doc_name", ""),
            "page_num": match.metadata.get("page_num", 0),
            "chunk_index": match.metadata.get("chunk_index", 0),
        })

    return results


def delete_document(doc_name: str) -> None:
    """
    Delete all vectors for a given document from the index.
    Uses Pinecone metadata filter delete.
    """
    index = get_index()
    index.delete(filter={"doc_name": {"$eq": doc_name}})


def get_index_stats() -> dict:
    """Return index stats: total vectors, dimension, etc."""
    index = get_index()
    stats = index.describe_index_stats()
    return {
        "total_vectors": stats.total_vector_count,
        "dimension": stats.dimension,
        "index_name": INDEX_NAME,
    }


def list_indexed_documents(index=None) -> list[str]:
    """
    Returns a list of unique doc_names stored in the index.
    Uses a dummy query to fetch metadata — Pinecone doesn't have
    a native 'list all metadata values' endpoint.

    Note: This is a best-effort scan. For production, maintain a
    separate doc registry (Phase 2: store in Redis or a DB).
    """
    if index is None:
        index = get_index()
    # Fetch with a zero vector to get any results
    zero_vec = [0.0] * DIMENSION
    response = index.query(vector=zero_vec, top_k=100, include_metadata=True)
    docs = list({m.metadata.get("doc_name", "") for m in response.matches if m.metadata})
    return sorted(d for d in docs if d)
