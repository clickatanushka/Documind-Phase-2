"""
phases/phase2_async/tasks.py
The actual ingestion Celery task.

This wraps the existing Phase 1 pipeline functions — nothing in core/ changes.
The task reports progress via Celery's update_state(), which the UI polls.
"""

from __future__ import annotations
import os
import sys

# Ensure project root is on path when worker is launched
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv()

from phases.phase2_async.worker import celery_app
from core.ingestion    import ingest_pdf
from core.embeddings   import embed_chunks
from core.vector_store import upsert_chunks


@celery_app.task(bind=True, name="documind.ingest")
def ingest_task(
    self,
    pdf_path: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> dict:
    """
    Async ingestion task: PDF → chunks → embeddings → Pinecone upsert.

    Progress states reported to Redis:
        STARTED   → task picked up by worker
        PROGRESS  → step name + percentage
        SUCCESS   → result dict with stats
        FAILURE   → error message

    Returns dict with: doc_name, chunk_count, page_count, total_tokens
    """
    from pathlib import Path
    pdf_path = Path(pdf_path)

    def progress(step: str, pct: float):
        self.update_state(
            state="PROGRESS",
            meta={"step": step, "pct": round(pct, 2)},
        )

    try:
        # Step 1 — chunk
        progress("Extracting & chunking PDF...", 0.10)
        chunks = ingest_pdf(pdf_path, chunk_size, chunk_overlap)

        # Step 2 — embed (most time is spent here)
        progress("Generating embeddings...", 0.40)
        embeddings = embed_chunks(chunks)

        # Step 3 — upsert
        progress("Uploading to vector store...", 0.80)
        upsert_chunks(chunks, embeddings)

        progress("Done.", 1.0)

        pages = sorted(set(c.page_num for c in chunks))
        return {
            "status": "success",
            "doc_name": pdf_path.stem,
            "chunk_count": len(chunks),
            "page_count": len(pages),
            "total_tokens": sum(c.token_count for c in chunks),
        }

    except Exception as exc:
        # Celery will mark task as FAILURE and store the exception
        self.update_state(
            state="FAILURE",
            meta={"step": "error", "pct": 0, "error": str(exc)},
        )
        raise exc
