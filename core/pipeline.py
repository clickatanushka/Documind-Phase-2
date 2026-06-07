"""
core/pipeline.py
Orchestrator — Phase 1 (sync) + Phase 2 (async) entry points.

UI calls:
  - ingest_document()      -> Phase 1 sync (blocks until done)
  - submit_ingest_job()    -> Phase 2 async (returns job_id instantly)
  - get_ingest_status()    -> Phase 2 poll (call repeatedly from UI)
  - query_document()       -> both phases (unchanged)
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

from core.ingestion    import ingest_pdf
from core.embeddings   import embed_chunks, embed_query
from core.vector_store import upsert_chunks, query_index, delete_document, get_index_stats
from core.generation   import generate_answer


def _async_available() -> bool:
    """True if Redis + Celery are installed and REDIS_URL is set."""
    try:
        import celery, redis  # noqa
        return bool(os.getenv("REDIS_URL") or os.getenv("CELERY_BROKER_URL"))
    except ImportError:
        return False


# ── Result models ─────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    doc_name: str
    chunk_count: int
    page_count: int
    total_tokens: int
    status: str = "success"
    error: str = ""


@dataclass
class QueryResult:
    answer: str
    sources: list[dict]
    query: str
    top_k: int


# ── Phase 1: Sync ingestion ───────────────────────────────────────────────────

def ingest_document(
    pdf_path: str | Path,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    progress_callback=None,
) -> IngestResult:
    """
    Synchronous ingestion — blocks until complete.
    Used by Phase 1 UI or when Redis is not available.
    progress_callback(step: str, pct: float) updates the Streamlit progress bar.
    """
    pdf_path = Path(pdf_path)
    try:
        if progress_callback:
            progress_callback("Extracting & chunking PDF...", 0.10)
        chunks = ingest_pdf(pdf_path, chunk_size, chunk_overlap)

        if progress_callback:
            progress_callback("Generating embeddings...", 0.40)
        embeddings = embed_chunks(chunks)

        if progress_callback:
            progress_callback("Uploading to vector store...", 0.80)
        upsert_chunks(chunks, embeddings)

        if progress_callback:
            progress_callback("Done.", 1.0)

        pages = sorted(set(c.page_num for c in chunks))
        return IngestResult(
            doc_name=pdf_path.stem,
            chunk_count=len(chunks),
            page_count=len(pages),
            total_tokens=sum(c.token_count for c in chunks),
        )
    except Exception as e:
        return IngestResult(
            doc_name=pdf_path.stem,
            chunk_count=0, page_count=0, total_tokens=0,
            status="error", error=str(e),
        )


# ── Phase 2: Async ingestion ──────────────────────────────────────────────────

def submit_ingest_job(
    pdf_path: str | Path,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> str:
    """
    Submit ingestion to Celery. Returns job_id in < 500 ms.
    Raises RuntimeError if Redis/Celery is not available.
    """
    if not _async_available():
        raise RuntimeError(
            "Phase 2 requires Redis. Set REDIS_URL in .env and start the Celery worker."
        )
    from phases.phase2_async.job_status import submit_ingest_job as _submit
    return _submit(str(pdf_path), chunk_size, chunk_overlap)


def get_ingest_status(job_id: str):
    """
    Poll Celery for job status. Returns a JobStatus dataclass.
    Call this repeatedly from the UI (every 1-2 seconds).
    """
    from phases.phase2_async.job_status import get_job_status
    return get_job_status(job_id)


# ── Query (unchanged across phases) ──────────────────────────────────────────

def query_document(
    question: str,
    top_k: int = 5,
    stream: bool = True,
    filter_doc: str | None = None,
):
    """
    Query pipeline: question -> embed -> retrieve -> generate.
    Returns (generator, sources) if stream=True, else QueryResult.
    Phase 3: add guardrails.check_input() before embed_query()
    """
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    q_embedding = embed_query(question)
    sources = query_index(q_embedding, top_k=top_k, filter_doc=filter_doc)

    if stream:
        return generate_answer(question, sources, stream=True), sources
    else:
        answer = generate_answer(question, sources, stream=False)
        return QueryResult(answer=answer, sources=sources, query=question, top_k=top_k)


# ── Utilities ─────────────────────────────────────────────────────────────────

def remove_document(doc_name: str) -> None:
    delete_document(doc_name)

def index_stats() -> dict:
    return get_index_stats()

def async_available() -> bool:
    return _async_available()
