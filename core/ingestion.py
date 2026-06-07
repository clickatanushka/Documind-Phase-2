"""
core/ingestion.py
Phase 1 — PDF → cleaned text → chunks

Phase 2 hook: replace `ingest_pdf()` return value with a job_id
              and move chunking into a Celery task.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
from pypdf import PdfReader


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """One retrievable passage from a document."""
    text: str
    doc_name: str
    page_num: int          # 1-indexed
    chunk_index: int       # position within doc
    token_count: int
    # Phase 3: add `embedding: list[float] | None = None`
    metadata: dict = field(default_factory=dict)

    def to_pinecone_metadata(self) -> dict:
        return {
            "text": self.text,
            "doc_name": self.doc_name,
            "page_num": self.page_num,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            **self.metadata,
        }


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str | Path) -> list[tuple[int, str]]:
    """
    Returns list of (page_num, page_text) tuples.
    page_num is 1-indexed.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _clean_text(text)
        if text.strip():
            pages.append((i, text))
    return pages


def _clean_text(text: str) -> str:
    """Remove noise: repeated whitespace, control chars, hyphenation."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # dehyphenate
    return text.strip()


# ── Chunking ─────────────────────────────────────────────────────────────────

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def chunk_pages(
    pages: list[tuple[int, str]],
    doc_name: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Fixed-size token chunking with overlap across page boundaries.

    Design note: We chunk across pages (not per-page) so context
    that spans a page break isn't lost. The page_num stored is the
    page where the chunk *starts*.

    Phase 3 swap: replace with semantic chunking when retrieval data
    shows fixed-size is losing context at boundaries.
    """
    # Flatten all pages into tokens, tracking page boundaries
    all_tokens: list[int] = []
    token_page_map: list[int] = []  # token_idx → page_num

    for page_num, text in pages:
        tokens = _ENCODER.encode(text)
        all_tokens.extend(tokens)
        token_page_map.extend([page_num] * len(tokens))

    chunks: list[Chunk] = []
    start = 0
    chunk_idx = 0

    while start < len(all_tokens):
        end = min(start + chunk_size, len(all_tokens))
        token_slice = all_tokens[start:end]
        text = _ENCODER.decode(token_slice)
        page_num = token_page_map[start]

        chunks.append(Chunk(
            text=text,
            doc_name=doc_name,
            page_num=page_num,
            chunk_index=chunk_idx,
            token_count=len(token_slice),
        ))

        chunk_idx += 1
        start += chunk_size - chunk_overlap  # slide with overlap

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: str | Path,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Full ingestion: PDF → cleaned pages → chunks.

    Returns list[Chunk] ready for embedding + upsert.

    Phase 2: wrap this in a Celery task. Return job_id immediately,
             run this function asynchronously.
    """
    pdf_path = Path(pdf_path)
    doc_name = pdf_path.stem

    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError(f"No extractable text in {pdf_path.name}")

    chunks = chunk_pages(pages, doc_name, chunk_size, chunk_overlap)
    return chunks
