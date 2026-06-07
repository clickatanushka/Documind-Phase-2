# DocuMind — Production RAG Engineer

Enterprise document assistant built across 4 phases.

## Quick Start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Set env vars
cp .env.example .env
# Fill in OPENAI_API_KEY and PINECONE_API_KEY

# 3. Run
streamlit run ui/app.py
```

## Architecture

```
documind/
├── core/               # Phase 1 — RAG pipeline
│   ├── ingestion.py    # PDF → chunks
│   ├── embeddings.py   # chunks → vectors
│   ├── vector_store.py # Pinecone CRUD
│   ├── retrieval.py    # similarity search
│   ├── generation.py   # LLM call + prompt template
│   └── pipeline.py     # end-to-end orchestrator
│
├── phases/             # Future phases (stubs ready)
│   ├── phase2_async/   # Week 6-7: Celery + Redis
│   ├── phase3_hard/    # Week 8-11: guardrails, cache, eval, RAGAS
│   └── phase4_obs/     # Week 12: cost/latency dashboard
│
├── ui/
│   └── app.py          # Streamlit interface
│
├── data/               # Local PDF storage
├── tests/              # Unit tests per module
├── .env.example
└── requirements.txt
```

## Phase Status

| Phase | Weeks | Status |
|-------|-------|--------|
| 1 — Build Core | 1–5 | ✅ Active |
| 2 — Production Shape | 6–7 | 🔲 Stub ready |
| 3 — Hardening | 8–11 | 🔲 Stub ready |
| 4 — Observability | 12 | 🔲 Stub ready |

## Component Choices

| Component | Choice | Why |
|-----------|--------|-----|
| Embedding | `text-embedding-3-small` | Fast, cheap, good English retrieval |
| Vector DB | Pinecone | Free tier, metadata filtering, managed |
| LLM | `gpt-4o-mini` | Cost-effective for dev; swap to gpt-4o for prod |
| Chunking | Fixed-size, 400 tok, 50 overlap | Debuggable baseline; revisit in Week 2 |
| Framework | Streamlit | Fastest path to a shareable demo UI |
