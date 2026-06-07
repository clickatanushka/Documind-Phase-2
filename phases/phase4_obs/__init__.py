"""
phases/phase4_obs/__init__.py

Phase 4 — Observability (Week 12)

WHAT TO BUILD:
  - Instrument every LLM call: token count, cost ($), latency (ms)
  - Instrument every retrieval: top-k scores, cache hit/miss
  - Streamlit dashboard page (or Grafana panel):
      * cost-per-day time series
      * p95 latency time series
      * cache hit rate
      * per-document query volume

IMPLEMENTATION OPTIONS:
  A) Streamlit: add a "📊 Dashboard" page to the existing app
  B) Prometheus + Grafana: instrument with prometheus_client, scrape + visualize
  C) Weights & Biases: log runs and build W&B reports

SWAP POINTS:
  - Wrap core/generation.py LLM calls with a @track_cost decorator
  - Wrap core/vector_store.py query_index() with @track_latency
  - Persist metrics to SQLite (simple) or Postgres (production)
"""
