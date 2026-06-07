"""
phases/phase3_hard/__init__.py

Phase 3 — Hardening (Weeks 8–11)

WEEK 8 — Guardrails
  - Block prompt injection (regex + classifier)
  - PII filter with Presidio before sending to LLM
  - Output grounding check: answer must cite retrieved context
  - Off-topic refusal: query outside document scope → reject

WEEK 9 — Semantic Caching
  - Cache (query_embedding, answer) pairs in Redis
  - On new query: cosine sim vs cached embeddings
  - If sim > threshold (e.g. 0.92): return cached answer
  - Benchmark: p50/p95 latency + cost before vs after

WEEK 10 — Evaluation + Drift
  - Log per-query: faithfulness, relevance, latency
  - Rolling baseline (last N queries)
  - Alert when metric drops > X% from baseline

WEEK 11 — RAGAS
  - Plug in RAGAS library
  - Evaluate on labeled Q&A test set (30 pairs)
  - Metrics: faithfulness, answer_relevancy, context_precision, context_recall
  - Commit CSV report + chart to docs/evaluation.md

SWAP POINTS:
  - core/generation.py has # Phase 3 comment blocks marking injection points
  - core/pipeline.py query_document() has guardrail hook comment
"""
