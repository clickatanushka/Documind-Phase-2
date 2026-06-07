import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide")

# ── env + phase detection ──────────────────────────────────────────────────────
missing = [k for k in ["OPENAI_API_KEY", "PINECONE_API_KEY"] if not os.getenv(k)]
if missing:
    st.warning(f"Missing: {', '.join(missing)} — add to .env and restart.")

from core.pipeline import async_available
ASYNC = async_available()

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"# DocuMind\n`{'PHASE 2 · ASYNC' if ASYNC else 'PHASE 1 · CORE'}`")
    chunk_size    = st.slider("Chunk size (tokens)",    100, 800, 400, 50)
    chunk_overlap = st.slider("Chunk overlap (tokens)",   0, 150,  50, 10)
    top_k         = st.slider("Top-k chunks",             1,  15,   5,  1)
    st.divider()
    st.markdown(f"""
- ✅ Phase 1 — Core RAG
- {'✅' if ASYNC else '🔲'} Phase 2 — Async
- 🔲 Phase 3 — Guardrails
- 🔲 Phase 4 — Observability
""")

if "jobs" not in st.session_state:
    st.session_state.jobs = []

tab_main, tab_jobs, tab_docs = st.tabs(["Ask", "Job Queue", "Documents"])


# ═══════════════════════════════
# ASK
# ═══════════════════════════════
with tab_main:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(f"**Upload** — {'⚡ Async' if ASYNC else '🔄 Sync'}")
        uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

        if uploaded and st.button("Ingest document", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            if ASYNC:
                from core.pipeline import submit_ingest_job
                try:
                    job_id = submit_ingest_job(tmp_path, chunk_size, chunk_overlap)
                    st.success("Job submitted!")
                    st.code(job_id)
                    st.caption("→ Track progress in Job Queue tab")
                    st.session_state.jobs.insert(0, {"job_id": job_id, "filename": uploaded.name})
                except Exception as e:
                    st.error(e)
            else:
                from core.pipeline import ingest_document
                pb, cap = st.progress(0), st.empty()
                r = ingest_document(tmp_path, chunk_size, chunk_overlap,
                                    lambda step, pct: (pb.progress(pct), cap.caption(step)))
                os.unlink(tmp_path)
                if r.status == "error":
                    st.error(r.error)
                else:
                    st.success(f"Ingested: {r.doc_name}")
                    st.caption(f"{r.chunk_count} chunks · {r.page_count} pages · {r.total_tokens:,} tokens")

        elif not uploaded:
            st.caption(f"Chunk size: {chunk_size} · Overlap: {chunk_overlap} · Top-k: {top_k}")

    with right:
        st.markdown("**Question**")
        question    = st.text_area("", placeholder="What is the refund policy?",
                                   height=120, label_visibility="collapsed")
        col1, col2  = st.columns([2, 1])
        show_chunks = col2.toggle("Sources", value=True)
        ask         = col1.button("Ask", use_container_width=True)

        if ask:
            if not question.strip():
                st.warning("Enter a question.")
            else:
                try:
                    from core.pipeline import query_document
                    with st.spinner(""):
                        gen, sources = query_document(question=question, top_k=top_k, stream=True)
                    st.markdown("**Answer**")
                    st.write_stream(gen)
                    if show_chunks and sources:
                        st.divider()
                        st.caption(f"SOURCES — {len(sources)} chunks")
                        for i, c in enumerate(sources, 1):
                            pct = int(c["score"] * 100)
                            st.caption(f"#{i} · {c['doc_name']} · p.{c['page_num']} · {pct}% match")
                            st.markdown(c["text"][:400] + ("…" if len(c["text"]) > 400 else ""))
                            st.divider()
                    elif not sources:
                        st.caption("No chunks found — ingest a document first.")
                except Exception as e:
                    st.error(e)


# ═══════════════════════════════
# JOB QUEUE
# ═══════════════════════════════
with tab_jobs:
    st.markdown("**Ingestion Job Queue**")
    if not ASYNC:
        st.info("Phase 2 feature — start Redis + Celery to enable.")
        st.code("redis-server &\ncelery -A phases.phase2_async.worker worker --loglevel=info")
    else:
        if st.button("Refresh"):
            st.rerun()
        jobs = st.session_state.jobs
        if not jobs:
            st.caption("No jobs yet.")
        else:
            from core.pipeline import get_ingest_status
            for job in jobs:
                s = get_ingest_status(job["job_id"])
                color = {"SUCCESS": "green", "FAILURE": "red",
                         "PROGRESS": "orange", "STARTED": "orange"}.get(s.state, "gray")
                st.markdown(f"**{job.get('filename','—')}** · :{color}[{s.state}]")
                st.caption(f"`{job['job_id']}` — {s.step}")
                if s.state == "PROGRESS":
                    st.progress(s.pct)
                elif s.state == "SUCCESS" and s.result:
                    r = s.result
                    st.caption(f"{r.get('chunk_count','—')} chunks · {r.get('page_count','—')} pages · {r.get('total_tokens',0):,} tokens")
                elif s.state == "FAILURE":
                    st.error(s.error)
                st.divider()


# ═══════════════════════════════
# DOCUMENTS
# ═══════════════════════════════
with tab_docs:
    st.markdown("**Indexed documents**")
    if st.button("Refresh", key="rdoc"):
        st.rerun()
    if not missing:
        try:
            from core.vector_store import list_indexed_documents, get_index_stats
            s = get_index_stats()
            st.caption(f"{s['total_vectors']:,} vectors · {s['dimension']} dims · cosine")
            for doc in list_indexed_documents():
                c1, c2 = st.columns([6, 1])
                c1.markdown(f"`{doc}`")
                if c2.button("Remove", key=f"d_{doc}"):
                    from core.pipeline import remove_document
                    remove_document(doc)
                    st.rerun()
        except Exception as e:
            st.error(e)
    else:
        st.caption("Add API keys to .env to see documents.")