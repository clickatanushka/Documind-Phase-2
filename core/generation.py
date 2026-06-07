"""
core/generation.py
Phase 1 — prompt template + OpenAI chat completion

Phase 3 hooks:
  - add guardrails check BEFORE sending to LLM
  - add output grounding validation AFTER response
  - add semantic cache lookup BEFORE LLM call
"""

from __future__ import annotations
import os
from typing import Generator

from openai import OpenAI


# ── Prompt template ───────────────────────────────────────────────────────────
# Design principles (Week 3):
#   - Explicit citation instruction ("cite the document name and page")
#   - Grounding constraint ("only use the provided context")
#   - Refusal instruction ("say you don't know if context is insufficient")
#   - System / user separation for clean few-shot extension later

SYSTEM_PROMPT = """You are DocuMind, an enterprise document assistant.
Your job is to answer questions using ONLY the context passages provided.

Rules:
1. Base your answer strictly on the provided context. Do not use prior knowledge.
2. Cite your sources: after each claim, add (Source: <doc_name>, p.<page_num>).
3. If the context does not contain enough information, say:
   "I couldn't find a clear answer in the uploaded documents."
4. Be concise and factual. No filler phrases.
5. If multiple passages are relevant, synthesize them into one coherent answer."""

USER_TEMPLATE = """Context passages:
{context_block}

Question: {question}

Answer (with citations):"""


def _build_context_block(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        lines.append(
            f"[{i}] (doc: {chunk['doc_name']}, page {chunk['page_num']})\n{chunk['text']}"
        )
    return "\n\n".join(lines)


# ── LLM call ─────────────────────────────────────────────────────────────────

LLM_MODEL = "gpt-4o-mini"   # swap to "gpt-4o" for higher quality


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    stream: bool = True,
    model: str = LLM_MODEL,
) -> str | Generator[str, None, None]:
    """
    Generate a grounded, cited answer from retrieved chunks.

    Args:
        question: the user's natural language question
        retrieved_chunks: list of dicts from vector_store.query_index()
        stream: if True, return a generator of text deltas (for Streamlit)
        model: OpenAI model name

    Phase 3 hooks (insert before/after this function):
        BEFORE: guardrails.check_input(question)
        BEFORE: semantic_cache.lookup(question)
        AFTER:  guardrails.check_output(response, retrieved_chunks)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)

    if not retrieved_chunks:
        msg = "I couldn't find relevant passages in the uploaded documents for your question."
        if stream:
            def _empty():
                yield msg
            return _empty()
        return msg

    context_block = _build_context_block(retrieved_chunks)
    user_message = USER_TEMPLATE.format(
        context_block=context_block,
        question=question,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    if stream:
        return _stream_response(client, messages, model)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,  # low temp for factual, grounded answers
        )
        return response.choices[0].message.content


def _stream_response(
    client: OpenAI,
    messages: list[dict],
    model: str,
) -> Generator[str, None, None]:
    """Yield text deltas for streaming into Streamlit's st.write_stream."""
    with client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
