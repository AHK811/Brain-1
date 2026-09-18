"""Retrieve top chunks and format a prompt block for the model."""

from __future__ import annotations

from typing import Optional

from brain.experimental.rag.index import RAGIndex
from brain.experimental.sft.templates import format_chat


def retrieve_and_format(
    index: RAGIndex,
    user_query: str,
    *,
    top_k: int = 5,
    domain: Optional[str] = None,
    system: Optional[str] = None,
    max_context_chars: int = 4000,
    add_generation_prompt: bool = True,
) -> str:
    """
    Build: system + retrieved context + user question in Brain chat format.
    """
    hits = index.search(user_query, top_k=top_k)
    parts = []
    total = 0
    for chunk, score in hits:
        block = f"[{chunk.source or chunk.id} | score={score:.3f}]\n{chunk.text}\n"
        if total + len(block) > max_context_chars:
            break
        parts.append(block)
        total += len(block)
    context = "\n".join(parts) if parts else "(no passages retrieved)"
    sys = system or (
        "Use the provided context passages to answer. "
        "If context is insufficient, say what is missing. Cite sources when useful."
    )
    sys = f"{sys}\n\nContext:\n{context}"
    return format_chat(
        user_query,
        system=sys,
        domain=domain,
        add_generation_prompt=add_generation_prompt,
    )
