
"""Pre-retrieval: query rewrite, sub-query decomposition, HyDE stub."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class QueryPlan:
    original: str
    rewritten: str
    sub_queries: list[str]
    hyde_doc: str | None = None

def rewrite_query(query: str) -> str:
    q = " ".join(query.strip().split())
    # light expansion heuristics
    if q.endswith("?"):
        q = q[:-1]
    return q

def decompose(query: str) -> list[str]:
    parts = [p.strip() for p in query.replace("?", ".").split(".") if p.strip()]
    if len(parts) <= 1 and " and " in query.lower():
        parts = [p.strip() for p in query.lower().split(" and ") if p.strip()]
    return parts or [query]

def hyde_stub(query: str) -> str:
    """Hypothetical document (without LLM): template prose for embedding."""
    return f"This document answers the question: {query}. It provides key facts, definitions, and examples."

def plan_query(query: str) -> QueryPlan:
    rw = rewrite_query(query)
    return QueryPlan(original=query, rewritten=rw, sub_queries=decompose(rw), hyde_doc=hyde_stub(rw))
