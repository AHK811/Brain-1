
"""Post-retrieval: simple cross-encoder-style rerank (lexical overlap proxy)."""
from __future__ import annotations
import re

_TOKEN = re.compile(r"[a-z0-9_]+", re.I)

def score(query: str, doc: str) -> float:
    q, d = set(_TOKEN.findall(query.lower())), set(_TOKEN.findall(doc.lower()))
    if not q: return 0.0
    return len(q & d) / len(q)

def rerank(query: str, docs: list[tuple[str, str]], k: int = 5) -> list[tuple[str, str, float]]:
    scored = [(did, text, score(query, text)) for did, text in docs]
    scored.sort(key=lambda x: -x[2])
    return scored[:k]
