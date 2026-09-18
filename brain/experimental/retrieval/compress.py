
"""Context compression (extractive; LLMLingua-style placeholder)."""
from __future__ import annotations
import re

def compress(text: str, max_chars: int = 1500, query: str = "") -> str:
    if len(text) <= max_chars:
        return text
    if not query:
        return text[:max_chars] + "\n[compressed]"
    # keep sentences overlapping query terms
    terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
    sents = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    n = 0
    for s in sents:
        st = set(re.findall(r"[a-z0-9_]+", s.lower()))
        if terms & st or not kept:
            if n + len(s) > max_chars: break
            kept.append(s); n += len(s)
    return " ".join(kept) if kept else text[:max_chars]
