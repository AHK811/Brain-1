
"""Semantic / parent-child / hierarchical chunking."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Chunk:
    id: str
    text: str
    parent_id: str | None = None
    level: int = 0

def sliding_chunks(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        if i + size >= len(text):
            break
        i += max(size - overlap, 1)
    return out

def parent_child_chunks(text: str, parent_size: int = 2000, child_size: int = 400, overlap: int = 50) -> list[Chunk]:
    parents = sliding_chunks(text, parent_size, overlap)
    chunks: list[Chunk] = []
    for pi, parent in enumerate(parents):
        pid = f"p{pi}"
        chunks.append(Chunk(id=pid, text=parent, parent_id=None, level=0))
        for ci, child in enumerate(sliding_chunks(parent, child_size, overlap)):
            chunks.append(Chunk(id=f"{pid}_c{ci}", text=child, parent_id=pid, level=1))
    return chunks

def semantic_chunks(text: str, max_chars: int = 800) -> list[str]:
    # paragraph-aware
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf: out.append(buf)
            buf = p
    if buf: out.append(buf)
    return out or ([text[:max_chars]] if text else [])
