"""
In-memory RAG index.

Default embedding: hashed bag-of-words (no extra deps).
Optional: pass embed_fn(text) -> np.ndarray from sentence-transformers etc.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class RAGChunk:
    id: str
    text: str
    source: str = ""
    meta: dict = field(default_factory=dict)


_TOKEN = re.compile(r"[a-z0-9_]+", re.I)


def simple_hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Fast dependency-free embedding for small corpora."""
    vec = np.zeros(dim, dtype=np.float32)
    toks = _TOKEN.findall(text.lower())
    if not toks:
        return vec
    for t in toks:
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = np.linalg.norm(vec)
    if n > 0:
        vec /= n
    return vec


@dataclass
class RAGIndex:
    chunks: list[RAGChunk] = field(default_factory=list)
    vectors: Optional[np.ndarray] = None  # (N, D)
    embed_fn: Callable[[str], np.ndarray] = simple_hash_embed
    dim: int = 256

    def add(self, text: str, source: str = "", chunk_id: Optional[str] = None, **meta) -> None:
        cid = chunk_id or f"c{len(self.chunks)}"
        self.chunks.append(RAGChunk(id=cid, text=text, source=source, meta=meta))
        self.vectors = None  # invalidate

    def add_file(self, path: str | Path, *, chunk_chars: int = 800, overlap: int = 100) -> int:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.add_text(text, source=str(path), chunk_chars=chunk_chars, overlap=overlap)

    def add_text(self, text: str, source: str = "", *, chunk_chars: int = 800, overlap: int = 100) -> int:
        n_before = len(self.chunks)
        if chunk_chars <= 0:
            self.add(text, source=source)
            return 1
        i = 0
        while i < len(text):
            piece = text[i : i + chunk_chars].strip()
            if piece:
                self.add(piece, source=source, chunk_id=f"{source}:{i}")
            if i + chunk_chars >= len(text):
                break
            i += max(chunk_chars - overlap, 1)
        self.vectors = None
        return len(self.chunks) - n_before

    def build(self) -> None:
        if not self.chunks:
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)
            return
        mats = [self.embed_fn(c.text) for c in self.chunks]
        self.vectors = np.stack(mats, axis=0).astype(np.float32)

    def search(self, query: str, top_k: int = 5) -> list[tuple[RAGChunk, float]]:
        if self.vectors is None:
            self.build()
        if self.vectors is None or len(self.chunks) == 0:
            return []
        q = self.embed_fn(query).astype(np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        scores = self.vectors @ q
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.chunks[i], float(scores[i])) for i in idx]
