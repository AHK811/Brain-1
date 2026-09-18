
"""Hybrid dense + BM25 with Reciprocal Rank Fusion."""
from __future__ import annotations
import math, re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9_]+", re.I)

def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())

@dataclass
class Doc:
    id: str
    text: str

class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[Doc] = []
        self.doc_len: list[int] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def add(self, doc: Doc) -> None:
        toks = tokenize(doc.text)
        self.docs.append(doc)
        self.doc_len.append(len(toks))
        for t in set(toks):
            self.df[t] += 1
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = tokenize(query)
        N = len(self.docs) or 1
        scores = []
        for i, doc in enumerate(self.docs):
            tf = Counter(tokenize(doc.text))
            score = 0.0
            dl = self.doc_len[i] or 1
            for t in q:
                if t not in tf: continue
                n = self.df.get(t, 0) or 1
                idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
                f = tf[t]
                score += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            scores.append((doc.id, score, doc.text))
        scores.sort(key=lambda x: -x[1])
        return [(i, s) for i, s, _ in scores[:k]]

def rrf_fuse(rank_lists: list[list[str]], k: int = 5, rrf_k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for lst in rank_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]

class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25Index()
        self.dense: list[tuple[str, list[float], str]] = []  # id, vec, text

    def add(self, doc_id: str, text: str, vector: list[float] | None = None) -> None:
        self.bm25.add(Doc(doc_id, text))
        if vector is not None:
            self.dense.append((doc_id, vector, text))

    def search(self, query: str, query_vec: list[float] | None = None, k: int = 5) -> list[str]:
        sparse = [d for d, _ in self.bm25.search(query, k=k * 2)]
        dense_ids = []
        if query_vec and self.dense:
            def cos(a, b):
                return sum(x*y for x, y in zip(a, b))
            scored = [(cos(query_vec, v), i) for i, v, _ in self.dense]
            scored.sort(reverse=True)
            dense_ids = [i for _, i in scored[: k * 2]]
        return rrf_fuse([sparse, dense_ids] if dense_ids else [sparse], k=k)
