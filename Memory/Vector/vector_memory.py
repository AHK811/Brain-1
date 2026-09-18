from __future__ import annotations
import hashlib, re, math
from dataclasses import dataclass, field
from typing import Callable
_TOKEN = re.compile(r"[a-z0-9_]+", re.I)
def hash_embed(text: str, dim: int = 256):
    vec = [0.0]*dim
    for t in _TOKEN.findall(text.lower()):
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x/n for x in vec]
def cosine(a, b):
    return sum(x*y for x,y in zip(a,b))
@dataclass
class VectorItem:
    id: str
    text: str
    vector: list
    meta: dict = field(default_factory=dict)
@dataclass
class VectorMemory:
    dim: int = 256
    items: list = field(default_factory=list)
    embed_fn: Callable | None = None
    def add(self, id: str, text: str, **meta):
        fn = self.embed_fn or (lambda t: hash_embed(t, self.dim))
        self.items.append(VectorItem(id=id, text=text, vector=fn(text), meta=meta))
    def search(self, query: str, k: int = 5):
        fn = self.embed_fn or (lambda t: hash_embed(t, self.dim))
        q = fn(query)
        scored = [(cosine(q, it.vector), it) for it in self.items]
        scored.sort(key=lambda x: -x[0])
        return [(it, s) for s, it in scored[:k]]
