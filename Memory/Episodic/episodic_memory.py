from __future__ import annotations
from dataclasses import dataclass, field
from time import time
from typing import Any
@dataclass
class Episode:
    key: str
    content: str
    ts: float = field(default_factory=time)
    meta: dict = field(default_factory=dict)
@dataclass
class EpisodicMemory:
    items: list = field(default_factory=list)
    max_items: int = 2000
    def add(self, key: str, content: str, **meta: Any) -> None:
        self.items.append(Episode(key=key, content=content, meta=meta))
        self.items = self.items[-self.max_items:]
    def search(self, query: str, k: int = 5):
        q = query.lower().split()
        scored = []
        for ep in self.items:
            blob = (ep.key + " " + ep.content).lower()
            score = sum(1 for w in q if w in blob)
            if score: scored.append((score, ep))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]
