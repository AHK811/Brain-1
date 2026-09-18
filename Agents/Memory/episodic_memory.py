from __future__ import annotations
from dataclasses import dataclass, field
from time import time
@dataclass
class Episode:
    key: str
    content: str
    ts: float = field(default_factory=time)
@dataclass
class EpisodicMemory:
    items: list[Episode] = field(default_factory=list)
    max_items: int = 500
    def add(self, key: str, content: str) -> None:
        self.items.append(Episode(key=key, content=content))
        self.items = self.items[-self.max_items:]
    def search(self, query: str, k: int = 5) -> list[Episode]:
        q = query.lower().split()
        scored = [(sum(1 for w in q if w in (e.key+" "+e.content).lower()), e) for e in self.items]
        scored = [x for x in scored if x[0]]
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]
