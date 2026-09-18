from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from Memory.Working.working_memory import WorkingMemory
from Memory.Episodic.episodic_memory import EpisodicMemory
from Memory.Vector.vector_memory import VectorMemory
@dataclass
class MemoryEngine:
    working: WorkingMemory = field(default_factory=WorkingMemory)
    episodic: EpisodicMemory = field(default_factory=EpisodicMemory)
    vector: VectorMemory = field(default_factory=VectorMemory)
    def remember(self, key: str, content: str, *, kind: str = "episode", **meta: Any) -> None:
        if kind == "working":
            self.working.add("memory", content, key=key, **meta)
        else:
            self.episodic.add(key, content, **meta)
            self.vector.add(key, content, **meta)
    def recall(self, query: str, k: int = 5) -> list[str]:
        hits = [ep.content for ep in self.episodic.search(query, k=k)]
        hits += [f"[{s:.3f}] {it.text}" for it, s in self.vector.search(query, k=k)]
        seen, out = set(), []
        for h in hits:
            if h not in seen:
                seen.add(h); out.append(h)
        return out[:k]
    def context(self, query: str = "", k: int = 5) -> str:
        parts = [self.working.as_text()]
        if query:
            parts.append("Recalled:\n" + "\n".join(f"- {h}" for h in self.recall(query, k=k)))
        return "\n".join(p for p in parts if p)
