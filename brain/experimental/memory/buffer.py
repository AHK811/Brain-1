"""
Session memory: store facts, tool results, and user prefs for multi-turn agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Optional


@dataclass
class MemoryItem:
    key: str
    content: str
    kind: str = "note"  # note | tool | user | fact
    score: float = 1.0
    ts: float = field(default_factory=time)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryBuffer:
    items: list[MemoryItem] = field(default_factory=list)
    max_items: int = 200

    def add(self, key: str, content: str, kind: str = "note", **meta) -> None:
        self.items.append(MemoryItem(key=key, content=content, kind=kind, meta=meta))
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]

    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        q = query.lower().split()
        scored = []
        for it in self.items:
            text = (it.key + " " + it.content).lower()
            hit = sum(1 for w in q if w in text)
            if hit:
                scored.append((hit * it.score, it))
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:top_k]]

    def as_context(self, query: str = "", top_k: int = 8) -> str:
        items = self.search(query, top_k=top_k) if query else self.items[-top_k:]
        if not items:
            return ""
        lines = [f"- [{it.kind}] {it.key}: {it.content}" for it in items]
        return "Memory:\n" + "\n".join(lines)
