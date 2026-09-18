from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Any
@dataclass
class WorkingMemory:
    max_items: int = 64
    items: deque = field(default_factory=deque)
    def add(self, role: str, content: str, **meta: Any) -> None:
        self.items.append({"role": role, "content": content, "meta": meta})
        while len(self.items) > self.max_items:
            self.items.popleft()
    def as_text(self, max_chars: int = 8000) -> str:
        parts, n = [], 0
        for it in self.items:
            line = f"{it['role']}: {it['content']}\n"
            if n + len(line) > max_chars: break
            parts.append(line); n += len(line)
        return "".join(parts)
    def clear(self) -> None:
        self.items.clear()
