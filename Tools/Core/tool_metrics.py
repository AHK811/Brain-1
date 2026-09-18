"""Simple counters for success/fail."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class ToolMetrics:
    ok: int = 0
    fail: int = 0
    by_tool: dict = field(default_factory=dict)

    def record(self, name: str, ok: bool) -> None:
        if ok:
            self.ok += 1
        else:
            self.fail += 1
        slot = self.by_tool.setdefault(name, {"ok": 0, "fail": 0})
        slot["ok" if ok else "fail"] += 1
