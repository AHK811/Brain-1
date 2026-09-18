"""Standard tool result envelope."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class ToolResult:
    name: str
    ok: bool
    content: str
    raw: Any = None
    error_type: Optional[str] = None
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name, "ok": self.ok, "content": self.content,
            "error_type": self.error_type, "latency_ms": self.latency_ms, "meta": self.meta,
        }
