from __future__ import annotations
from dataclasses import dataclass, field
@dataclass
class PolicyEngine:
    denied_tools: set[str] = field(default_factory=set)
    def allow(self, tool: str) -> bool:
        return tool not in self.denied_tools
