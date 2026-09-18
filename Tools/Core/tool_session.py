"""Session-scoped tool state (cwd, env, history)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from Tools.Core.tool_context import ToolContext
from Tools.Core.tool_result import ToolResult

@dataclass
class ToolSession:
    context: ToolContext = field(default_factory=ToolContext)
    history: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, arguments: dict, result: ToolResult) -> None:
        self.history.append({"name": name, "arguments": arguments, "ok": result.ok, "content": result.content[:500]})
