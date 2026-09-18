"""Route natural-language intent or structured calls to tools."""
from __future__ import annotations
import json, re
from typing import Any
from Tools.Core.tool_registry import ToolRegistry
from Tools.Core.tool_executor import ToolExecutor
from Tools.Core.tool_result import ToolResult

_BLOCK = re.compile(
    r"<\|tool_call\|>\s*(\{.*?\})\s*(?:<\|end_tool_call\|>|(?=<\|tool_call\|>)|$)",
    re.DOTALL,
)

def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for m in _BLOCK.finditer(text or ""):
        raw = m.group(1).strip()
        raw = re.sub(r",\s*}", "}", raw)
        try:
            obj = json.loads(raw)
        except Exception:
            try:
                obj = json.loads(raw.replace("'", '"'))
            except Exception:
                continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments") if "arguments" in obj else obj.get("args")
        if args is None:
            args = {k: v for k, v in obj.items() if k != "name"}
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": str(obj["name"]), "arguments": args})
    return calls

class ToolRouter:
    def __init__(self, registry: ToolRegistry, executor: ToolExecutor | None = None):
        self.registry = registry
        self.executor = executor or ToolExecutor(registry)

    def route_text(self, text: str) -> list[ToolResult]:
        return [self.executor.run(c["name"], c.get("arguments") or {}) for c in parse_tool_calls(text)]

    def route_call(self, name: str, arguments: dict | None = None) -> ToolResult:
        return self.executor.run(name, arguments or {})
