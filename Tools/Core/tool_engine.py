"""Top-level tool engine used by agents."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from Tools.Core.tool_manager import ToolManager
from Tools.Core.tool_result import ToolResult
from Tools.Core.tool_context import ToolContext

class ToolEngine:
    def __init__(self, context: ToolContext | None = None):
        self.manager = ToolManager(context=context)

    def run(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        return self.manager.executor.run(name, arguments or {})

    def run_many(self, calls: list[dict[str, Any]], parallel: bool = True) -> list[ToolResult]:
        if not parallel or len(calls) <= 1:
            return [self.run(c["name"], c.get("arguments") or {}) for c in calls]
        results: list[ToolResult | None] = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=min(8, len(calls))) as ex:
            futs = {ex.submit(self.run, c["name"], c.get("arguments") or {}): i for i, c in enumerate(calls)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        return [r for r in results if r is not None]

    def schemas(self) -> str:
        return self.manager.prompt_schemas()
