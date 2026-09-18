"""Order tool calls (serial dependencies vs parallel)."""
from __future__ import annotations
from typing import Any
from Tools.Core.tool_engine import ToolEngine
from Tools.Core.tool_result import ToolResult

class ToolScheduler:
    def __init__(self, engine: ToolEngine):
        self.engine = engine

    def run_plan(self, steps: list[dict[str, Any]]) -> list[ToolResult]:
        # steps: {name, arguments, parallel_group?}
        results = []
        i = 0
        while i < len(steps):
            group = [steps[i]]
            g = steps[i].get("parallel_group")
            j = i + 1
            while g is not None and j < len(steps) and steps[j].get("parallel_group") == g:
                group.append(steps[j]); j += 1
            if len(group) > 1:
                results.extend(self.engine.run_many(group, parallel=True))
            else:
                results.append(self.engine.run(group[0]["name"], group[0].get("arguments")))
            i = j
        return results
