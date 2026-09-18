"""Hooks: before/after tool call."""
from __future__ import annotations
from typing import Callable, Any
from Tools.Core.tool_result import ToolResult

class LifecycleHooks:
    def __init__(self):
        self.before: list[Callable] = []
        self.after: list[Callable] = []

    def run_before(self, name: str, arguments: dict) -> None:
        for h in self.before:
            h(name, arguments)

    def run_after(self, name: str, arguments: dict, result: ToolResult) -> None:
        for h in self.after:
            h(name, arguments, result)
