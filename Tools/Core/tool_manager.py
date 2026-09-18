"""High-level facade: load categories, list, run."""
from __future__ import annotations
from typing import Optional
from Tools.Core.tool_registry import ToolRegistry
from Tools.Core.tool_executor import ToolExecutor
from Tools.Core.tool_context import ToolContext
from Tools.Core.tool_router import ToolRouter
from Tools.Core.tool_loader import load_all_builtin_tools

class ToolManager:
    def __init__(self, context: Optional[ToolContext] = None, auto_load: bool = True):
        self.context = context or ToolContext()
        self.registry = ToolRegistry()
        if auto_load:
            load_all_builtin_tools(self.registry)
        self.executor = ToolExecutor(self.registry, self.context)
        self.router = ToolRouter(self.registry, self.executor)

    def list(self, category: Optional[str] = None):
        return self.registry.list_tools(category)

    def run(self, name: str, **kwargs):
        return self.executor.run(name, kwargs)

    def prompt_schemas(self) -> str:
        return self.registry.schemas_for_prompt()
