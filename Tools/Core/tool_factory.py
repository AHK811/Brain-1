"""Build ToolSpec quickly."""
from __future__ import annotations
from typing import Any, Callable, Optional
from Tools.Core.tool_registry import ToolSpec

def make_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    handler: Callable,
    category: str = "general",
    permission: Optional[str] = None,
    parallel_safe: bool = True,
) -> ToolSpec:
    return ToolSpec(
        name=name, description=description, parameters=parameters,
        handler=handler, category=category, required_permission=permission,
        parallel_safe=parallel_safe,
    )
