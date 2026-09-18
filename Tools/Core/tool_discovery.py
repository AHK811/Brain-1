"""List tools by capability tags."""
from __future__ import annotations
from Tools.Core.tool_registry import ToolRegistry

def discover(registry: ToolRegistry, query: str = "") -> list[str]:
    q = query.lower().strip()
    out = []
    for t in registry.list_tools():
        blob = f"{t.name} {t.description} {t.category}".lower()
        if not q or q in blob:
            out.append(t.name)
    return out
