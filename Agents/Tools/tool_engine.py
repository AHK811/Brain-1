"""Bridge to top-level Tools package."""
from __future__ import annotations
from typing import Any

def get_tool_engine(context=None):
    from Tools import ToolEngine, ToolContext
    return ToolEngine(context or ToolContext())
