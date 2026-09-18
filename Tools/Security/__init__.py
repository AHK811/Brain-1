from __future__ import annotations
import re
from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult
_DANGEROUS = re.compile(r"(rm\s+-rf\s+/|mkfs.|dd\s+if=)", re.I)

def command_filter(ctx=None, command: str = "", **kw) -> ToolResult:
    if _DANGEROUS.search(command or ""):
        return ToolResult("command_filter", False, "blocked dangerous command", error_type="permission")
    return ToolResult("command_filter", True, "ok")

def register(registry):
    registry.register(make_tool("command_filter", "Check command safety", {"command":{"type":"string","required":True}}, command_filter, "security", None))
