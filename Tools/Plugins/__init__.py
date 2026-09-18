from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult
def list_plugins(ctx=None, **kw):
    return ToolResult("list_plugins", True, "core filesystem terminal python search git documents media security")
def register(registry):
    registry.register(make_tool("list_plugins", "List plugin groups", {}, list_plugins, "plugins", None))
