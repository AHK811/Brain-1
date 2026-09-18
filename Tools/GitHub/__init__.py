from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult
def github_api(ctx=None, path: str = "", **kw):
    return ToolResult("github_api", True, f"[stub] GET https://api.github.com{path}")
def register(registry):
    registry.register(make_tool("github_api", "GitHub API helper", {"path":{"type":"string","required":True}}, github_api, "github", "net:read"))
