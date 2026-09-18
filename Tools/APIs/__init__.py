from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult
def http_get(ctx=None, url: str = "", **kw):
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=20) as r:
            body = r.read(50000).decode("utf-8", errors="ignore")
        return ToolResult("http_get", True, body)
    except Exception as e:
        return ToolResult("http_get", False, str(e), error_type="runtime")
def register(registry):
    registry.register(make_tool("http_get", "HTTP GET", {"url":{"type":"string","required":True}}, http_get, "apis", "net:read"))
