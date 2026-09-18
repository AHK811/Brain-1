from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult
def web_open(ctx=None, url: str = "", max_chars: int = 12000, **kw):
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=20) as r:
            body = r.read(max_chars * 2).decode("utf-8", errors="ignore")
        return ToolResult("web_open", True, body[:max_chars])
    except Exception as e:
        return ToolResult("web_open", False, str(e), error_type="runtime")
def register(registry):
    registry.register(make_tool("web_open", "Fetch URL text", {"url":{"type":"string","required":True},"max_chars":{"type":"integer"}}, web_open, "browser", "net:read"))
