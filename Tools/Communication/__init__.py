from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult
def webhook_post(ctx=None, url: str = "", body: str = "", **kw):
    try:
        import urllib.request
        req = urllib.request.Request(url, data=body.encode(), method="POST", headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return ToolResult("webhook_post", True, r.read(5000).decode("utf-8", errors="ignore"))
    except Exception as e:
        return ToolResult("webhook_post", False, str(e), error_type="runtime")
def register(registry):
    registry.register(make_tool("webhook_post", "POST JSON webhook", {"url":{"type":"string","required":True},"body":{"type":"string","required":True}}, webhook_post, "communication", "net:write", parallel_safe=False))
