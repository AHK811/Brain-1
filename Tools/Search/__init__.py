from __future__ import annotations
from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult

def web_search(ctx=None, query: str = "", num_results: int = 5, **kw) -> ToolResult:
    # Host should override; placeholder uses no network assumption
    return ToolResult("web_search", True, f"[web_search stub] q={query!r} n={num_results}. Wire Serp/Brave in host.")

def local_search(ctx=None, query: str = "", path: str = ".", **kw) -> ToolResult:
    from Tools.Filesystem import search_files
    return search_files(ctx=ctx, query=query, path=path)

def register(registry):
    registry.register(make_tool("web_search", "Search the public web", {"query":{"type":"string","required":True},"num_results":{"type":"integer"}}, web_search, "search", "net:read"))
    registry.register(make_tool("local_search", "Search local workspace", {"query":{"type":"string","required":True},"path":{"type":"string"}}, local_search, "search", "fs:read"))
