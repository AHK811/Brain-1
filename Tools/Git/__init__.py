from __future__ import annotations
import subprocess
from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult

def _git(args, cwd="."):
    try:
        p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
        out = (p.stdout or p.stderr or "").strip()
        return ToolResult("git_" + args[0], p.returncode == 0, out or f"(exit {p.returncode})")
    except Exception as e:
        return ToolResult("git", False, str(e), error_type="runtime")

def git_status(ctx=None, path: str = ".", **kw):
    return _git(["status", "--short"], cwd=path)

def git_diff(ctx=None, path: str = ".", **kw):
    return _git(["diff"], cwd=path)

def git_log(ctx=None, path: str = ".", n: int = 10, **kw):
    return _git(["log", f"-{n}", "--oneline"], cwd=path)

def register(registry):
    registry.register(make_tool("git_status", "git status", {"path":{"type":"string"}}, git_status, "git", "fs:read"))
    registry.register(make_tool("git_diff", "git diff", {"path":{"type":"string"}}, git_diff, "git", "fs:read"))
    registry.register(make_tool("git_log", "git log", {"path":{"type":"string"}, "n":{"type":"integer"}}, git_log, "git", "fs:read"))
