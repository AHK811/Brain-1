from __future__ import annotations
import ast, re
from pathlib import Path
from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult

def code_parse(ctx=None, path: str = "", **kw) -> ToolResult:
    p = Path(path)
    src = p.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        return ToolResult("code_parse", True, f"classes={classes}\\nfuncs={funcs}")
    except Exception as e:
        return ToolResult("code_parse", False, str(e), error_type="runtime")

def code_complexity(ctx=None, path: str = "", **kw) -> ToolResult:
    p = Path(path)
    src = p.read_text(encoding="utf-8", errors="ignore")
    # crude: count branches
    branches = len(re.findall(r"\\b(if|for|while|elif|except|with|case)\\b", src))
    lines = len(src.splitlines())
    return ToolResult("code_complexity", True, f"lines={lines} branch_keywords={branches}")

def register(registry):
    registry.register(make_tool("code_parse", "Parse Python AST symbols", {"path":{"type":"string","required":True}}, code_parse, "code", "fs:read"))
    registry.register(make_tool("code_complexity", "Rough complexity metrics", {"path":{"type":"string","required":True}}, code_complexity, "code", "fs:read"))
