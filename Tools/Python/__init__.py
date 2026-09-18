from __future__ import annotations
import traceback
from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult

def python_eval(ctx=None, code: str = "", **kw) -> ToolResult:
    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range,
        "str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict,
        "print": print, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip,
    }
    ns = {"__builtins__": safe_builtins}
    try:
        # try eval then exec
        try:
            val = eval(code, ns, ns)
            return ToolResult("python_eval", True, repr(val), raw=val)
        except SyntaxError:
            exec(code, ns, ns)
            return ToolResult("python_eval", True, "ok", raw=ns)
    except Exception as e:
        return ToolResult("python_eval", False, traceback.format_exc(), error_type="runtime")

def register(registry):
    registry.register(make_tool(
        "python_eval", "Evaluate Python in a restricted namespace",
        {"code": {"type": "string", "required": True}},
        python_eval, "python", "python:eval", parallel_safe=False,
    ))
