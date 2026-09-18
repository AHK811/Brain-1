from __future__ import annotations
import os, subprocess, shlex
from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult

def run_terminal(ctx=None, command: str = "", cwd: str | None = None, timeout: int = 60, **kw) -> ToolResult:
    work = cwd or str(getattr(ctx, "cwd", ".") if ctx else ".")
    try:
        proc = subprocess.run(
            command, shell=True, cwd=work, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, **(getattr(ctx, "env", {}) or {})},
        )
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        return ToolResult("run_terminal", proc.returncode == 0, out.strip() or f"(exit {proc.returncode})",
                          meta={"returncode": proc.returncode})
    except subprocess.TimeoutExpired:
        return ToolResult("run_terminal", False, f"timeout after {timeout}s", error_type="runtime")
    except Exception as e:
        return ToolResult("run_terminal", False, str(e), error_type="runtime")

def register(registry):
    registry.register(make_tool(
        "run_terminal", "Run shell command",
        {"command": {"type": "string", "required": True}, "cwd": {"type": "string"}, "timeout": {"type": "integer"}},
        run_terminal, "terminal", "terminal:run", parallel_safe=False,
    ))
