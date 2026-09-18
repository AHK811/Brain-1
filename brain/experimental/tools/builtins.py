"""
brain/tools/builtins.py

Richer tool set with accurate schemas. Handlers optional — host injects real ones.
"""

from __future__ import annotations

from typing import Any, Optional

from brain.experimental.tools.registry import ToolRegistry, ToolSpec, ToolResult


def _stub(msg: str) -> str:
    return msg


def register_builtin_tools(
    registry: Optional[ToolRegistry] = None,
    *,
    enable_stubs: bool = True,
    custom_handlers: Optional[dict[str, Any]] = None,
) -> ToolRegistry:
    reg = registry or ToolRegistry()
    H = custom_handlers or {}

    def handler(name: str, stub_fn):
        if name in H:
            return H[name]
        return stub_fn if enable_stubs else None

    specs = [
        ToolSpec(
            name="web_search",
            description="Search the public web. Returns ranked titles, URLs, snippets.",
            parameters={
                "query": {"type": "string", "description": "Search query", "required": True},
                "num_results": {"type": "integer", "description": "1-10 results", "required": False},
            },
            handler=handler(
                "web_search",
                lambda query, num_results=5: _stub(
                    f"[web_search] q={query!r} n={num_results}. Wire Serp/Brave/Bing in host."
                ),
            ),
            parallel_safe=True,
        ),
        ToolSpec(
            name="web_open",
            description="Fetch a URL and return main text (HTML stripped).",
            parameters={
                "url": {"type": "string", "description": "https URL", "required": True},
                "max_chars": {"type": "integer", "description": "Truncate length", "required": False},
            },
            handler=handler(
                "web_open",
                lambda url, max_chars=12000: _stub(f"[web_open] {url} max={max_chars}. Wire fetcher in host."),
            ),
            parallel_safe=True,
        ),
        ToolSpec(
            name="codebase_search",
            description="Semantic/text search over the open workspace (VS Code / IDE).",
            parameters={
                "query": {"type": "string", "required": True},
                "path": {"type": "string", "description": "Subfolder", "required": False},
                "max_hits": {"type": "integer", "required": False},
            },
            handler=handler(
                "codebase_search",
                lambda query, path=".", max_hits=10: _stub(
                    f"[codebase_search] q={query!r} path={path} hits={max_hits}"
                ),
            ),
            parallel_safe=True,
        ),
        ToolSpec(
            name="read_file",
            description="Read file contents from workspace or allowed paths.",
            parameters={
                "path": {"type": "string", "required": True},
                "offset": {"type": "integer", "description": "Start line 1-based", "required": False},
                "limit": {"type": "integer", "description": "Max lines", "required": False},
            },
            handler=handler(
                "read_file",
                lambda path, offset=1, limit=200: _stub(f"[read_file] {path}:{offset}+{limit}"),
            ),
            parallel_safe=True,
        ),
        ToolSpec(
            name="write_file",
            description="Create or overwrite a file (host should confirm destructive writes).",
            parameters={
                "path": {"type": "string", "required": True},
                "content": {"type": "string", "required": True},
            },
            handler=handler(
                "write_file",
                lambda path, content: _stub(f"[write_file] {path} ({len(content)} chars)"),
            ),
            parallel_safe=False,
        ),
        ToolSpec(
            name="list_dir",
            description="List directory entries.",
            parameters={"path": {"type": "string", "required": False}},
            handler=handler("list_dir", lambda path=".": _stub(f"[list_dir] {path}")),
            parallel_safe=True,
        ),
        ToolSpec(
            name="run_terminal",
            description="Run a shell command in a sandbox or user-approved terminal.",
            parameters={
                "command": {"type": "string", "required": True},
                "cwd": {"type": "string", "required": False},
            },
            handler=handler(
                "run_terminal",
                lambda command, cwd=".": _stub(f"[run_terminal] cwd={cwd} cmd={command!r}"),
            ),
            parallel_safe=False,
            timeout_s=120.0,
        ),
        ToolSpec(
            name="python_eval",
            description="Execute Python in a sandbox; return stdout/stderr/result.",
            parameters={"code": {"type": "string", "required": True}},
            handler=handler(
                "python_eval",
                lambda code: _stub(f"[python_eval] ({len(code)} chars)"),
            ),
            parallel_safe=False,
            timeout_s=60.0,
        ),
        ToolSpec(
            name="calculator",
            description="Evaluate a numeric math expression safely (no side effects).",
            parameters={"expression": {"type": "string", "required": True}},
            handler=handler("calculator", _safe_calc),
            parallel_safe=True,
        ),
        ToolSpec(
            name="rag_query",
            description="Query the RAG index (local docs/embeddings) and return top passages.",
            parameters={
                "query": {"type": "string", "required": True},
                "top_k": {"type": "integer", "required": False},
            },
            handler=handler(
                "rag_query",
                lambda query, top_k=5: _stub(
                    f"[rag_query] q={query!r} k={top_k}. Attach brain.experimental.rag index in host."
                ),
            ),
            parallel_safe=True,
        ),
    ]
    for s in specs:
        reg.register(s)
    return reg


def _safe_calc(expression: str) -> str:
    """Very small safe evaluator for + - * / ** % and parentheses."""
    import ast
    import operator as op

    allowed = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.operand))
        raise ValueError("disallowed expression")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        val = _eval(tree)
        return str(val)
    except Exception as e:
        return f"calculator error: {e}"
