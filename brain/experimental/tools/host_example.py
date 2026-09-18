"""
brain/tools/host_example.py

Example: how a host (VS Code extension, desktop app, Colab with network)
wires real backends into the tool registry.

No hard tool-call limit — AgentLoop runs until the model answers or soft max_turns.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from brain.experimental.tools.builtins import register_builtin_tools
from brain.experimental.tools.registry import ToolRegistry


def build_registry_for_host(
    *,
    web_search: Optional[Callable[..., Any]] = None,
    web_open: Optional[Callable[..., Any]] = None,
    read_file: Optional[Callable[..., Any]] = None,
    write_file: Optional[Callable[..., Any]] = None,
    codebase_search: Optional[Callable[..., Any]] = None,
    run_terminal: Optional[Callable[..., Any]] = None,
    python_eval: Optional[Callable[..., Any]] = None,
    list_dir: Optional[Callable[..., Any]] = None,
    enable_stubs_for_missing: bool = True,
) -> ToolRegistry:
    """
    Pass only the handlers you implement. Example VS Code host:

        reg = build_registry_for_host(
            read_file=vscode_workspace_read,
            write_file=vscode_workspace_write,
            codebase_search=vscode_symbol_search,
            run_terminal=vscode_sandbox_terminal,
            web_search=my_serp_api,
            web_open=my_fetch_page,
        )
    """
    handlers = {
        k: v
        for k, v in {
            "web_search": web_search,
            "web_open": web_open,
            "read_file": read_file,
            "write_file": write_file,
            "codebase_search": codebase_search,
            "run_terminal": run_terminal,
            "python_eval": python_eval,
            "list_dir": list_dir,
        }.items()
        if v is not None
    }
    return register_builtin_tools(
        enable_stubs=enable_stubs_for_missing,
        custom_handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Optional: thin real web helpers when `requests` / host network is available
# ---------------------------------------------------------------------------

def try_requests_web_open(url: str, timeout: float = 15.0, max_chars: int = 12000) -> str:
    """Best-effort page text; host may replace with a better extractor."""
    try:
        import re
        import requests
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "BrainAgent/0.3"})
        r.raise_for_status()
        text = re.sub(r"<script[\s\S]*?</script>", " ", r.text, flags=re.I)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        return f"web_open failed: {e}"
