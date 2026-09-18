"""Execute a single tool call with validation, permissions, timing."""
from __future__ import annotations
import time
from typing import Any
from Tools.Core.tool_registry import ToolRegistry, ToolSpec
from Tools.Core.tool_result import ToolResult
from Tools.Core.tool_context import ToolContext
from Tools.Core.tool_validator import validate_args

class ToolExecutor:
    def __init__(self, registry: ToolRegistry, context: ToolContext | None = None):
        self.registry = registry
        self.context = context or ToolContext()

    def run(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        arguments = arguments or {}
        t0 = time.perf_counter()
        spec = self.registry.get(name)
        if spec is None:
            known = ", ".join(sorted(self.registry.tools)) or "(none)"
            return ToolResult(name, False, f"Unknown tool: {name}. Known: {known}", error_type="unknown_tool")
        if spec.required_permission and not self.context.has(spec.required_permission):
            return ToolResult(name, False, f"Permission denied: {spec.required_permission}", error_type="permission")
        err = validate_args(spec.parameters, arguments)
        if err:
            return ToolResult(name, False, f"Validation: {err}", error_type="validation")
        if spec.handler is None:
            return ToolResult(name, False, f"No handler for {name}", error_type="no_handler")
        try:
            out = spec.handler(ctx=self.context, **arguments)
            if isinstance(out, ToolResult):
                out.latency_ms = (time.perf_counter() - t0) * 1000
                return out
            return ToolResult(name, True, str(out), raw=out, latency_ms=(time.perf_counter() - t0) * 1000)
        except TypeError:
            # handlers that don't take ctx=
            try:
                out = spec.handler(**arguments)
                if isinstance(out, ToolResult):
                    out.latency_ms = (time.perf_counter() - t0) * 1000
                    return out
                return ToolResult(name, True, str(out), raw=out, latency_ms=(time.perf_counter() - t0) * 1000)
            except Exception as e:
                return ToolResult(name, False, f"Runtime: {e}", error_type="runtime",
                                  latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as e:
            return ToolResult(name, False, f"Runtime: {e}", error_type="runtime",
                              latency_ms=(time.perf_counter() - t0) * 1000)
