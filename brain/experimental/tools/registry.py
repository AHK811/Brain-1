"""
brain/tools/registry.py

Advanced tool registry: schema validation, robust parsing, parallel-ready
execution, error recovery hints, and unlimited calls when handlers exist.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Optional[Callable[..., Any]] = None
    # Optional: max seconds host should allow (advisory)
    timeout_s: float = 60.0
    # If True, safe to run in parallel with other tools in the same turn
    parallel_safe: bool = True


@dataclass
class ToolResult:
    name: str
    ok: bool
    content: str
    raw: Any = None
    error_type: Optional[str] = None  # validation | unknown_tool | no_handler | runtime


@dataclass
class ToolRegistry:
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self.tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return list(self.tools.values())

    def schemas_for_prompt(self) -> str:
        lines = []
        for t in self.tools.values():
            req = []
            opt = []
            for k, v in t.parameters.items():
                desc = v.get("description", v.get("type", "any"))
                slot = f"{k}: {v.get('type', 'any')} — {desc}"
                if v.get("required", False):
                    req.append(slot)
                else:
                    opt.append(slot)
            params = "; ".join(req + [f"(opt) {x}" for x in opt])
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines) if lines else "(no tools registered)"

    def validate_args(self, spec: ToolSpec, arguments: dict[str, Any]) -> Optional[str]:
        """Return error message if invalid, else None."""
        if not isinstance(arguments, dict):
            return "arguments must be a JSON object"
        for key, meta in spec.parameters.items():
            if meta.get("required") and key not in arguments:
                return f"missing required argument: {key}"
            if key in arguments:
                val = arguments[key]
                typ = meta.get("type")
                if typ == "string" and not isinstance(val, str):
                    return f"{key} must be string"
                if typ == "integer" and not isinstance(val, int):
                    return f"{key} must be integer"
                if typ == "number" and not isinstance(val, (int, float)):
                    return f"{key} must be number"
                if typ == "boolean" and not isinstance(val, bool):
                    return f"{key} must be boolean"
                if typ == "array" and not isinstance(val, list):
                    return f"{key} must be array"
                if "enum" in meta and val not in meta["enum"]:
                    return f"{key} must be one of {meta['enum']}"
        return None

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        spec = self.tools.get(name)
        if spec is None:
            known = ", ".join(sorted(self.tools)) or "(none)"
            return ToolResult(
                name=name, ok=False,
                content=f"Unknown tool: {name}. Known: {known}",
                error_type="unknown_tool",
            )
        err = self.validate_args(spec, arguments or {})
        if err:
            return ToolResult(name=name, ok=False, content=f"Validation: {err}", error_type="validation")
        if spec.handler is None:
            return ToolResult(
                name=name, ok=False, error_type="no_handler",
                content=(
                    f"Tool '{name}' has no handler in this process. "
                    "Wire a host backend (VS Code, browser, CLI, sandbox)."
                ),
            )
        try:
            out = spec.handler(**(arguments or {}))
            if isinstance(out, ToolResult):
                return out
            return ToolResult(name=name, ok=True, content=str(out), raw=out)
        except TypeError as e:
            return ToolResult(name=name, ok=False, content=f"Bad arguments: {e}", error_type="validation")
        except Exception as e:
            return ToolResult(name=name, ok=False, content=f"Runtime error: {e}", error_type="runtime")

    def run_many(
        self,
        calls: list[dict[str, Any]],
        *,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> list[ToolResult]:
        """Execute multiple tool calls; parallel when all are parallel_safe."""
        if not calls:
            return []
        if not parallel or len(calls) == 1:
            return [self.run(c["name"], c.get("arguments") or {}) for c in calls]

        can_parallel = all(
            (self.tools.get(c["name"]) and self.tools[c["name"]].parallel_safe)
            for c in calls
            if c.get("name") in self.tools
        )
        if not can_parallel:
            return [self.run(c["name"], c.get("arguments") or {}) for c in calls]

        results: list[Optional[ToolResult]] = [None] * len(calls)

        def _job(i: int, c: dict) -> tuple[int, ToolResult]:
            return i, self.run(c["name"], c.get("arguments") or {})

        with ThreadPoolExecutor(max_workers=min(max_workers, len(calls))) as ex:
            futs = [ex.submit(_job, i, c) for i, c in enumerate(calls)]
            for fut in as_completed(futs):
                i, res = fut.result()
                results[i] = res
        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Robust parsing of model tool calls
# Supports:
#   <|tool_call|>\n{json}\n<|end_tool_call|>
#   multiple blocks
#   slightly messy JSON (trailing commas stripped)
# ---------------------------------------------------------------------------

_TOOL_BLOCK = re.compile(
    r"<\|tool_call\|>\s*(\{.*?\})\s*(?:<\|end_tool_call\|>|(?=<\|tool_call\|>)|$)",
    re.DOTALL,
)


def _clean_json(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for m in _TOOL_BLOCK.finditer(text or ""):
        raw = _clean_json(m.group(1))
        obj = None
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            try:
                obj = json.loads(raw.replace("'", '"'))
            except Exception:
                continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments") if "arguments" in obj else obj.get("args")
        if args is None:
            # allow flat {name, query, ...} style
            args = {k: v for k, v in obj.items() if k != "name"}
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": str(obj["name"]).strip(), "arguments": args})
    return calls


def format_tool_result(result: ToolResult) -> str:
    status = "ok" if result.ok else "error"
    extra = f" type={result.error_type}" if result.error_type else ""
    return f"<|tool_result|>\n[{status}{extra}] {result.name}: {result.content}\n"


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<|tool_call|>\n{payload}\n<|end_tool_call|>"
