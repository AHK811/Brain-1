"""Central registry of tool specs + handlers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from Tools.Core.tool_result import ToolResult
from Tools.Core.tool_validator import validate_args

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Optional[Callable[..., Any]] = None
    category: str = "general"
    parallel_safe: bool = True
    required_permission: Optional[str] = None
    timeout_s: float = 60.0

@dataclass
class ToolRegistry:
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self.tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> list[ToolSpec]:
        vals = list(self.tools.values())
        if category:
            vals = [t for t in vals if t.category == category]
        return vals

    def schemas_for_prompt(self) -> str:
        lines = []
        for t in sorted(self.tools.values(), key=lambda x: (x.category, x.name)):
            params = ", ".join(
                f"{k}{'*' if v.get('required') else ''}: {v.get('type','any')}"
                for k, v in t.parameters.items()
            )
            lines.append(f"- [{t.category}] {t.name}({params}): {t.description}")
        return "\n".join(lines) if lines else "(no tools)"
