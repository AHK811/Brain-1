"""Execution context passed into every tool call."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

@dataclass
class ToolContext:
    cwd: Path = field(default_factory=lambda: Path.cwd())
    workspace_root: Optional[Path] = None
    session_id: str = "default"
    user_id: Optional[str] = None
    permissions: set[str] = field(default_factory=lambda: {"fs:read", "fs:write", "fs:delete", "terminal:run", "python:eval", "net:read", "net:write", "git:write", "db:write"})
    env: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def has(self, perm: str) -> bool:
        return perm in self.permissions or "*" in self.permissions
