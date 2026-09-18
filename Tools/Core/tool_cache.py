"""Simple in-memory result cache for idempotent tools."""
from __future__ import annotations
import hashlib, json, time
from typing import Any, Optional
from Tools.Core.tool_result import ToolResult

class ToolCache:
    def __init__(self, ttl_s: float = 300.0):
        self.ttl_s = ttl_s
        self._store: dict[str, tuple[float, ToolResult]] = {}

    def _key(self, name: str, arguments: dict) -> str:
        raw = name + json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, name: str, arguments: dict) -> Optional[ToolResult]:
        k = self._key(name, arguments)
        item = self._store.get(k)
        if not item:
            return None
        ts, res = item
        if time.time() - ts > self.ttl_s:
            del self._store[k]
            return None
        return res

    def set(self, name: str, arguments: dict, result: ToolResult) -> None:
        self._store[self._key(name, arguments)] = (time.time(), result)
