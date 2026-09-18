from __future__ import annotations
from typing import Any
class InterfaceRegistry:
    def __init__(self):
        self._items: dict[str, Any] = {}
    def register(self, name: str, obj: Any) -> None:
        self._items[name] = obj
    def get(self, name: str):
        return self._items.get(name)
    def list(self):
        return sorted(self._items)
