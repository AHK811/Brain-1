from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
class ToolInterface(ABC):
    @abstractmethod
    def run(self, name: str, arguments: dict[str, Any] | None = None) -> Any: ...
    @abstractmethod
    def list_tools(self) -> list[str]: ...
