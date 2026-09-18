from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
class AgentInterface(ABC):
    @abstractmethod
    def run(self, goal: str) -> dict[str, Any]: ...
