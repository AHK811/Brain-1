from __future__ import annotations
from abc import ABC, abstractmethod
class MemoryInterface(ABC):
    @abstractmethod
    def remember(self, key: str, content: str, **meta) -> None: ...
    @abstractmethod
    def recall(self, query: str, k: int = 5) -> list[str]: ...
