from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
class ModelInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str: ...
    @abstractmethod
    def encode(self, text: str) -> list[int]: ...
