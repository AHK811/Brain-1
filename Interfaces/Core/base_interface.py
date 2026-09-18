from __future__ import annotations
from abc import ABC
from typing import Any
class BaseInterface(ABC):
    name: str = "base"
    version: str = "0.1"
    def info(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}
