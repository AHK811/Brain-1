from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
@dataclass
class DatasetMeta:
    name: str
    path: str
    kind: str = "text"
    n_records: int = 0
    meta: dict = field(default_factory=dict)
@dataclass
class DatasetRegistry:
    items: dict[str, DatasetMeta] = field(default_factory=dict)
    def register(self, meta: DatasetMeta) -> None:
        self.items[meta.name] = meta
    def get(self, name: str) -> DatasetMeta | None:
        return self.items.get(name)
    def list(self) -> list[str]:
        return sorted(self.items)
