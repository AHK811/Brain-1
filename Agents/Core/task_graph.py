from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
@dataclass
class TaskNode:
    id: str
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"
    result: Any = None
@dataclass
class TaskGraph:
    nodes: dict[str, TaskNode] = field(default_factory=dict)
    def add(self, node: TaskNode) -> None:
        self.nodes[node.id] = node
    def ready(self) -> list[TaskNode]:
        return [n for n in self.nodes.values() if n.status=="pending" and all(self.nodes[d].status=="done" for d in n.depends_on if d in self.nodes)]
    def mark(self, id: str, status: str, result: Any=None) -> None:
        self.nodes[id].status = status; self.nodes[id].result = result
