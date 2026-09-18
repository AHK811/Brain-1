from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
@dataclass
class KGNode:
    id: str
    label: str
    attrs: dict[str, Any] = field(default_factory=dict)
@dataclass
class KGEdge:
    src: str
    rel: str
    dst: str
    weight: float = 1.0
@dataclass
class GraphStore:
    nodes: dict[str, KGNode] = field(default_factory=dict)
    edges: list[KGEdge] = field(default_factory=list)
    def add_node(self, node: KGNode) -> None:
        self.nodes[node.id] = node
    def add_edge(self, edge: KGEdge) -> None:
        self.edges.append(edge)
    def neighbors(self, node_id: str) -> list[KGEdge]:
        return [e for e in self.edges if e.src == node_id or e.dst == node_id]
    def stats(self) -> dict[str, int]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}
