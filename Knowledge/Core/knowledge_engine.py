from __future__ import annotations
from Knowledge.Facts.fact_store import FactStore, Fact
from Knowledge.KnowledgeGraph.graph_store import GraphStore, KGNode, KGEdge
from Knowledge.Retrieval.retrieval_engine import RetrievalEngine
class KnowledgeEngine:
    def __init__(self):
        self.facts = FactStore()
        self.graph = GraphStore()
        self.retrieval = RetrievalEngine(facts=self.facts, graph=self.graph)
    def assert_fact(self, id: str, s: str, p: str, o: str, confidence: float = 1.0, source: str = "") -> None:
        self.facts.add(Fact(id=id, subject=s, predicate=p, object=o, confidence=confidence, source=source))
        self.graph.add_node(KGNode(id=s, label=s))
        self.graph.add_node(KGNode(id=o, label=o))
        self.graph.add_edge(KGEdge(src=s, rel=p, dst=o, weight=confidence))
    def ask(self, query: str, k: int = 5) -> list[str]:
        return self.retrieval.retrieve(query, k=k)
