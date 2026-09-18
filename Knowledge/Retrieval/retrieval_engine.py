from __future__ import annotations
from dataclasses import dataclass
from Knowledge.Facts.fact_store import FactStore
from Knowledge.KnowledgeGraph.graph_store import GraphStore
@dataclass
class RetrievalEngine:
    facts: FactStore | None = None
    graph: GraphStore | None = None
    def retrieve(self, query: str, k: int = 5) -> list[str]:
        q = query.lower().split()
        hits = []
        if self.facts:
            for f in self.facts.facts.values():
                blob = ("%s %s %s" % (f.subject, f.predicate, f.object)).lower()
                score = sum(1 for w in q if w in blob)
                if score:
                    hits.append((score, "%s %s %s" % (f.subject, f.predicate, f.object)))
        if self.graph:
            for n in self.graph.nodes.values():
                blob = ("%s %s" % (n.id, n.label)).lower()
                score = sum(1 for w in q if w in blob)
                if score:
                    hits.append((score, "node:%s" % n.label))
        hits.sort(key=lambda x: -x[0])
        return [h for _, h in hits[:k]]
