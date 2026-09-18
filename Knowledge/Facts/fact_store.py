from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
@dataclass
class Fact:
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
@dataclass
class FactStore:
    facts: dict[str, Fact] = field(default_factory=dict)
    def add(self, fact: Fact) -> None:
        self.facts[fact.id] = fact
    def query(self, subject: str | None = None, predicate: str | None = None) -> list[Fact]:
        out = list(self.facts.values())
        if subject:
            out = [f for f in out if f.subject.lower() == subject.lower()]
        if predicate:
            out = [f for f in out if f.predicate.lower() == predicate.lower()]
        return out
