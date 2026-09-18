"""Aggregate latency stats per tool."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class ToolProfiler:
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_ms: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def record(self, name: str, latency_ms: float) -> None:
        self.counts[name] += 1
        self.total_ms[name] += latency_ms

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for k, n in self.counts.items():
            out[k] = {"count": n, "avg_ms": self.total_ms[k] / max(n, 1)}
        return out
